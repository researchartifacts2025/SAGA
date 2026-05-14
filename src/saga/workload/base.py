"""Workload-generator base classes.

A workload produces a stream of ``(arrival_time_ms, Task, AEG, ToolPlan)``
tuples. The engine uses ``arrival_time_ms`` to schedule arrival events, the
``Task`` for accounting, the ``AEG`` for workflow-aware policies, and the
``ToolPlan`` (a per-node tool-call duration) as the ground-truth tool
durations the simulator will inject.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from saga.core.aeg import AgentExecutionGraph
from saga.core.types import Task, ToolType
from saga.utils.seeds import RNG


@dataclass
class ToolPlan:
    """Ground-truth tool-call durations for each step of a task.

    ``durations_ms[i]`` is the wall-clock duration of the tool call following
    the i-th LLM step. ``observation_tokens[i]`` is the number of tokens the
    tool result adds to the context.
    """

    durations_ms: list[float]
    observation_tokens: list[int]


@dataclass
class AgentTaskTemplate:
    """A complete task description ready for the engine to admit."""

    task: Task
    aeg: AgentExecutionGraph
    tool_plan: ToolPlan
    tenant_weight: float = 1.0


@dataclass
class WorkloadSpec:
    """Configuration knobs shared across workloads."""

    n_tasks: int = 100
    n_tenants: int = 1
    arrival_rate_per_minute: float = 8.0
    tenant_weights: list[float] = field(default_factory=list)
    seed: int = 42
    tag: str = "generic"


class WorkloadGenerator(ABC):
    """Abstract workload generator."""

    name: str = "generic"

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec

    @abstractmethod
    def sample(self, rng: RNG, index: int, tenant_id: str) -> AgentTaskTemplate:
        """Sample a single task template."""

    # ------------------------------------------------------- driver

    def stream(self) -> Iterator[tuple[float, AgentTaskTemplate]]:
        """Yield ``(arrival_time_ms, template)`` pairs in submit order."""
        rng = RNG(self.spec.seed)
        weights = list(self.spec.tenant_weights) or [1.0] * self.spec.n_tenants
        if len(weights) != self.spec.n_tenants:
            weights = (weights * self.spec.n_tenants)[: self.spec.n_tenants]
        tenants = [f"tenant_{i}" for i in range(self.spec.n_tenants)]

        per_minute = max(0.001, self.spec.arrival_rate_per_minute)
        mean_gap_ms = 60_000.0 / per_minute

        t = 0.0
        for i in range(self.spec.n_tasks):
            tenant_idx = i % self.spec.n_tenants
            tenant_id = tenants[tenant_idx]
            template = self.sample(rng.fork("task", i, tenant_id), i, tenant_id)
            template.task.submit_time = t
            template.tenant_weight = weights[tenant_idx]
            yield t, template
            t += rng.exponential(mean_gap_ms)


class _ToolVarianceOverride:
    """Process-wide override for the tool-latency coefficient of variation.

    Used by the tool-variance experiment to sweep CV while keeping the mean
    latency fixed. ``None`` means "use the defaults from the latency table".
    """

    def __init__(self) -> None:
        self._cv: float | None = None

    def set(self, cv: float | None) -> None:
        self._cv = cv

    def get(self) -> float | None:
        return self._cv


_tool_variance_override = _ToolVarianceOverride()


def default_tool_plan(
    aeg: AgentExecutionGraph,
    rng: RNG,
) -> ToolPlan:
    """Sample ground-truth tool durations consistent with the AEG.

    For each node, draw the tool's realized duration from a log-normal whose
    parameters come from the (cold) defaults baked into the latency table.
    When :data:`_tool_variance_override` is set, replace the default sigma
    with one that yields the requested coefficient of variation while
    holding the mean constant.
    """
    import math

    from saga.cache.ttl import (
        _TOOL_LATENCY_DEFAULTS,
        fit_lognormal_from_percentiles,
    )

    cv_override = _tool_variance_override.get()

    durations: list[float] = []
    obs_tokens: list[int] = []
    for node in aeg.nodes:
        if node.tool_type == ToolType.NONE:
            durations.append(0.0)
            obs_tokens.append(node.observation_tokens_est)
            continue
        p50, p95, _p99 = _TOOL_LATENCY_DEFAULTS.get(node.tool_type, (200.0, 1_000.0, 5_000.0))
        mu, sigma = fit_lognormal_from_percentiles(p50, p95)
        if cv_override is not None and cv_override >= 0.0:
            sigma_target = math.sqrt(math.log1p(cv_override * cv_override))
            mean = math.exp(mu + 0.5 * sigma * sigma)
            mu = math.log(max(mean, 1e-6)) - 0.5 * sigma_target * sigma_target
            sigma = sigma_target
        d = max(1.0, rng.lognormal(mu, sigma))
        durations.append(d)
        obs_tokens.append(node.observation_tokens_est)
    return ToolPlan(durations_ms=durations, observation_tokens=obs_tokens)
