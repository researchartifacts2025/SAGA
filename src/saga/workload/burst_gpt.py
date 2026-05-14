"""BurstGPT-derived synthetic multi-tenant workload.

10 tenants partitioned as:

  * 3 heavy tenants:  100-step agents arriving continuously (~16 t/m/tenant)
  * 4 medium tenants: 30-step agents arriving intermittently (~8 t/m/tenant)
  * 3 light tenants:  10-step agents arriving occasionally (~4 t/m/tenant)

Aggregate offered load is calibrated to ~80% of cluster peak throughput; the
exact mapping from offered load to per-worker pressure depends on cluster
size and is set by the cluster config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from saga.core.aeg import build_linear_aeg
from saga.core.types import Task, ToolType
from saga.utils.seeds import RNG
from saga.workload.base import (
    AgentTaskTemplate,
    WorkloadGenerator,
    WorkloadSpec,
    default_tool_plan,
)


@dataclass(frozen=True)
class TenantProfile:
    """Per-tenant intensity profile for the BurstGPT mix."""

    kind: str
    weight: float
    arrival_per_minute: float
    mean_steps: float
    max_steps: int
    prompt_min: int
    prompt_max: int
    output_min: int
    output_max: int


_DEFAULT_PROFILES: tuple[TenantProfile, ...] = (
    TenantProfile("heavy", 3.0, 16.0, 100.0, 200, 3_000, 5_000, 200, 600),
    TenantProfile("heavy", 3.0, 16.0, 100.0, 200, 3_000, 5_000, 200, 600),
    TenantProfile("heavy", 3.0, 16.0, 100.0, 200, 3_000, 5_000, 200, 600),
    TenantProfile("medium", 2.0, 8.0, 30.0, 80, 2_000, 4_000, 100, 400),
    TenantProfile("medium", 2.0, 8.0, 30.0, 80, 2_000, 4_000, 100, 400),
    TenantProfile("medium", 2.0, 8.0, 30.0, 80, 2_000, 4_000, 100, 400),
    TenantProfile("medium", 2.0, 8.0, 30.0, 80, 2_000, 4_000, 100, 400),
    TenantProfile("light", 1.0, 4.0, 10.0, 30, 1_500, 3_000, 50, 200),
    TenantProfile("light", 1.0, 4.0, 10.0, 30, 1_500, 3_000, 50, 200),
    TenantProfile("light", 1.0, 4.0, 10.0, 30, 1_500, 3_000, 50, 200),
)


@dataclass
class BurstGPTWorkload(WorkloadGenerator):
    """A multi-tenant arrival process sampled per-tenant."""

    name: str = "burst_gpt"
    profiles: tuple[TenantProfile, ...] = field(default_factory=lambda: _DEFAULT_PROFILES)

    def __init__(
        self,
        spec: WorkloadSpec | None = None,
        profiles: tuple[TenantProfile, ...] | None = None,
        horizon_minutes: float = 10.0,
    ) -> None:
        s = spec or WorkloadSpec(tag="burst_gpt", n_tenants=10)
        super().__init__(s)
        self.profiles = profiles or _DEFAULT_PROFILES
        self.horizon_minutes = horizon_minutes

    # The base ``stream()`` round-robins tenants which does not match the
    # per-tenant Poisson process we want; override.
    def stream(self):
        rng = RNG(self.spec.seed)
        events: list[tuple[float, AgentTaskTemplate]] = []
        horizon_ms = self.horizon_minutes * 60_000.0

        for tenant_idx, profile in enumerate(self.profiles):
            mean_gap_ms = 60_000.0 / max(1e-3, profile.arrival_per_minute)
            t = 0.0
            i = 0
            while t < horizon_ms:
                template = self._sample_for_profile(
                    rng.fork("burst", tenant_idx, i),
                    profile,
                    tenant_idx,
                    i,
                )
                template.task.submit_time = t
                template.tenant_weight = profile.weight
                events.append((t, template))
                i += 1
                t += rng.exponential(mean_gap_ms)

        events.sort(key=lambda x: x[0])
        for t, template in events:
            yield t, template

    def sample(self, rng: RNG, index: int, tenant_id: str) -> AgentTaskTemplate:
        # Used only when the base stream() is invoked; route by tenant prefix.
        idx = int(tenant_id.rsplit("_", 1)[-1]) % len(self.profiles)
        profile = self.profiles[idx]
        return self._sample_for_profile(rng, profile, idx, index)

    def _sample_for_profile(
        self,
        rng: RNG,
        profile: TenantProfile,
        tenant_idx: int,
        index: int,
    ) -> AgentTaskTemplate:
        n_steps = max(1, min(profile.max_steps, int(round(rng.gamma(2.0, profile.mean_steps / 2.0)))))
        tools = [ToolType.CODE_EXECUTION if rng.uniform() < 0.5 else ToolType.WEB_API for _ in range(n_steps)]
        prompt = rng.randint(profile.prompt_min, profile.prompt_max + 1)
        output = rng.randint(profile.output_min, profile.output_max + 1)
        obs = max(1, int(rng.gamma(2.0, 200.0)))

        aeg = build_linear_aeg(
            graph_id=f"burst/{tenant_idx}/{index}",
            n_steps=n_steps,
            tool_types=tools,
            prompt_tokens_est=prompt,
            output_tokens_est=output,
            observation_tokens_est=obs,
            termination_prob=1.0 / max(1.0, profile.mean_steps),
            workload_kind=f"burst_{profile.kind}",
        )

        tenant_id = f"tenant_{tenant_idx}"
        expected_tct_ms = profile.mean_steps * 1_200.0

        task = Task(
            task_id=f"{tenant_id}/burst/{index:04d}",
            tenant_id=tenant_id,
            workload_kind=f"burst_{profile.kind}",
            submit_time=0.0,
            n_steps=n_steps,
            aeg_id=aeg.graph_id,
            expected_tct_ms=expected_tct_ms,
            tokens_prefilled_initial=prompt,
        )
        tool_plan = default_tool_plan(aeg, rng.fork("tool", index))
        return AgentTaskTemplate(
            task=task,
            aeg=aeg,
            tool_plan=tool_plan,
            tenant_weight=profile.weight,
        )
