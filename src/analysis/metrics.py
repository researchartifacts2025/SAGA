"""Metric helpers operating on lists of ``SimulationResult``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from saga.core.types import Task
from saga.sim.engine import SimulationResult


@dataclass
class MetricSummary:
    """Sample mean/std/N + percentiles for one named metric."""

    name: str
    n: int
    mean: float
    std: float
    p50: float
    p95: float
    p99: float

    def format(self, fmt: str = "{:.1f}") -> str:
        return f"{fmt.format(self.mean)}±{fmt.format(self.std)}"


def summarize(name: str, values: Sequence[float]) -> MetricSummary:
    if not values:
        return MetricSummary(name=name, n=0, mean=0.0, std=0.0, p50=0.0, p95=0.0, p99=0.0)
    arr = np.asarray(values, dtype=np.float64)
    return MetricSummary(
        name=name,
        n=arr.size,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def summarize_tcts(result: SimulationResult) -> MetricSummary:
    return summarize("TCT_s", result.tct_seconds())


def cache_hit_rate(result: SimulationResult) -> float:
    return result.cache_hit_rate


def memory_utilization(result: SimulationResult) -> float:
    return result.mean_memory_utilization()


def throughput_per_minute(result: SimulationResult) -> float:
    return result.throughput_per_min()


def slo_attainment(result: SimulationResult, multiplier: float = 1.5) -> float:
    total = sum(1 for t in result.tasks if t.is_complete)
    if total == 0:
        return 0.0
    met = sum(1 for t in result.tasks if t.met_slo(multiplier))
    return met / total


def geomean_speedup(baseline_tcts: Sequence[float], ours_tcts: Sequence[float]) -> float:
    if not baseline_tcts or not ours_tcts:
        return 1.0
    b = max(np.mean(baseline_tcts), 1e-9)
    o = max(np.mean(ours_tcts), 1e-9)
    return float(b / o)


def competitive_ratio(policy_cost: float, optimal_cost: float) -> float:
    if optimal_cost <= 0.0:
        return float("inf") if policy_cost > 0.0 else 1.0
    return policy_cost / optimal_cost


# ----------------------------------------------------- aggregation


def aggregate_results(
    results_by_label: dict[str, list[SimulationResult]],
) -> dict[str, dict[str, MetricSummary]]:
    """Aggregate a dict ``{label: [SimulationResult, ...]}``.

    Returns ``{label: {"tct": MetricSummary, "mem": MetricSummary,
    "throughput": MetricSummary, "slo": MetricSummary}}``.
    """
    out: dict[str, dict[str, MetricSummary]] = {}
    for label, runs in results_by_label.items():
        tct_means = [float(np.mean(r.tct_seconds())) if r.tct_seconds() else 0.0 for r in runs]
        mem_means = [r.mean_memory_utilization() for r in runs]
        thr_means = [r.throughput_per_min() for r in runs]
        slo_vals = [slo_attainment(r) for r in runs]
        out[label] = {
            "tct": summarize("tct_s", tct_means),
            "mem": summarize("mem_frac", mem_means),
            "throughput": summarize("tpm", thr_means),
            "slo": summarize("slo", slo_vals),
        }
    return out


def per_tenant_slo(result: SimulationResult, multiplier: float = 1.5) -> dict[str, float]:
    by_tenant: dict[str, list[Task]] = {}
    for t in result.tasks:
        by_tenant.setdefault(t.tenant_id, []).append(t)
    out: dict[str, float] = {}
    for tenant, tasks in by_tenant.items():
        completed = [t for t in tasks if t.is_complete]
        if not completed:
            out[tenant] = 0.0
            continue
        met = sum(1 for t in completed if t.met_slo(multiplier))
        out[tenant] = met / len(completed)
    return out


def tenant_class_slo(
    result: SimulationResult,
    classifier: callable[[str], str] | None = None,
    multiplier: float = 1.5,
) -> dict[str, float]:
    """Group tenants into classes via ``classifier`` (default workload_kind)."""

    def _default(tid: str) -> str:
        for t in result.tasks:
            if t.tenant_id == tid:
                return t.workload_kind.replace("burst_", "") if t.workload_kind.startswith("burst_") else "all"
        return "all"

    fn = classifier or _default
    per = per_tenant_slo(result, multiplier)
    classes: dict[str, list[float]] = {}
    for tenant, val in per.items():
        classes.setdefault(fn(tenant), []).append(val)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in classes.items()}
