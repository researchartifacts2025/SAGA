"""Result analysis: metrics, statistics, and table rendering."""

from saga.analysis.metrics import (
    MetricSummary,
    aggregate_results,
    cache_hit_rate,
    competitive_ratio,
    geomean_speedup,
    memory_utilization,
    slo_attainment,
    summarize_tcts,
    throughput_per_minute,
)
from saga.analysis.stats import (
    bootstrap_ci,
    welch_t_test,
)
from saga.analysis.tables import (
    render_ablation_table,
    render_competitive_table,
    render_e2e_table,
    render_fairness_table,
    render_sensitivity_table,
)


__all__ = [
    "MetricSummary",
    "aggregate_results",
    "bootstrap_ci",
    "cache_hit_rate",
    "competitive_ratio",
    "geomean_speedup",
    "memory_utilization",
    "render_ablation_table",
    "render_competitive_table",
    "render_e2e_table",
    "render_fairness_table",
    "render_sensitivity_table",
    "slo_attainment",
    "summarize_tcts",
    "throughput_per_minute",
    "welch_t_test",
]
