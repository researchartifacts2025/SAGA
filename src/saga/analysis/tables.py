"""Pretty-printed result tables.

Render functions accept a dict of ``{label: list[SimulationResult]}`` and
return a markdown-style table as a string (suitable for stdout, README, or
docs). Numeric formatting uses the same decimal precision as the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tabulate import tabulate

from saga.analysis.metrics import (
    aggregate_results,
    geomean_speedup,
    slo_attainment,
    summarize,
    tenant_class_slo,
)
from saga.analysis.stats import welch_t_test
from saga.sim.engine import SimulationResult


# --------------------------------------------------------- e2e


def render_e2e_table(
    results: dict[str, dict[str, list[SimulationResult]]],
    ours_label: str = "saga",
    baseline_for_geomean: str = "vllm_apc",
    fmt: str = "github",
) -> str:
    """Render the SWE-bench / WebArena end-to-end comparison."""
    workloads = list(results.keys())
    labels = sorted({lab for w in results.values() for lab in w})

    header = ["System"]
    for w in workloads:
        header += [f"{w} TCT (s)", f"{w} Mem%"]
    header += ["Speedup vs ours"]

    rows: list[list[object]] = []
    for label in labels:
        row: list[object] = [label]
        for w in workloads:
            runs = results[w].get(label, [])
            agg = aggregate_results({label: runs}).get(label, {})
            tct = agg.get("tct")
            mem = agg.get("mem")
            row.append(tct.format("{:.1f}") if tct else "-")
            row.append(f"{mem.mean * 100:.1f}" if mem else "-")
        ours_runs = results[workloads[0]].get(ours_label, [])
        my_runs = results[workloads[0]].get(label, [])
        if label == ours_label or not ours_runs or not my_runs:
            row.append("--")
        else:
            base = [np.mean(r.tct_seconds()) for r in my_runs if r.tct_seconds()]
            ours = [np.mean(r.tct_seconds()) for r in ours_runs if r.tct_seconds()]
            if base and ours:
                speedup = geomean_speedup(base, ours)
                test = welch_t_test(base, ours)
                row.append(f"{speedup:.2f}x {test.stars}")
            else:
                row.append("--")
        rows.append(row)

    title = "## End-to-End Performance\n\n"
    geomean_line = ""
    ours_per_w = []
    base_per_w = []
    for w in workloads:
        ours_runs = results[w].get(ours_label, [])
        base_runs = results[w].get(baseline_for_geomean, [])
        if ours_runs and base_runs:
            ours_mean = float(
                np.mean([np.mean(r.tct_seconds()) for r in ours_runs if r.tct_seconds()])
            )
            base_mean = float(
                np.mean([np.mean(r.tct_seconds()) for r in base_runs if r.tct_seconds()])
            )
            if ours_mean > 0:
                ours_per_w.append(ours_mean)
                base_per_w.append(base_mean)
    if ours_per_w:
        speedups = [b / o for b, o in zip(base_per_w, ours_per_w, strict=True)]
        from saga.analysis.stats import geomean

        gm = geomean(speedups)
        geomean_line = f"\nGeometric mean speedup vs `{baseline_for_geomean}`: **{gm:.2f}x**\n"

    return title + tabulate(rows, headers=header, tablefmt=fmt) + "\n" + geomean_line


# ------------------------------------------------------ ablation


def render_ablation_table(
    results: dict[str, list[SimulationResult]],
    full_label: str = "saga",
    fmt: str = "github",
) -> str:
    """Render ablation rows showing % slowdown when each component is removed."""
    if full_label not in results:
        raise ValueError(f"missing baseline {full_label!r} for ablation")

    full_tcts = [np.mean(r.tct_seconds()) for r in results[full_label] if r.tct_seconds()]
    if not full_tcts:
        raise ValueError("baseline runs produced no completed tasks")
    full_mean = float(np.mean(full_tcts))

    rows: list[list[object]] = []
    for label, runs in results.items():
        tcts = [np.mean(r.tct_seconds()) for r in runs if r.tct_seconds()]
        summary = summarize("tct", tcts)
        delta = (summary.mean / full_mean - 1.0) * 100.0 if full_mean > 0 else 0.0
        marker = "(baseline)" if label == full_label else f"{delta:+.1f}%"
        rows.append(
            [
                label,
                f"{summary.mean:.1f}±{summary.std:.1f}",
                marker,
            ]
        )

    rows.sort(
        key=lambda r: (
            0.0
            if r[0] == full_label
            else float(r[2].rstrip("%").replace("+", ""))
            if r[2] != "(baseline)"
            else 0.0
        )
    )
    header = ["Configuration", "TCT (s)", "vs Full"]
    return "## Ablation\n\n" + tabulate(rows, headers=header, tablefmt=fmt)


# --------------------------------------------------- competitive


def render_competitive_table(
    ratios: dict[str, dict[str, float]],
    fmt: str = "github",
) -> str:
    """Render the competitive-ratio table against Belady's oracle.

    ``ratios = {policy_name: {workload_name: ratio, ...}, ...}``.
    """
    workloads = sorted({w for r in ratios.values() for w in r})
    rows: list[list[object]] = []
    for policy, vals in ratios.items():
        row: list[object] = [policy]
        for w in workloads:
            v = vals.get(w)
            row.append(f"{v:.2f}" if v is not None else "-")
        means = [v for v in vals.values() if v is not None]
        row.append(f"{float(np.mean(means)):.2f}" if means else "-")
        rows.append(row)

    header = ["Policy"] + workloads + ["Mean"]
    return "## Competitive Ratio vs Belady's Optimal\n\n" + tabulate(
        rows, headers=header, tablefmt=fmt
    )


# ---------------------------------------------------- fairness


def render_fairness_table(
    results: dict[str, list[SimulationResult]],
    fmt: str = "github",
) -> str:
    """Render multi-tenant SLO attainment by class (heavy/medium/light)."""

    def class_of(tenant_id: str, runs: Sequence[SimulationResult]) -> str:
        for r in runs:
            for task in r.tasks:
                if task.tenant_id == tenant_id and task.workload_kind.startswith("burst_"):
                    return task.workload_kind.replace("burst_", "")
        return "all"

    rows: list[list[object]] = []
    classes = ["heavy", "medium", "light", "all"]

    for label, runs in results.items():
        cls_vals: dict[str, list[float]] = {c: [] for c in classes}
        for r in runs:
            per_class = tenant_class_slo(
                r,
                classifier=lambda tid, _r=r: class_of(tid, [_r]),
            )
            for c in classes[:-1]:
                if c in per_class:
                    cls_vals[c].append(per_class[c] * 100.0)
            cls_vals["all"].append(slo_attainment(r) * 100.0)

        row: list[object] = [label]
        for c in classes:
            vals = cls_vals[c]
            row.append(f"{float(np.mean(vals)):.1f}" if vals else "-")
        rows.append(row)

    header = ["System", "Heavy %", "Medium %", "Light %", "Overall %"]
    return "## Multi-Tenant SLO Attainment\n\n" + tabulate(rows, headers=header, tablefmt=fmt)


# --------------------------------------------------- sensitivity


def render_sensitivity_table(
    sweeps: dict[str, dict[float, list[SimulationResult]]],
    fmt: str = "github",
) -> str:
    """Render a parameter-sensitivity table.

    ``sweeps = {param_name: {param_value: [SimulationResult, ...], ...}, ...}``.
    """
    rows: list[list[object]] = []
    for param, by_value in sweeps.items():
        base_tcts: list[float] = []
        for vals in by_value.values():
            base_tcts.extend(np.mean(r.tct_seconds()) for r in vals if r.tct_seconds())
        baseline = float(np.mean(base_tcts)) if base_tcts else 1.0

        max_delta = 0.0
        for vals in by_value.values():
            tcts = [np.mean(r.tct_seconds()) for r in vals if r.tct_seconds()]
            if not tcts:
                continue
            m = float(np.mean(tcts))
            delta = abs(m - baseline) / max(baseline, 1e-9) * 100.0
            max_delta = max(max_delta, delta)
        values_str = ", ".join(f"{v:g}" for v in sorted(by_value))
        rows.append([param, values_str, f"<{max_delta:.1f}%"])

    header = ["Parameter", "Tested Range", "Max TCT delta"]
    return "## Parameter Sensitivity\n\n" + tabulate(rows, headers=header, tablefmt=fmt)
