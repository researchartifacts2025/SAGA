"""Run the full benchmark suite.

The benchmark sweeps over (preset, workload, seed) and writes a JSON result
file plus a markdown table for each experiment.

Supported ``experiment=`` values (from ``configs/experiment/``):

  * ``e2e_main``     --- e2e SWE-bench + WebArena across all presets.
  * ``ablation``     --- SAGA with each component removed.
  * ``fairness``     --- multi-tenant BurstGPT, SLO by tenant class.
  * ``competitive``  --- WA-LRU vs LRU vs LRU+Prefix vs Belady oracle.
  * ``sensitivity``  --- single-axis parameter sweeps.
  * ``demo``         --- a 60-second sanity run.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console

from saga.analysis.metrics import (
    cache_hit_rate,
    memory_utilization,
    slo_attainment,
    summarize_tcts,
    throughput_per_minute,
)
from saga.analysis.tables import (
    render_ablation_table,
    render_competitive_table,
    render_e2e_table,
    render_fairness_table,
    render_sensitivity_table,
)
from saga.presets import get_preset
from saga.scheduler.coordinator import CoordinatorConfig
from saga.sim.cluster import ClusterConfig
from saga.sim.engine import EngineConfig, SimulationResult, SimulatorEngine
from saga.utils.logging import get_logger, setup_logging
from saga.workload import build_workload
from saga.workload.base import WorkloadSpec


log = get_logger("saga.benchmark")
console = Console()


# --------------------------------------------------------- runners


def _run_combo(
    preset_name: str,
    workload_name: str,
    workload_spec: WorkloadSpec,
    seed: int,
    cluster_overrides: dict[str, Any] | None = None,
    coord_overrides: dict[str, Any] | None = None,
    horizon_ms: float = 1_200_000.0,
    label_suffix: str = "",
) -> SimulationResult:
    preset = get_preset(preset_name)
    cluster: ClusterConfig = deepcopy(preset.cluster)
    coord: CoordinatorConfig = deepcopy(preset.coordinator)
    if cluster_overrides:
        for k, v in cluster_overrides.items():
            setattr(cluster, k, v)
    if coord_overrides:
        for k, v in coord_overrides.items():
            setattr(coord, k, v)

    label = preset.label + (f":{label_suffix}" if label_suffix else "")
    engine_cfg = EngineConfig(
        seed=seed,
        horizon_ms=horizon_ms,
        enable_speculative_prefetch=preset_name not in {"saga_no_prefetch"},
        label=label,
    )
    spec = WorkloadSpec(
        n_tasks=workload_spec.n_tasks,
        n_tenants=workload_spec.n_tenants,
        arrival_rate_per_minute=workload_spec.arrival_rate_per_minute,
        seed=seed,
        tag=workload_spec.tag,
        tenant_weights=list(workload_spec.tenant_weights),
    )
    gen = build_workload(workload_name, spec=spec)
    templates = [tmpl for _t, tmpl in gen.stream()]

    engine = SimulatorEngine(cluster, coord, engine_cfg)
    engine.admit(templates)
    return engine.run()


# --------------------------------------------------------- e2e


def _e2e(cfg: DictConfig, out_dir: Path) -> None:
    workloads_cfg = cfg.experiment.workloads
    presets = list(cfg.experiment.presets)
    n_seeds = int(cfg.experiment.n_seeds)
    seeds = list(cfg.experiment.get("seeds", list(range(n_seeds))))
    horizon_ms = float(cfg.get("horizon_ms", 1_200_000.0))

    results_by_workload: dict[str, dict[str, list[SimulationResult]]] = {}
    for w_cfg in workloads_cfg:
        wname = w_cfg.name
        spec = WorkloadSpec(
            n_tasks=int(w_cfg.n_tasks),
            n_tenants=int(w_cfg.get("n_tenants", 1)),
            arrival_rate_per_minute=float(w_cfg.get("arrival_rate_per_minute", 8.0)),
            seed=0,
            tag=wname,
        )
        results_by_workload[wname] = {}
        for preset in presets:
            runs: list[SimulationResult] = []
            for seed in seeds:
                console.log(f"  e2e: preset={preset:<22} workload={wname:<10} seed={seed}")
                r = _run_combo(preset, wname, spec, seed, horizon_ms=horizon_ms)
                runs.append(r)
            results_by_workload[wname][preset] = runs

    table = render_e2e_table(results_by_workload, ours_label="saga")
    console.print(table)
    (out_dir / "e2e.md").write_text(table)
    (out_dir / "e2e.json").write_text(json.dumps(_serialize(results_by_workload), indent=2, default=float))


# --------------------------------------------------------- ablation


def _ablation(cfg: DictConfig, out_dir: Path) -> None:
    presets = list(cfg.experiment.presets)
    spec = WorkloadSpec(
        n_tasks=int(cfg.experiment.workload.n_tasks),
        arrival_rate_per_minute=float(cfg.experiment.workload.get("arrival_rate_per_minute", 8.0)),
        seed=0,
        tag="swe_bench",
    )
    seeds = list(cfg.experiment.get("seeds", list(range(int(cfg.experiment.n_seeds)))))

    results: dict[str, list[SimulationResult]] = {}
    for preset in presets:
        runs: list[SimulationResult] = []
        for seed in seeds:
            console.log(f"  ablation: preset={preset:<24} seed={seed}")
            r = _run_combo(preset, cfg.experiment.workload.name, spec, seed)
            runs.append(r)
        results[preset] = runs

    table = render_ablation_table(results, full_label="saga")
    console.print(table)
    (out_dir / "ablation.md").write_text(table)
    (out_dir / "ablation.json").write_text(json.dumps(_serialize({"swe": results}), indent=2, default=float))


# --------------------------------------------------------- fairness


def _fairness(cfg: DictConfig, out_dir: Path) -> None:
    presets = list(cfg.experiment.presets)
    spec = WorkloadSpec(
        n_tasks=int(cfg.experiment.workload.get("n_tasks", 200)),
        n_tenants=int(cfg.experiment.workload.get("n_tenants", 10)),
        arrival_rate_per_minute=float(cfg.experiment.workload.get("arrival_rate_per_minute", 8.0)),
        seed=0,
        tag="burst_gpt",
    )
    seeds = list(cfg.experiment.get("seeds", list(range(int(cfg.experiment.n_seeds)))))

    results: dict[str, list[SimulationResult]] = {}
    for preset in presets:
        runs: list[SimulationResult] = []
        for seed in seeds:
            console.log(f"  fairness: preset={preset:<24} seed={seed}")
            r = _run_combo(preset, "burst_gpt", spec, seed)
            runs.append(r)
        results[preset] = runs

    table = render_fairness_table(results)
    console.print(table)
    (out_dir / "fairness.md").write_text(table)
    (out_dir / "fairness.json").write_text(json.dumps(_serialize({"burst": results}), indent=2, default=float))


# --------------------------------------------------------- competitive


def _competitive(cfg: DictConfig, out_dir: Path) -> None:
    policies = ["lru", "lru_prefix", "walru", "belady"]
    workloads = list(cfg.experiment.workloads)
    spec_factories: dict[str, WorkloadSpec] = {}
    for w in workloads:
        spec_factories[w.name] = WorkloadSpec(
            n_tasks=int(w.n_tasks),
            arrival_rate_per_minute=float(w.get("arrival_rate_per_minute", 8.0)),
            seed=0,
            tag=w.name,
        )
    seeds = list(cfg.experiment.get("seeds", list(range(int(cfg.experiment.n_seeds)))))

    ratios: dict[str, dict[str, float]] = {p: {} for p in policies}
    oracle_costs: dict[str, list[float]] = {}

    for w_name, spec in spec_factories.items():
        oracle_costs[w_name] = []
        for seed in seeds:
            r_oracle = _run_combo(
                "saga",
                w_name,
                spec,
                seed,
                cluster_overrides={"eviction_policy": "belady"},
                label_suffix="oracle",
            )
            oracle_costs[w_name].append(max(1.0, r_oracle.regenerated_tokens))

        for policy in policies:
            costs: list[float] = []
            for seed_idx, seed in enumerate(seeds):
                r = _run_combo(
                    "saga",
                    w_name,
                    spec,
                    seed,
                    cluster_overrides={"eviction_policy": policy},
                    label_suffix=policy,
                )
                cost = max(1.0, r.regenerated_tokens)
                opt = oracle_costs[w_name][seed_idx]
                costs.append(cost / opt)
            mean_ratio = float(np.mean(costs)) if costs else 1.0
            ratios[policy][w_name] = mean_ratio

    table = render_competitive_table(ratios)
    console.print(table)
    (out_dir / "competitive.md").write_text(table)
    (out_dir / "competitive.json").write_text(json.dumps(ratios, indent=2, default=float))


# --------------------------------------------------------- sensitivity


def _sensitivity(cfg: DictConfig, out_dir: Path) -> None:
    workload_name = str(cfg.experiment.workload.name)
    base_spec = WorkloadSpec(
        n_tasks=int(cfg.experiment.workload.n_tasks),
        arrival_rate_per_minute=float(cfg.experiment.workload.get("arrival_rate_per_minute", 8.0)),
        seed=0,
        tag=workload_name,
    )
    seeds = list(cfg.experiment.get("seeds", list(range(int(cfg.experiment.n_seeds)))))
    sweeps_cfg = cfg.experiment.sweeps

    sweeps: dict[str, dict[float, list[SimulationResult]]] = {}
    for param_name, values in sweeps_cfg.items():
        sweeps[param_name] = {}
        for val in values:
            runs: list[SimulationResult] = []
            for seed in seeds:
                cluster_ov: dict[str, Any] = {}
                coord_ov: dict[str, Any] = {}
                if param_name in {"walru_alpha", "walru_beta", "walru_gamma"}:
                    cluster_ov[param_name] = float(val)
                elif param_name in {"load_threshold", "t_idle_ms", "r_max"}:
                    coord_ov[param_name] = float(val)
                elif param_name in {"ttl_max_ms", "pressure_low", "pressure_high"}:
                    cluster_ov[param_name] = float(val)
                else:
                    cluster_ov[param_name] = float(val)

                console.log(f"  sensitivity: {param_name}={val} seed={seed}")
                r = _run_combo(
                    "saga",
                    workload_name,
                    base_spec,
                    seed,
                    cluster_overrides=cluster_ov,
                    coord_overrides=coord_ov,
                )
                runs.append(r)
            sweeps[param_name][float(val)] = runs

    table = render_sensitivity_table(sweeps)
    console.print(table)
    (out_dir / "sensitivity.md").write_text(table)
    (out_dir / "sensitivity.json").write_text(
        json.dumps(
            {k: {str(v): _serialize_runs(runs) for v, runs in sw.items()} for k, sw in sweeps.items()},
            indent=2,
            default=float,
        )
    )


# --------------------------------------------------------- bfsdfs


def _bfsdfs(cfg: DictConfig, out_dir: Path) -> None:
    spec = WorkloadSpec(
        n_tasks=int(cfg.experiment.workload.n_tasks),
        arrival_rate_per_minute=float(cfg.experiment.workload.get("arrival_rate_per_minute", 8.0)),
        seed=0,
        tag="swe_bench",
    )
    strategies = list(cfg.experiment.strategies)
    seeds = list(cfg.experiment.get("seeds", list(range(int(cfg.experiment.n_seeds)))))

    rows: list[list[Any]] = []
    for strat in strategies:
        tcts: list[float] = []
        throughputs: list[float] = []
        evict_rates: list[float] = []
        for seed in seeds:
            console.log(f"  bfsdfs: strategy={strat:<8} seed={seed}")
            r = _run_combo(
                "saga",
                "swe_bench",
                spec,
                seed,
                coord_overrides={"queue_strategy": strat},
            )
            tct_seq = r.tct_seconds()
            if tct_seq:
                tcts.append(float(np.mean(tct_seq)))
            throughputs.append(r.throughput_per_min())
            if r.tokens_admitted > 0:
                evict_rates.append(r.n_evictions / max(1, r.n_cache_admits))
        rows.append(
            [
                strat.upper(),
                f"{float(np.mean(tcts)):.1f}±{float(np.std(tcts)):.1f}" if tcts else "-",
                f"{float(np.mean(throughputs)):.2f} t/m",
                f"{float(np.mean(evict_rates)) * 100:.0f}%" if evict_rates else "-",
            ]
        )

    from tabulate import tabulate as _tab

    table = "## Execution-Strategy Tradeoff\n\n" + _tab(
        rows,
        headers=["Strategy", "TCT (s)", "Throughput", "Evict Rate"],
        tablefmt="github",
    )
    console.print(table)
    (out_dir / "bfsdfs.md").write_text(table)


# --------------------------------------------------------- tool variance


def _tool_variance(cfg: DictConfig, out_dir: Path) -> None:
    base_spec = WorkloadSpec(
        n_tasks=int(cfg.experiment.workload.n_tasks),
        arrival_rate_per_minute=float(cfg.experiment.workload.get("arrival_rate_per_minute", 8.0)),
        seed=0,
        tag="swe_bench",
    )
    cvs = list(cfg.experiment.coefficient_of_variation)
    seeds = list(cfg.experiment.get("seeds", list(range(int(cfg.experiment.n_seeds)))))

    from saga.workload.base import _tool_variance_override  # late import

    rows: list[list[Any]] = []
    for cv in cvs:
        cv_f = float(cv)
        _tool_variance_override.set(cv_f)
        tcts: list[float] = []
        ttl_acc: list[float] = []
        evict_rates: list[float] = []
        for seed in seeds:
            console.log(f"  tool_variance: cv={cv_f} seed={seed}")
            r = _run_combo("saga", "swe_bench", base_spec, seed)
            ts = r.tct_seconds()
            if ts:
                tcts.append(float(np.mean(ts)))
            ttl_acc.append(r.cache_hit_rate)
            if r.n_cache_admits > 0:
                evict_rates.append(r.n_evictions / r.n_cache_admits)
        rows.append(
            [
                f"{cv_f:g}",
                f"{float(np.mean(tcts)):.1f}±{float(np.std(tcts)):.1f}" if tcts else "-",
                f"{float(np.mean(ttl_acc)) * 100:.0f}%",
                f"{float(np.mean(evict_rates)) * 100:.0f}%" if evict_rates else "-",
            ]
        )
    _tool_variance_override.set(None)

    from tabulate import tabulate as _tab

    table = "## Tool-Latency Variance Sensitivity\n\n" + _tab(
        rows,
        headers=["CV", "TCT (s)", "TTL Accuracy", "Evict Rate"],
        tablefmt="github",
    )
    console.print(table)
    (out_dir / "tool_variance.md").write_text(table)


# --------------------------------------------------------- demo


def _demo(cfg: DictConfig, out_dir: Path) -> None:
    spec = WorkloadSpec(
        n_tasks=20,
        n_tenants=1,
        arrival_rate_per_minute=8.0,
        seed=0,
        tag="swe_bench",
    )
    runs: dict[str, list[SimulationResult]] = {}
    for preset in ["vllm", "vllm_apc", "saga"]:
        r = _run_combo(preset, "swe_bench", spec, 0, horizon_ms=600_000.0)
        runs[preset] = [r]
    table = render_e2e_table({"swe_bench": runs}, ours_label="saga")
    console.print(table)
    (out_dir / "demo.md").write_text(table)


# --------------------------------------------------------- helpers


def _serialize(by_w: dict[str, dict[str, list[SimulationResult]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for w, by_p in by_w.items():
        out[w] = {p: _serialize_runs(runs) for p, runs in by_p.items()}
    return out


def _serialize_runs(runs: list[SimulationResult]) -> list[dict[str, Any]]:
    return [
        {
            "label": r.config_label,
            "seed": r.seed,
            "sim_time_ms": r.sim_time_ms,
            "n_tasks": len(r.tasks),
            "n_completed": sum(1 for t in r.tasks if t.is_complete),
            "tct_mean_s": summarize_tcts(r).mean,
            "tct_std_s": summarize_tcts(r).std,
            "memory_utilization": memory_utilization(r),
            "throughput_per_minute": throughput_per_minute(r),
            "slo_attainment": slo_attainment(r),
            "cache_hit_rate": cache_hit_rate(r),
            "regenerated_tokens": r.regenerated_tokens,
            "tokens_admitted": r.tokens_admitted,
            "n_evictions": r.n_evictions,
            "n_steals": r.n_steals,
            "n_migrations": r.n_migrations,
        }
        for r in runs
    ]


# --------------------------------------------------------- main


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.get("log_level", "INFO"))
    if "experiment" not in cfg:
        console.print("[red]No experiment selected[/red]; pass experiment=e2e_main (or ablation/...)")
        return
    exp_name = cfg.experiment.get("name", "unknown")
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    console.rule(f"[bold cyan]Benchmark: {exp_name}[/bold cyan]")

    if exp_name in {"e2e_main", "e2e"}:
        _e2e(cfg, out_dir)
    elif exp_name == "ablation":
        _ablation(cfg, out_dir)
    elif exp_name == "fairness":
        _fairness(cfg, out_dir)
    elif exp_name == "competitive":
        _competitive(cfg, out_dir)
    elif exp_name == "sensitivity":
        _sensitivity(cfg, out_dir)
    elif exp_name == "bfsdfs":
        _bfsdfs(cfg, out_dir)
    elif exp_name == "tool_variance":
        _tool_variance(cfg, out_dir)
    elif exp_name == "demo":
        _demo(cfg, out_dir)
    else:
        console.print(f"[red]Unknown experiment {exp_name!r}[/red]")
        return

    console.print(f"\n[green]Outputs in[/green] {out_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
