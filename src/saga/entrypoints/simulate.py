"""Run a single simulation under a Hydra configuration.

Usage:
    python -m saga.entrypoints.simulate
    python -m saga.entrypoints.simulate experiment=demo
    python -m saga.entrypoints.simulate +preset=vllm_apc workload=swe_bench

The Hydra config selects:
  * a workload (``configs/workload/*.yaml``)
  * a cluster (``configs/cluster/*.yaml``)
  * a scheduler preset (``configs/scheduler/*.yaml``)
  * an experiment overlay (``configs/experiment/*.yaml``)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from saga.analysis.metrics import (
    cache_hit_rate,
    memory_utilization,
    slo_attainment,
    summarize_tcts,
    throughput_per_minute,
)
from saga.presets import get_preset
from saga.scheduler.coordinator import CoordinatorConfig
from saga.sim.cluster import ClusterConfig
from saga.sim.engine import EngineConfig, SimulatorEngine
from saga.utils.logging import get_logger, setup_logging
from saga.workload import build_workload
from saga.workload.base import WorkloadSpec


log = get_logger("saga.simulate")
console = Console()


def _build_workload_spec(cfg: DictConfig) -> WorkloadSpec:
    return WorkloadSpec(
        n_tasks=int(cfg.workload.get("n_tasks", 100)),
        n_tenants=int(cfg.workload.get("n_tenants", 1)),
        arrival_rate_per_minute=float(cfg.workload.get("arrival_rate_per_minute", 8.0)),
        seed=int(cfg.seed),
        tag=str(cfg.workload.get("name", "generic")),
    )


def _apply_overrides(base: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(base, key):
            setattr(base, key, value)


def run_once(cfg: DictConfig) -> dict[str, Any]:
    """Build the engine, run the simulation, return a flat dict of results."""
    setup_logging(cfg.get("log_level", "INFO"))

    preset = get_preset(str(cfg.scheduler.preset))
    cluster_cfg: ClusterConfig = preset.cluster
    coord_cfg: CoordinatorConfig = preset.coordinator

    overrides = OmegaConf.to_container(cfg.cluster.get("overrides", {}), resolve=True) if cfg.get("cluster") else {}
    if isinstance(overrides, dict):
        _apply_overrides(cluster_cfg, overrides)
    sched_overrides = OmegaConf.to_container(cfg.scheduler.get("overrides", {}), resolve=True) if cfg.get("scheduler") else {}
    if isinstance(sched_overrides, dict):
        _apply_overrides(coord_cfg, sched_overrides)

    engine_cfg = EngineConfig(
        seed=int(cfg.seed),
        horizon_ms=float(cfg.get("horizon_ms", 600_000.0)),
        enable_speculative_prefetch=bool(cfg.scheduler.get("speculative_prefetch", True)),
        label=preset.label,
    )

    workload_name = str(cfg.workload.name)
    spec = _build_workload_spec(cfg)
    gen = build_workload(workload_name, spec=spec)
    templates = [tmpl for _t, tmpl in gen.stream()]

    engine = SimulatorEngine(cluster_cfg, coord_cfg, engine_cfg)
    engine.admit(templates)
    result = engine.run()

    summary = {
        "preset": preset.label,
        "workload": workload_name,
        "seed": engine_cfg.seed,
        "n_tasks": len(result.tasks),
        "n_completed": sum(1 for t in result.tasks if t.is_complete),
        "tct_seconds": summarize_tcts(result).__dict__,
        "memory_utilization": memory_utilization(result),
        "throughput_per_minute": throughput_per_minute(result),
        "slo_attainment": slo_attainment(result, multiplier=float(cfg.get("slo_multiplier", 1.5))),
        "cache_hit_rate": cache_hit_rate(result),
        "regenerated_tokens": result.regenerated_tokens,
        "tokens_admitted": result.tokens_admitted,
        "regeneration_ratio": result.regeneration_ratio,
        "n_evictions": result.n_evictions,
        "n_steals": result.n_steals,
        "n_migrations": result.n_migrations,
        "sim_time_ms": result.sim_time_ms,
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title=f"Simulation: {summary['preset']} on {summary['workload']}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    tct = summary["tct_seconds"]
    table.add_row("Tasks completed", f"{summary['n_completed']} / {summary['n_tasks']}")
    table.add_row("TCT mean (s)", f"{tct['mean']:.1f} ± {tct['std']:.1f}")
    table.add_row("TCT P95 (s)", f"{tct['p95']:.1f}")
    table.add_row("Memory utilization", f"{summary['memory_utilization'] * 100:.1f} %")
    table.add_row("Throughput (tasks/min)", f"{summary['throughput_per_minute']:.2f}")
    table.add_row("SLO attainment", f"{summary['slo_attainment'] * 100:.1f} %")
    table.add_row("Cache hit rate", f"{summary['cache_hit_rate'] * 100:.1f} %")
    table.add_row("Regen ratio", f"{summary['regeneration_ratio']:.3f}")
    table.add_row("Steals / migrations", f"{summary['n_steals']} / {summary['n_migrations']}")

    console.print(table)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    summary = run_once(cfg)
    _print_summary(summary)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    console.print(f"\n[green]Wrote[/green] {out_dir / 'summary.json'}")


def _direct() -> int:
    """Direct invocation path for ``python -m saga.entrypoints.simulate``."""
    if "--help" in sys.argv:
        main()
        return 0
    main()
    return 0


if __name__ == "__main__":
    sys.exit(_direct())
