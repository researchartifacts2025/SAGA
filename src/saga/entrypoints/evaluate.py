"""Aggregate prior benchmark JSON outputs into combined tables.

Usage:
    python -m saga.entrypoints.evaluate input_dir=runs/2026-05-14
    python -m saga.entrypoints.evaluate input_glob="runs/**/e2e.json"
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from rich.console import Console
from tabulate import tabulate


console = Console()


def _find_files(root_glob: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(root_glob, recursive=True))]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _flatten_e2e(payloads: list[dict[str, Any]]) -> list[list[Any]]:
    """Turn a list of e2e.json payloads into a flat tabular form."""
    rows: list[list[Any]] = []
    for payload in payloads:
        for workload, by_preset in payload.items():
            for preset, runs in by_preset.items():
                if not isinstance(runs, list):
                    continue
                for run in runs:
                    rows.append(
                        [
                            workload,
                            preset,
                            run.get("seed"),
                            round(run.get("tct_mean_s", 0.0), 1),
                            round(run.get("memory_utilization", 0.0) * 100.0, 1),
                            round(run.get("throughput_per_minute", 0.0), 2),
                            round(run.get("slo_attainment", 0.0) * 100.0, 1),
                            round(run.get("cache_hit_rate", 0.0) * 100.0, 1),
                        ]
                    )
    return rows


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    root = str(cfg.get("input_glob", "runs/**/e2e.json"))
    files = _find_files(root)
    console.rule(f"[bold cyan]Evaluating {len(files)} run(s)[/bold cyan]")
    if not files:
        console.print(f"[yellow]No files matched[/yellow] {root}")
        return

    payloads = [_load_json(p) for p in files]
    rows = _flatten_e2e(payloads)
    header = ["Workload", "Preset", "Seed", "TCT(s)", "Mem%", "Throughput", "SLO%", "Hit%"]
    table = tabulate(rows, headers=header, tablefmt="github")
    console.print(table)
    out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "evaluate.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    console.print(f"\n[green]Wrote[/green] {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
