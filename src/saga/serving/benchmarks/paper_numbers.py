"""Load the canonical paper numbers from ``results/paper.yaml``.

When the host does not have the 64-A100 cluster, benchmark consumers
(README scripts, docs builders, CI) load this object instead of measuring
wall-clock TCT. The accessor signatures mirror :class:`WallClockResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_PAPER_YAML = Path(__file__).resolve().parents[4] / "results" / "paper.yaml"


@dataclass
class PaperResults:
    """Read-only view over the canonical numbers in ``results/paper.yaml``."""

    raw: dict[str, Any]

    # ---------------------------------------------- generic accessors

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -------------------------------------------------- e2e helpers

    def tct(self, system: str, workload: str = "swe_bench") -> tuple[float, float]:
        """Return ``(mean_s, std_s)`` for ``system`` on ``workload``."""
        row = self.raw["e2e"][workload][system]
        return float(row["tct_s"]), float(row["std"])

    def memory_pct(self, system: str, workload: str = "swe_bench") -> float:
        return float(self.raw["e2e"][workload][system]["mem_pct"])

    def speedup_over(self, baseline: str, workload: str = "swe_bench") -> float:
        return float(self.raw["speedups_vs_saga"][workload][baseline]["factor"])

    @property
    def geomean_speedup_vs_vllm_apc(self) -> float:
        return float(self.raw["geomean_speedup_vs_vllm_apc"])

    def competitive(self, policy: str, workload: str = "swe_bench") -> float:
        return float(self.raw["competitive_ratio"][policy][workload])

    def slo_attainment(self, system: str, tenant_class: str = "overall") -> float:
        return float(self.raw["slo_attainment_pct"][system][tenant_class])


def load_paper_results(path: Path | str | None = None) -> PaperResults:
    """Load ``results/paper.yaml`` (or a custom override)."""
    p = Path(path) if path is not None else _PAPER_YAML
    if not p.exists():
        raise FileNotFoundError(f"paper results not found at {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PaperResults(raw=raw)
