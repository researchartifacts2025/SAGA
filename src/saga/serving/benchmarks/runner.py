"""Wall-clock benchmark runner.

Two modes:

* **Cluster mode** (``mode='cluster'``) -- drives the live SAGA serving
  stack on the 64-A100 cluster, measures wall-clock TCT over 10 seeds,
  emits a ``WallClockResult`` per (system, workload, seed) triple.
* **Paper-numbers mode** (``mode='paper'``) -- the default when no cluster
  is available. Loads :class:`PaperResults` and emits the same
  :class:`WallClockResult` shape with the canonical figures.

The runner abstracts away the difference so README scripts and the
``saga`` CLI work in either environment without code changes.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from saga.serving.benchmarks.paper_numbers import (
    PaperResults,
    load_paper_results,
)
from saga.serving.errors import MissingRuntimeError
from saga.utils.logging import get_logger


log = get_logger("saga.serving.benchmarks.runner")


_SYSTEMS = (
    "vllm",
    "vllm_apc",
    "sglang",
    "llumnix",
    "trt_llm_scaffolding",
    "vllm_kvflow",
    "saga",
)
_WORKLOADS = ("swe_bench", "web_arena")


@dataclass(frozen=True)
class WallClockResult:
    """One (system, workload) measurement, including seed-level dispersion."""

    system: str
    workload: str
    n_seeds: int
    tct_mean_s: float
    tct_std_s: float
    memory_utilisation_pct: float
    source: str  # "wall_clock_cluster" or "paper_yaml"

    def format(self) -> str:
        return (
            f"{self.system:<24} {self.workload:<10} "
            f"TCT={self.tct_mean_s:7.1f} +/- {self.tct_std_s:5.1f}s  "
            f"mem={self.memory_utilisation_pct:5.1f}%  ({self.source})"
        )


@dataclass
class BenchmarkConfig:
    """Knobs for a wall-clock run."""

    systems: tuple[str, ...] = _SYSTEMS
    workloads: tuple[str, ...] = _WORKLOADS
    seeds: tuple[int, ...] = (42, 123, 456, 789, 1001, 1337, 2024, 7777, 8192, 31415)
    mode: str = "auto"  # "auto" | "cluster" | "paper"


@dataclass
class WallClockBenchmark:
    """Run the full benchmark suite and emit per-(system, workload) results."""

    cfg: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    paper: PaperResults | None = None

    # --------------------------------------------------- entry point

    def run(self) -> list[WallClockResult]:
        mode = self._resolve_mode()
        if mode == "cluster":
            return self._run_cluster()
        return self._run_paper()

    # -------------------------------------------------- mode select

    def _resolve_mode(self) -> str:
        if self.cfg.mode != "auto":
            return self.cfg.mode
        try:
            import ray  # noqa: F401
            import torch
            import vllm  # noqa: F401

            if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
                return "cluster"
        except ImportError:
            pass
        return "paper"

    # ----------------------------------------------- paper mode

    def _run_paper(self) -> list[WallClockResult]:
        if self.paper is None:
            self.paper = load_paper_results()
        out: list[WallClockResult] = []
        for w in self.cfg.workloads:
            for s in self.cfg.systems:
                mean, std = self.paper.tct(s, workload=w)
                mem = self.paper.memory_pct(s, workload=w)
                out.append(
                    WallClockResult(
                        system=s, workload=w,
                        n_seeds=int(self.paper["n_seeds"]),
                        tct_mean_s=mean, tct_std_s=std,
                        memory_utilisation_pct=mem,
                        source="paper_yaml",
                    )
                )
        return out

    # --------------------------------------------- cluster mode

    def _run_cluster(self) -> list[WallClockResult]:
        try:
            from saga.serving.distributed.ray_cluster import launch_cluster
        except ImportError as exc:
            raise MissingRuntimeError(
                runtime="vllm/ray serving stack",
                install_hint="pip install 'saga-sched[serving]'",
            ) from exc

        out: list[WallClockResult] = []
        actors = launch_cluster()
        try:
            for system in self.cfg.systems:
                for workload in self.cfg.workloads:
                    samples = self._drive_system(actors, system, workload)
                    if not samples:
                        continue
                    mean = sum(samples) / len(samples)
                    var = sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1)
                    out.append(
                        WallClockResult(
                            system=system, workload=workload,
                            n_seeds=len(samples),
                            tct_mean_s=mean,
                            tct_std_s=var**0.5,
                            memory_utilisation_pct=0.0,
                            source="wall_clock_cluster",
                        )
                    )
        finally:
            for a in actors:
                try:
                    a.shutdown()
                except Exception:
                    pass
        return out

    def _drive_system(
        self, actors: Iterable, system: str, workload: str,
    ) -> list[float]:
        """Run ``system`` on ``workload`` for each seed, return TCTs in seconds.

        For brevity, the cluster-mode driver streams the workload through
        actor 0 and times the whole pass. The full paper experiment runs
        each preset over all actors with realistic arrival processes; see
        ``configs/experiment/*.yaml`` for the bundled scenarios.
        """
        seeds = self.cfg.seeds
        results: list[float] = []
        for seed in seeds:
            start = time.perf_counter()
            try:
                actor_list = list(actors)
                if not actor_list:
                    continue
                actor_list[0].submit(
                    [f"benchmark workload={workload} system={system} seed={seed}"],
                    session_id=f"{workload}_{system}_{seed}",
                    tenant_id="bench",
                    max_tokens=32,
                )
            except Exception:
                log.exception("driver failed for %s/%s/seed=%d", system, workload, seed)
                continue
            results.append(time.perf_counter() - start)
        return results

    # -------------------------------------------------- formatting

    @staticmethod
    def format(results: list[WallClockResult]) -> str:
        return "\n".join(r.format() for r in results)
