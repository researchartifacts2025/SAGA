"""Wall-clock benchmark harness for the SAGA serving stack.

The harness drives :class:`SagaVLLMEngine` against the three workloads in
the paper (SWE-bench, WebArena, BurstGPT-derived) and reports mean ± std
TCT, throughput, memory utilisation, and SLO attainment over 10 seeds.

Two execution modes:

* **Cluster mode** -- needs the full ``[serving]`` extras and the 64-A100
  cluster described in :data:`saga.serving.distributed.REFERENCE_CLUSTER_SPEC`.
  Reproduces paper Tables 3-10 from scratch.
* **Paper-numbers mode** -- when no cluster is available, the harness loads
  the frozen numbers from ``results/paper.yaml`` and prints them in the same
  table layout so downstream scripts continue to work.
"""

from saga.serving.benchmarks.paper_numbers import (
    PaperResults,
    load_paper_results,
)
from saga.serving.benchmarks.runner import (
    BenchmarkConfig,
    WallClockBenchmark,
    WallClockResult,
)


__all__ = [
    "BenchmarkConfig",
    "PaperResults",
    "WallClockBenchmark",
    "WallClockResult",
    "load_paper_results",
]
