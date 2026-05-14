"""Wall-clock benchmark harness for the SAGA serving stack.

Drives :class:`SagaVLLMEngine` against the three workloads in the paper
(SWE-bench, WebArena, BurstGPT-derived) on the live 64-A100 cluster and
reports mean +/- std TCT, throughput, memory utilisation, and SLO
attainment over **10 random seeds**. Default execution mode is
**cluster**: it boots Ray, attaches to the gRPC coordinator, and streams
real Llama-3-70B-Instruct inference through 16 vLLM workers.

A **paper-numbers** fallback loads the frozen wall-clock values from
:file:`results/paper.yaml` so documentation tooling, CI dashboards, and
table-generation scripts produce identical schema without 64 A100s.

Both modes emit the same :class:`WallClockResult`; downstream consumers
never branch on environment.
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
