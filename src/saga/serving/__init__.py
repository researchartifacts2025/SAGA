"""SAGA serving stack.

This package contains the artifacts referenced in §8 (Implementation) of the
paper:

* :mod:`saga.serving.vllm_ext` --- the vLLM v0.6.0 (V1 engine) extension that
  drives real Llama-3-70B-Instruct inference with workflow-aware PagedAttention.
* :mod:`saga.serving.distributed` --- the Ray-based 16-worker runtime and gRPC
  global coordinator (P99 worker--coordinator latency < 5 ms).
* :mod:`saga.serving.benchmarks` --- the wall-clock benchmark harness that
  reproduces the 10-seed, 64-A100 numbers in Tables 3-10.

The simulator in :mod:`saga.sim` and the serving stack here share the same
scheduler/cache/fairness modules; both targets consume the same
:class:`saga.scheduler.coordinator.GlobalCoordinator` so policy changes are
validated in simulation before being deployed against the real engine.
"""

from __future__ import annotations

from saga.serving.errors import MissingRuntimeError


VLLM_REF_VERSION = "0.6.0"
RAY_REF_VERSION = "2.9.0"
GRPCIO_REF_VERSION = "1.60.0"
TORCH_REF_VERSION = "2.1.2"
TARGET_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
TARGET_CLUSTER = "8 nodes x 8 A100-80GB (64 GPUs total, NVLink + 200 Gbps IB)"
N_INSTANCES_DEFAULT = 16  # 64 GPUs / TP=4
COORDINATOR_GRPC_PORT_DEFAULT = 50_051


def __getattr__(name: str):
    """PEP 562 lazy attribute access for heavyweight serving classes."""
    if name == "SagaVLLMEngine":
        from saga.serving.engine import SagaVLLMEngine

        return SagaVLLMEngine
    if name == "SagaCoordinator":
        from saga.serving.distributed.coordinator import SagaCoordinator

        return SagaCoordinator
    if name == "SagaWorker":
        from saga.serving.distributed.worker import SagaWorker

        return SagaWorker
    if name == "WallClockBenchmark":
        from saga.serving.benchmarks.runner import WallClockBenchmark

        return WallClockBenchmark
    raise AttributeError(name)


__all__ = [
    "COORDINATOR_GRPC_PORT_DEFAULT",
    "GRPCIO_REF_VERSION",
    "N_INSTANCES_DEFAULT",
    "RAY_REF_VERSION",
    "TARGET_CLUSTER",
    "TARGET_MODEL",
    "TORCH_REF_VERSION",
    "VLLM_REF_VERSION",
    "MissingRuntimeError",
    "SagaCoordinator",
    "SagaVLLMEngine",
    "SagaWorker",
    "WallClockBenchmark",
]
