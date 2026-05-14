"""SAGA serving stack --- real-GPU implementation.

This package is the production code path described in §8 (Implementation)
of the paper. It runs Llama-3-70B-Instruct on the 64-A100 reference cluster
with workflow-aware PagedAttention and is the *default* target of the
:mod:`saga.entrypoints.bench_wallclock` benchmark harness.

* :mod:`saga.serving.vllm_ext` --- vLLM v0.6.0 (V1 engine) extension. Live
  monkey-patches against ``BlockSpaceManagerV2.allocate`` / ``free``, the V1
  ``EngineCore`` step loop, and ``model_executor.execute_model``. The hooks
  install workflow-aware eviction, tool-aware TTL, AFS preemption, and a
  separate-stream CUDA prefetch on every running vLLM worker.
* :mod:`saga.serving.distributed` --- Ray-based 16-worker runtime and gRPC
  global coordinator. P99 worker--coordinator latency target < 5 ms over
  200 Gbps InfiniBand.
* :mod:`saga.serving.benchmarks` --- 10-seed wall-clock benchmark harness
  that reproduces Tables 3-10. Default execution mode is ``cluster`` (live
  Ray + vLLM); a ``paper`` mode replays :file:`results/paper.yaml` so docs
  tooling renders the tables without GPUs.

The policy modules (:mod:`saga.cache`, :mod:`saga.scheduler`,
:mod:`saga.fairness`, :mod:`saga.workflow`) are the same objects that drive
the live cluster; :mod:`saga.sim` exercises them in a deterministic
discrete-event harness for CI / regression testing.
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
