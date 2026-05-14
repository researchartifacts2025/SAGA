"""SagaWorker -- user-facing facade over one Ray worker actor.

Composes:

* :class:`SagaWorkerActor` (Ray actor that hosts the vLLM engine).
* :class:`WorkerClient` (gRPC stub to the coordinator).
* :class:`SagaVLLMEngine` (the actual inference engine).

Most callers use :func:`launch_cluster` instead of constructing :class:`SagaWorker`
directly --- it bundles the 16 actors and their gRPC connections in one call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from saga.serving.distributed.cluster_spec import WorkerSpec
from saga.serving.distributed.grpc_worker import WorkerClient
from saga.serving.vllm_ext.llama3_70b import LLAMA3_70B, ModelConfig
from saga.utils.logging import get_logger


log = get_logger("saga.serving.distributed.worker")


@dataclass
class SagaWorker:
    """One vLLM serving instance + its gRPC link to the coordinator."""

    spec: WorkerSpec
    coordinator_host: str = "saga-coord-0"
    coordinator_port: int = 50_051
    model: ModelConfig = field(default_factory=lambda: LLAMA3_70B)
    heartbeat_period_s: float = 0.1  # 10 Hz

    _engine: Any = field(default=None, repr=False)
    _client: WorkerClient | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)

    # -----------------------------------------------------------------

    def configure_cuda(self) -> None:
        """Pin the worker to its 4 GPUs via ``CUDA_VISIBLE_DEVICES``."""
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = self.spec.cuda_visible_devices
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "ib0")
        os.environ.setdefault("NCCL_IB_HCA", self.spec.nic)

    def start(self) -> None:
        """Boot the engine and open the gRPC channel."""
        from saga.serving.engine import SagaVLLMEngine

        self.configure_cuda()
        self._engine = SagaVLLMEngine(model_config=self.model)
        self._client = WorkerClient(
            host=self.coordinator_host,
            port=self.coordinator_port,
            worker_id=self.spec.worker_id,
        )
        try:
            self._client.connect()
        except Exception:  # pragma: no cover
            log.exception("WorkerClient connect failed (will retry on first RPC)")
        self._running = True
        log.info(
            "worker_%d ready (gpus=%s, coord=%s:%d)",
            self.spec.worker_id,
            self.spec.cuda_visible_devices,
            self.coordinator_host,
            self.coordinator_port,
        )

    def stop(self) -> None:
        self._running = False
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._engine = None

    # -----------------------------------------------------------------
    # Public API exposed as Ray-RPC methods

    def submit(
        self,
        prompts: list[str] | str,
        session_id: str,
        tenant_id: str = "default",
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("worker not started; call start() first")
        return self._engine.generate(
            prompts=prompts,
            session_id=session_id,
            tenant_id=tenant_id,
            max_tokens=max_tokens,
        )

    def heartbeat(self) -> dict[str, Any]:
        return {
            "worker_id": self.spec.worker_id,
            "running": self._running,
            "wallclock_ms": time.monotonic() * 1000.0,
        }
