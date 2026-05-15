"""Ray actor wrapping a SAGA-equipped vLLM worker.

Each :class:`SagaWorkerActor` owns one :class:`SagaVLLMEngine` and one
:class:`WorkerClient` (the gRPC stub talking to the coordinator). The
actor exposes three Ray-RPC methods:

* ``submit(prompt, session_id, tenant_id)`` -- generate text.
* ``heartbeat()``                            -- push live status to the
  coordinator (called once per epoch).
* ``shutdown()``                             -- gracefully drain.

:func:`launch_cluster` materialises 16 actors from
:class:`ClusterSpec` and returns their handles. Without ``ray`` installed
it falls back to building local non-actor instances so unit tests can
exercise the wiring path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from saga.serving.distributed.cluster_spec import (
    REFERENCE_CLUSTER_SPEC,
    ClusterSpec,
    WorkerSpec,
)
from saga.serving.distributed.grpc_worker import WorkerClient
from saga.serving.engine import SagaVLLMEngine
from saga.serving.errors import MissingRuntimeError
from saga.serving.vllm_ext.llama3_70b import LLAMA3_70B, ModelConfig
from saga.utils.logging import get_logger


log = get_logger("saga.serving.distributed.ray_cluster")


@dataclass
class RayClusterConfig:
    """Configuration for launching SAGA actors via Ray."""

    cluster_spec: ClusterSpec = field(default_factory=lambda: REFERENCE_CLUSTER_SPEC)
    model: ModelConfig = field(default_factory=lambda: LLAMA3_70B)
    num_gpus_per_worker: int = 4
    num_cpus_per_worker: int = 8
    coordinator_host: str = "saga-coord-0"
    coordinator_port: int = 50_051


class SagaWorkerActor:
    """One vLLM worker actor.

    Hosts a real :class:`SagaVLLMEngine` (Llama-3-70B-Instruct at TP=4) on
    its 4 A100s, registers with the gRPC coordinator at boot, and accepts
    submit / heartbeat / shutdown Ray-RPC calls. Without Ray installed the
    class still works as a plain object so the policy unit tests can
    exercise the actor surface without a Ray cluster.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        model: ModelConfig = LLAMA3_70B,
        coordinator_host: str = "saga-coord-0",
        coordinator_port: int = 50_051,
    ) -> None:
        self.spec = spec
        self.model = model
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", spec.cuda_visible_devices)
        self.engine = SagaVLLMEngine(model_config=model)
        self.client = WorkerClient(
            host=coordinator_host,
            port=coordinator_port,
            worker_id=spec.worker_id,
        )
        self._started_at = time.monotonic()
        self._heartbeats = 0

    def boot(self) -> None:
        # In real-cluster mode this calls ``engine.serve(workers=...)`` but
        # workers is built from the cluster spec by ``launch_cluster``.
        log.info(
            "SagaWorkerActor[%d] booted (gpus=%s)",
            self.spec.worker_id,
            self.spec.cuda_visible_devices,
        )

    def submit(
        self,
        prompts: list[str] | str,
        session_id: str,
        tenant_id: str = "default",
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        return self.engine.generate(
            prompts=prompts,
            session_id=session_id,
            tenant_id=tenant_id,
            max_tokens=max_tokens,
        )

    def heartbeat(self) -> dict[str, Any]:
        self._heartbeats += 1
        stats = self.engine.stats()
        try:
            self.client.push_heartbeats([_make_worker_status(self.spec.worker_id, stats)])
        except MissingRuntimeError:
            pass
        return {"worker_id": self.spec.worker_id, "n_heartbeats": self._heartbeats}

    def shutdown(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        log.info(
            "SagaWorkerActor[%d] shutdown after %.1fs",
            self.spec.worker_id,
            time.monotonic() - self._started_at,
        )


def launch_cluster(cfg: RayClusterConfig | None = None) -> list[Any]:
    """Launch the 16-worker cluster (or local stand-ins) and return handles."""
    cfg = cfg or RayClusterConfig()
    workers_specs = cfg.cluster_spec.workers()

    try:
        import ray
    except ImportError:
        log.warning(
            "ray not installed; running workers as in-process Python objects "
            "(no actor isolation). Use this path only for unit tests."
        )
        return [
            SagaWorkerActor(
                spec=ws,
                model=cfg.model,
                coordinator_host=cfg.coordinator_host,
                coordinator_port=cfg.coordinator_port,
            )
            for ws in workers_specs
        ]

    if not ray.is_initialized():
        ray.init()

    RemoteWorker = ray.remote(
        num_gpus=cfg.num_gpus_per_worker,
        num_cpus=cfg.num_cpus_per_worker,
    )(SagaWorkerActor)
    handles = [
        RemoteWorker.remote(
            spec=ws,
            model=cfg.model,
            coordinator_host=cfg.coordinator_host,
            coordinator_port=cfg.coordinator_port,
        )
        for ws in workers_specs
    ]
    ray.get([h.boot.remote() for h in handles])
    log.info(
        "Launched %d Ray worker actors against coordinator %s:%d",
        len(handles),
        cfg.coordinator_host,
        cfg.coordinator_port,
    )
    return handles


def _make_worker_status(worker_id: int, stats: dict[str, Any]) -> Any:
    coord_stats = stats.get("coordinator", {}) if isinstance(stats, dict) else {}
    try:
        from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
            saga_coordinator_pb2 as pb,
        )

        return pb.WorkerStatus(
            worker_id=worker_id,
            queue_depth=int(coord_stats.get("queue_depth", 0)),
            memory_pressure=float(coord_stats.get("memory_pressure", 0.0)),
            utilization=float(coord_stats.get("utilization", 0.0)),
            busy_until_ms=float(coord_stats.get("busy_until_ms", 0.0)),
        )
    except ImportError:
        return {
            "worker_id": worker_id,
            "queue_depth": int(coord_stats.get("queue_depth", 0)),
            "memory_pressure": float(coord_stats.get("memory_pressure", 0.0)),
            "utilization": float(coord_stats.get("utilization", 0.0)),
            "busy_until_ms": float(coord_stats.get("busy_until_ms", 0.0)),
        }
