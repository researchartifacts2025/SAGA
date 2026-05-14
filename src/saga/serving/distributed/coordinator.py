"""SagaCoordinator -- user-facing facade over the gRPC coordinator service.

Composes:

* :class:`GlobalCoordinator` (policy: AFS, routing, work stealing).
* :class:`CoordinatorService` (gRPC adapter implementing the wire protocol).
* :func:`saga.serving.distributed.grpc_coordinator.serve` (the actual server).

Use this class to start the SAGA coordinator from a script::

    from saga.serving.distributed import SagaCoordinator, REFERENCE_CLUSTER_SPEC

    coord = SagaCoordinator.from_cluster_spec(REFERENCE_CLUSTER_SPEC)
    coord.serve_sync()    # blocks
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from saga.core.types import Worker
from saga.scheduler.coordinator import CoordinatorConfig, GlobalCoordinator
from saga.serving.distributed.cluster_spec import (
    REFERENCE_CLUSTER_SPEC,
    ClusterSpec,
)
from saga.serving.distributed.grpc_coordinator import (
    CoordinatorService,
    GrpcCoordinatorConfig,
)
from saga.serving.distributed.grpc_coordinator import serve as _grpc_serve
from saga.utils.logging import get_logger


log = get_logger("saga.serving.distributed.coordinator")


@dataclass
class SagaCoordinator:
    """One-call entrypoint for booting the SAGA coordinator."""

    coordinator_config: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    grpc_config: GrpcCoordinatorConfig = field(default_factory=GrpcCoordinatorConfig)
    _coordinator: GlobalCoordinator | None = field(default=None, repr=False)
    _service: CoordinatorService | None = field(default=None, repr=False)
    _server: Any = field(default=None, repr=False)

    # -----------------------------------------------------------------
    # Constructors

    @classmethod
    def from_cluster_spec(
        cls,
        spec: ClusterSpec = REFERENCE_CLUSTER_SPEC,
        coordinator_config: CoordinatorConfig | None = None,
        grpc_config: GrpcCoordinatorConfig | None = None,
        worker_capacity_tokens: int = 1_500_000,
    ) -> SagaCoordinator:
        """Build a coordinator from the paper's reference cluster spec."""
        sc = cls(
            coordinator_config=coordinator_config or CoordinatorConfig(),
            grpc_config=grpc_config or GrpcCoordinatorConfig(
                host="0.0.0.0", port=spec.coordinator_port
            ),
        )
        sc.bind_workers(
            Worker(worker_id=w.worker_id, capacity_tokens=worker_capacity_tokens)
            for w in spec.workers()
        )
        return sc

    # -----------------------------------------------------------------
    # Wiring

    def bind_workers(self, workers: Iterable[Worker]) -> GlobalCoordinator:
        """Construct the underlying :class:`GlobalCoordinator`."""
        self._coordinator = GlobalCoordinator(
            workers=list(workers), cfg=self.coordinator_config
        )
        self._service = CoordinatorService(
            coordinator=self._coordinator, cfg=self.grpc_config
        )
        return self._coordinator

    @property
    def coordinator(self) -> GlobalCoordinator:
        if self._coordinator is None:
            raise RuntimeError("call bind_workers() / from_cluster_spec() first")
        return self._coordinator

    @property
    def service(self) -> CoordinatorService:
        if self._service is None:
            raise RuntimeError("call bind_workers() / from_cluster_spec() first")
        return self._service

    # -----------------------------------------------------------------
    # Server lifecycle

    def serve_sync(self, block: bool = True) -> Any:
        """Start the gRPC server. By default blocks until SIGINT."""
        if self._coordinator is None:
            raise RuntimeError("bind workers before calling serve_sync()")
        self._server = _grpc_serve(self._coordinator, self.grpc_config)
        if not block:
            return self._server
        try:
            self._server.wait_for_termination()
        except KeyboardInterrupt:  # pragma: no cover
            log.info("coordinator shutting down on SIGINT")
            self.shutdown()
        return self._server

    def shutdown(self, grace_s: float = 5.0) -> None:
        if self._server is None:
            return
        try:
            self._server.stop(grace_s)
        finally:
            self._server = None
