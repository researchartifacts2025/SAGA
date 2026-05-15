"""gRPC client used by SAGA workers to talk to the coordinator."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from saga.serving.errors import MissingRuntimeError
from saga.utils.logging import get_logger


log = get_logger("saga.serving.distributed.grpc_worker")


@dataclass
class WorkerClient:
    """Thin gRPC client wrapping the SAGA coordinator stub.

    Each worker keeps one persistent channel; the channel multiplexes the
    submit / route / event-stream / heartbeat / steal RPCs over HTTP/2. The
    client is thread-safe.
    """

    host: str = "saga-coord-0"
    port: int = 50_051
    worker_id: int = 0
    _channel: Any = field(default=None, init=False, repr=False)
    _stub: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ----------------------------------------------------- connect

    def connect(self) -> None:
        if self._stub is not None:
            return
        try:
            import grpc
        except ImportError as exc:
            raise MissingRuntimeError(
                runtime="grpcio",
                install_hint="pip install 'saga-sched[serving]'",
            ) from exc
        try:
            from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
                saga_coordinator_pb2_grpc as stubs,
            )
        except ImportError as exc:
            raise MissingRuntimeError(
                runtime="generated protobuf stubs",
                install_hint="make proto",
            ) from exc

        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self._stub = stubs.SagaCoordinatorServiceStub(self._channel)
        log.info("WorkerClient[%d] connected to %s:%d", self.worker_id, self.host, self.port)

    def close(self) -> None:
        with self._lock:
            if self._channel is not None:
                try:
                    self._channel.close()
                except Exception:
                    pass
            self._channel = None
            self._stub = None

    # ----------------------------------------------------- RPC API

    def submit_task(self, submit_req: Any) -> Any:
        self.connect()
        with self._lock:
            return self._stub.SubmitTask(submit_req)

    def route(self, route_req: Any) -> Any:
        self.connect()
        with self._lock:
            return self._stub.Route(route_req)

    def push_events(self, events: Iterable[Any]) -> Any:
        self.connect()
        with self._lock:
            return self._stub.EventStream(iter(events))

    def push_heartbeats(self, statuses: Iterable[Any]) -> Any:
        self.connect()
        with self._lock:
            return self._stub.HeartbeatStream(iter(statuses))

    def request_steal(self, request: Any) -> Any:
        self.connect()
        with self._lock:
            return self._stub.Steal(request)

    # ----------------------------------------------- context manager

    def __enter__(self) -> WorkerClient:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
