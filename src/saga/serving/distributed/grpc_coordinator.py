"""gRPC service hosting the SAGA :class:`GlobalCoordinator`.

Live in production on the 64-A100 reference cluster: one process on the
head node fields RPCs from all 16 vLLM workers + external agent clients
over a single gRPC endpoint. The service is a thin shim around the
:class:`GlobalCoordinator` that the policy unit tests pin down, so the
algorithm runs unmodified from CI to cluster.

The P99 worker -- coordinator latency budget (5 ms) is met by:

* Pinning the coordinator process to one node (no co-tenancy).
* Batched-flush event ingestion: ``EventStream`` accepts step events as a
  bidirectional stream and flushes them into the coordinator in 10 ms
  windows so per-event Python locks amortise.
* Bypassing Ray serialisation -- gRPC uses Protocol Buffers directly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any

from saga.core.aeg import AEGEdge, AEGNode, AgentExecutionGraph
from saga.core.types import Session, Task, ToolType
from saga.scheduler.coordinator import GlobalCoordinator
from saga.serving.errors import MissingRuntimeError
from saga.utils.logging import get_logger


log = get_logger("saga.serving.distributed.grpc_coordinator")


@dataclass
class GrpcCoordinatorConfig:
    """Tunable knobs for the coordinator gRPC server."""

    host: str = "0.0.0.0"
    port: int = 50_051
    max_workers: int = 32  # gRPC threadpool size
    event_flush_window_ms: float = 10.0  # batched flush window
    keepalive_ms: int = 30_000
    max_message_mb: int = 32


@dataclass
class CoordinatorService:
    """Bind the SAGA coordinator to the generated gRPC stubs.

    ``servicer()`` returns the object plugged into the gRPC server; it
    matches the protobuf-generated abstract base class.
    """

    coordinator: GlobalCoordinator
    cfg: GrpcCoordinatorConfig = field(default_factory=GrpcCoordinatorConfig)
    _event_buffer: list[Any] = field(default_factory=list, init=False, repr=False)
    _buffer_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _last_flush_ms: float = field(default=0.0, init=False, repr=False)
    # AEGs submitted via SubmitTask, keyed by session_id. The assigned
    # worker fetches its AEG from here on first admit so the WALRU hook
    # can compute predict_reuse. Bounded staleness: cleared on session
    # complete or by the coordinator's session-forget path.
    _aeg_by_session: dict[str, AgentExecutionGraph] = field(
        default_factory=dict, init=False, repr=False
    )

    # ----------------------------------------- service implementation

    def submit_task(self, request: Any, context: Any | None = None) -> Any:
        """Register a new task + AEG with the coordinator."""
        pb = request
        task = Task(
            task_id=pb.task.task_id,
            tenant_id=pb.task.tenant_id,
            workload_kind=pb.task.workload_kind,
            submit_time=float(pb.task.submit_time_ms),
            n_steps=int(pb.task.n_steps),
            aeg_id=pb.task.aeg_id,
            expected_tct_ms=float(pb.task.expected_tct_ms),
        )
        nodes = [
            AEGNode(
                index=n.index,
                tool_type=_tool_type_of(n.tool_name),
                prompt_tokens_est=int(n.prompt_tokens_est),
                output_tokens_est=int(n.output_tokens_est),
                observation_tokens_est=int(n.observation_tokens_est),
                is_terminal=bool(n.is_terminal),
            )
            for n in pb.aeg.nodes
        ]
        edges = [
            AEGEdge(src=int(e.src), dst=int(e.dst), probability=float(e.probability))
            for e in pb.aeg.edges
        ]
        aeg = AgentExecutionGraph(
            graph_id=pb.aeg.graph_id,
            nodes=nodes,
            edges=edges,
            workload_kind=pb.aeg.workload_kind,
            termination_prob=float(pb.aeg.termination_prob),
        )
        self._aeg_by_session[task.task_id] = aeg
        session = Session(session_id=task.task_id, task=task)
        if self.coordinator.afs is not None:
            weight = float(getattr(pb, "tenant_weight", 1.0)) or 1.0
            self.coordinator.afs.ensure_tenant(task.tenant_id, weight=weight)
        self.coordinator.register_session(session)
        decision = self.coordinator.route(session)
        return _make_submit_task_response(
            accepted=True,
            reason=decision.reason,
            worker_id=decision.worker_id,
        )

    def route(self, request: Any, context: Any | None = None) -> Any:
        sid = request.session_id
        sess = self.coordinator.get_session(sid)
        if sess is None:
            return _make_route_response(
                worker_id=-1, reason="unknown_session", cache_hit_expected=False
            )
        decision = self.coordinator.route(sess)
        return _make_route_response(
            worker_id=decision.worker_id,
            reason=decision.reason,
            cache_hit_expected=decision.cache_hit_expected,
        )

    def event_stream(
        self,
        request_iterator: Iterable[Any],
        context: Any | None = None,
    ) -> Any:
        """Consume a stream of step events and flush them in 10 ms windows."""
        for event in request_iterator:
            self._enqueue_event(event)
            now_ms = time.monotonic() * 1000.0
            if now_ms - self._last_flush_ms >= self.cfg.event_flush_window_ms:
                self._flush_events(now_ms)
        # Final flush.
        self._flush_events(time.monotonic() * 1000.0)
        return _make_ack()

    def heartbeat_stream(
        self,
        request_iterator: Iterable[Any],
        context: Any | None = None,
    ) -> Any:
        for status in request_iterator:
            try:
                w = self.coordinator.all_workers()[int(status.worker_id)]
                w.queue_depth = int(status.queue_depth)
                w.busy_until = float(status.busy_until_ms)
                w.cumulative_busy_ms = float(status.utilization) * max(1.0, w.busy_until)
            except (IndexError, ValueError):
                continue
        return _make_ack()

    def steal(self, request: Any, context: Any | None = None) -> Any:
        actions = self.coordinator.tick(now=float(request.now_ms))
        for a in actions:
            if a.thief_id == int(request.thief_worker_id) and a.success:
                return _make_steal_response(
                    success=True,
                    victim_worker_id=a.victim_id,
                    session_id=a.session_id or "",
                    migration_ms=a.migration_ms,
                    reason=a.reason,
                )
        return _make_steal_response(
            success=False,
            victim_worker_id=-1,
            session_id="",
            migration_ms=0.0,
            reason="no_candidate",
        )

    # ------------------------------------------------------ internals

    def _enqueue_event(self, event: Any) -> None:
        with self._buffer_lock:
            self._event_buffer.append(event)

    def _flush_events(self, now_ms: float) -> None:
        with self._buffer_lock:
            events = self._event_buffer
            self._event_buffer = []
            self._last_flush_ms = now_ms
        for ev in events:
            if self.coordinator.afs is not None and ev.aeg_node_index >= 0:
                sess = self.coordinator.get_session(ev.session_id)
                if sess is not None:
                    self.coordinator.afs.note_progress(
                        sess.tenant_id,
                        sess.task.task_id,
                        gpu_ms=float(ev.prefill_tokens + ev.decode_tokens) * 0.04,
                    )


# ---------------------------------------------------- factory helpers


def serve(
    coordinator: GlobalCoordinator,
    cfg: GrpcCoordinatorConfig | None = None,
) -> Any:
    """Start a gRPC server hosting ``coordinator`` and return its handle.

    Raises :class:`MissingRuntimeError` if ``grpc`` is not installed. The
    returned object exposes ``stop(grace)`` (gRPC API).
    """
    try:
        import grpc
    except ImportError as exc:
        raise MissingRuntimeError(
            runtime="grpcio",
            install_hint="pip install 'saga-sched[serving]'",
        ) from exc

    try:
        from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
            saga_coordinator_pb2_grpc,
        )
    except ImportError as exc:
        raise MissingRuntimeError(
            runtime="generated protobuf stubs",
            install_hint="make proto",
        ) from exc

    cfg = cfg or GrpcCoordinatorConfig()
    service = CoordinatorService(coordinator=coordinator, cfg=cfg)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.max_workers),
        options=[
            ("grpc.keepalive_time_ms", cfg.keepalive_ms),
            ("grpc.max_send_message_length", cfg.max_message_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", cfg.max_message_mb * 1024 * 1024),
        ],
    )
    saga_coordinator_pb2_grpc.add_SagaCoordinatorServiceServicer_to_server(
        _ServicerAdapter(service),
        server,
    )
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    log.info("SAGA coordinator gRPC server listening on %s:%d", cfg.host, cfg.port)
    return server


# --------------------------------------------- adapter for protobuf stubs


class _ServicerAdapter:
    """Bridges the dataclass ``CoordinatorService`` to the generated stub class.

    The protobuf-generated abstract servicer expects camelCase method names
    (``SubmitTask``, ``Route``, ``EventStream`` ...). This adapter forwards
    each one into the snake_case methods of :class:`CoordinatorService`.
    """

    def __init__(self, service: CoordinatorService) -> None:
        self._svc = service

    def SubmitTask(self, request, context):  # noqa: N802
        return self._svc.submit_task(request, context)

    def Route(self, request, context):  # noqa: N802
        return self._svc.route(request, context)

    def EventStream(self, request_iterator, context):  # noqa: N802
        return self._svc.event_stream(request_iterator, context)

    def HeartbeatStream(self, request_iterator, context):  # noqa: N802
        return self._svc.heartbeat_stream(request_iterator, context)

    def Steal(self, request, context):  # noqa: N802
        return self._svc.steal(request, context)


# ----------------------------------- helpers for proto-less unit tests


def _tool_type_of(name: str) -> ToolType:
    from saga.workflow.analyzer import tool_type_of as _impl

    return _impl(name)


def _make_submit_task_response(accepted: bool, reason: str, worker_id: int) -> Any:
    try:
        from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
            saga_coordinator_pb2 as pb,
        )

        return pb.SubmitTaskResponse(accepted=accepted, reason=reason, assigned_worker_id=worker_id)
    except ImportError:
        return {"accepted": accepted, "reason": reason, "assigned_worker_id": worker_id}


def _make_route_response(worker_id: int, reason: str, cache_hit_expected: bool) -> Any:
    try:
        from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
            saga_coordinator_pb2 as pb,
        )

        return pb.RouteResponse(
            worker_id=worker_id,
            reason=reason,
            cache_hit_expected=cache_hit_expected,
        )
    except ImportError:
        return {
            "worker_id": worker_id,
            "reason": reason,
            "cache_hit_expected": cache_hit_expected,
        }


def _make_ack() -> Any:
    try:
        from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
            saga_coordinator_pb2 as pb,
        )

        return pb.Ack()
    except ImportError:
        return {"ack": True}


def _make_steal_response(
    success: bool,
    victim_worker_id: int,
    session_id: str,
    migration_ms: float,
    reason: str,
) -> Any:
    try:
        from saga.serving.distributed.proto import (  # type: ignore[attr-defined]
            saga_coordinator_pb2 as pb,
        )

        return pb.StealResponse(
            success=success,
            victim_worker_id=victim_worker_id,
            session_id=session_id,
            migration_ms=migration_ms,
            reason=reason,
        )
    except ImportError:
        return {
            "success": success,
            "victim_worker_id": victim_worker_id,
            "session_id": session_id,
            "migration_ms": migration_ms,
            "reason": reason,
        }
