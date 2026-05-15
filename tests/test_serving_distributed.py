"""Tests for the Ray + gRPC distributed runtime.

Runs without ray/grpcio/vllm installed; the actor factory falls back to
in-process objects and the gRPC server raises a clear MissingRuntimeError
that the tests assert on.
"""

from __future__ import annotations

import pytest

from saga.core.types import Worker
from saga.scheduler.coordinator import CoordinatorConfig, GlobalCoordinator
from saga.serving.distributed.cluster_spec import (
    REFERENCE_CLUSTER_SPEC,
    assert_paper_invariants,
)
from saga.serving.distributed.grpc_coordinator import (
    CoordinatorService,
    GrpcCoordinatorConfig,
    serve,
)
from saga.serving.distributed.grpc_worker import WorkerClient
from saga.serving.distributed.ray_cluster import (
    RayClusterConfig,
    SagaWorkerActor,
    launch_cluster,
)
from saga.serving.errors import MissingRuntimeError


@pytest.mark.unit
def test_reference_cluster_has_16_workers_64_gpus() -> None:
    assert_paper_invariants()
    assert REFERENCE_CLUSTER_SPEC.n_workers == 16
    assert REFERENCE_CLUSTER_SPEC.n_gpus == 64
    for w in REFERENCE_CLUSTER_SPEC.workers():
        assert len(w.gpu_indices) == 4


@pytest.mark.unit
def test_worker_cuda_visible_devices_string() -> None:
    w = REFERENCE_CLUSTER_SPEC.workers()[0]
    assert w.cuda_visible_devices == "0,1,2,3"
    w7 = REFERENCE_CLUSTER_SPEC.workers()[1]
    assert w7.cuda_visible_devices == "4,5,6,7"


@pytest.mark.unit
def test_coordinator_service_constructs() -> None:
    workers = [
        Worker(
            worker_id=i,
            node_id=0,
            gpu_indices=(i,),
            kv_capacity_tokens=1_000_000,
            decode_tokens_per_ms=38.0,
            prefill_tokens_per_ms=850.0,
        )
        for i in range(2)
    ]
    coord = GlobalCoordinator(workers=workers, cfg=CoordinatorConfig())
    svc = CoordinatorService(
        coordinator=coord,
        cfg=GrpcCoordinatorConfig(port=0),
    )
    assert svc.coordinator is coord


@pytest.mark.unit
def test_serve_without_grpc_raises_clear_error() -> None:
    workers = [
        Worker(
            worker_id=0,
            node_id=0,
            gpu_indices=(0,),
            kv_capacity_tokens=1_000_000,
            decode_tokens_per_ms=38.0,
            prefill_tokens_per_ms=850.0,
        )
    ]
    coord = GlobalCoordinator(workers=workers, cfg=CoordinatorConfig())
    try:
        import grpc  # noqa: F401
    except ImportError:
        with pytest.raises(MissingRuntimeError):
            serve(coordinator=coord)
    else:
        # grpc installed but generated stubs may be missing -- accept either path.
        try:
            server = serve(coordinator=coord, cfg=GrpcCoordinatorConfig(port=0))
        except MissingRuntimeError:
            return
        try:
            assert server is not None
        finally:
            server.stop(grace=None)


@pytest.mark.unit
def test_worker_client_connect_without_grpc_raises() -> None:
    client = WorkerClient(worker_id=0)
    try:
        import grpc  # noqa: F401
    except ImportError:
        with pytest.raises(MissingRuntimeError):
            client.connect()
    else:
        # Either the connect succeeds (stubs available) or the stub
        # import path raises the expected MissingRuntimeError.
        try:
            client.connect()
            client.close()
        except MissingRuntimeError:
            pass


@pytest.mark.unit
def test_launch_cluster_without_ray_returns_local_actors() -> None:
    cfg = RayClusterConfig()
    actors = launch_cluster(cfg)
    assert len(actors) == REFERENCE_CLUSTER_SPEC.n_workers == 16
    assert all(isinstance(a, SagaWorkerActor) for a in actors)
    # heartbeat is a no-op without the gRPC stack.
    actors[0].heartbeat()
    for a in actors:
        a.shutdown()
