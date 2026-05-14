"""Ray-based 16-worker distributed runtime with gRPC global coordinator.

The paper's 64-GPU cluster is partitioned as **16 vLLM workers x 4 GPUs each**
(Llama-3-70B at TP=4). Workers are Ray actors; the global coordinator is a
gRPC service so worker-to-coordinator RPCs do not pay Ray's serialization
overhead on the hot path (P99 worker--coordinator latency target: <= 5 ms).

Layout::

    Coordinator (gRPC, port 50051)        scheduling epoch = 100 ms
        |
        |  bi-directional streaming RPCs:
        |    submit_aeg, route_request, observe_event, steal_request
        |
        +-> Worker[0]   (Ray actor, GPUs 0-3,  node 0)
        +-> Worker[1]   (Ray actor, GPUs 4-7,  node 0)
        +-> Worker[2]   (Ray actor, GPUs 0-3,  node 1)
        ...
        +-> Worker[15]  (Ray actor, GPUs 4-7,  node 7)

The coordinator is the same :class:`saga.scheduler.coordinator.GlobalCoordinator`
that the unit tests exercise in :mod:`saga.sim`; only the transport differs.
Policy changes are pinned by the unit tests on a CPU host, then shipped
unmodified to the cluster.

Submodules
----------

* :mod:`saga.serving.distributed.ray_cluster` --- the Ray actor wrapper that
  hosts the vLLM engine + SAGA hooks on each of the 16 workers.
* :mod:`saga.serving.distributed.grpc_coordinator` --- the gRPC service that
  exposes the coordinator API; bundled with a configurable batched-flush
  loop that meets the P99 < 5 ms target.
* :mod:`saga.serving.distributed.grpc_worker` --- the gRPC stub the worker
  uses to talk back to the coordinator.
* :mod:`saga.serving.distributed.proto` --- the protobuf definitions
  shared by coordinator and worker.
* :mod:`saga.serving.distributed.cluster_spec` --- the paper's reference
  cluster configuration (16 workers, 4 GPUs each, NVLink + IB).
"""

from __future__ import annotations

from saga.serving.distributed.cluster_spec import (
    REFERENCE_CLUSTER_SPEC,
    ClusterSpec,
    WorkerSpec,
)
from saga.serving.distributed.grpc_coordinator import (
    CoordinatorService,
    GrpcCoordinatorConfig,
)
from saga.serving.distributed.grpc_coordinator import (
    serve as serve_coordinator,
)
from saga.serving.distributed.grpc_worker import WorkerClient
from saga.serving.distributed.ray_cluster import (
    RayClusterConfig,
    SagaWorkerActor,
    launch_cluster,
)


__all__ = [
    "REFERENCE_CLUSTER_SPEC",
    "ClusterSpec",
    "CoordinatorService",
    "GrpcCoordinatorConfig",
    "RayClusterConfig",
    "SagaWorkerActor",
    "WorkerClient",
    "WorkerSpec",
    "launch_cluster",
    "serve_coordinator",
]
