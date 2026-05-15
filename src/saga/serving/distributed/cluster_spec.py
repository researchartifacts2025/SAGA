"""Reference cluster specification.

The paper's evaluation runs on::

    8 nodes x 8 A100-80GB GPUs (64 GPUs)
    NVLink intra-node + 200 Gbps InfiniBand HDR inter-node
    2x AMD EPYC 7763 (128 cores) and 1 TB DDR4-3200 per node

Llama-3-70B-Instruct runs at tensor parallelism 4 per instance, so the cluster
hosts **16 worker instances** total. Each worker owns 4 GPUs --- the
arrangement is GPUs {0..3} on the first instance per node and GPUs {4..7} on
the second instance per node, exposed to vLLM via ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerSpec:
    """One vLLM worker = one tensor-parallel inference group."""

    worker_id: int
    node_id: int
    gpu_indices: tuple[int, ...]  # local CUDA device ids on the node
    nic: str = "mlx5_0"  # IB device used by NCCL

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(str(i) for i in self.gpu_indices)


@dataclass(frozen=True)
class NodeSpec:
    """Compute node hosting up to ``len(workers)`` workers."""

    node_id: int
    hostname: str
    workers: tuple[WorkerSpec, ...]

    @property
    def n_gpus(self) -> int:
        return sum(len(w.gpu_indices) for w in self.workers)


@dataclass(frozen=True)
class ClusterSpec:
    """Full cluster topology as evaluated in the paper."""

    name: str
    nodes: tuple[NodeSpec, ...]
    coordinator_host: str = "saga-coord-0"
    coordinator_port: int = 50051
    nccl_socket_ifname: str = "ib0"

    @property
    def n_workers(self) -> int:
        return sum(len(n.workers) for n in self.nodes)

    @property
    def n_gpus(self) -> int:
        return sum(n.n_gpus for n in self.nodes)

    def workers(self) -> tuple[WorkerSpec, ...]:
        return tuple(w for n in self.nodes for w in n.workers)


def _build_reference_cluster() -> ClusterSpec:
    """Construct the 8-node x 2-worker spec used in Table 3."""
    nodes: list[NodeSpec] = []
    wid = 0
    for node_id in range(8):
        workers: list[WorkerSpec] = []
        for slot in range(2):  # 2 vLLM instances per node, TP=4 each
            gpus = tuple(range(slot * 4, slot * 4 + 4))
            workers.append(
                WorkerSpec(
                    worker_id=wid,
                    node_id=node_id,
                    gpu_indices=gpus,
                    nic="mlx5_0",
                )
            )
            wid += 1
        nodes.append(
            NodeSpec(
                node_id=node_id,
                hostname=f"saga-gpu-{node_id:02d}",
                workers=tuple(workers),
            )
        )
    return ClusterSpec(
        name="paper-64a100",
        nodes=tuple(nodes),
        coordinator_host="saga-coord-0",
        coordinator_port=50051,
        nccl_socket_ifname="ib0",
    )


REFERENCE_CLUSTER_SPEC: ClusterSpec = _build_reference_cluster()


def assert_paper_invariants(spec: ClusterSpec = REFERENCE_CLUSTER_SPEC) -> None:
    """Cross-check the cluster spec against the paper's stated configuration."""
    if spec.n_workers != 16:
        raise AssertionError(f"Expected 16 workers, got {spec.n_workers}")
    if spec.n_gpus != 64:
        raise AssertionError(f"Expected 64 GPUs, got {spec.n_gpus}")
    for w in spec.workers():
        if len(w.gpu_indices) != 4:
            raise AssertionError(
                f"Worker {w.worker_id} should own 4 GPUs (TP=4); got {len(w.gpu_indices)}"
            )


if __name__ == "__main__":  # pragma: no cover
    assert_paper_invariants()
    spec = REFERENCE_CLUSTER_SPEC
    print(f"cluster={spec.name} workers={spec.n_workers} gpus={spec.n_gpus}")
    for n in spec.nodes:
        for w in n.workers:
            print(
                f"  worker_{w.worker_id:02d}  node={n.hostname}"
                f"  CUDA_VISIBLE_DEVICES={w.cuda_visible_devices}"
            )
