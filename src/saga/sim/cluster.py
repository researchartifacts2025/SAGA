"""Cluster construction.

A cluster is described by a small dataclass; ``build_cluster`` materializes
``Worker`` objects, the cache pool per worker, and the throughput parameters
used by the inference model.

Defaults match the paper's hardware (8 nodes x 8 A100-80GB GPUs with TP=4,
giving 16 workers). The simulator does not actually simulate GPUs; it uses
the throughput coefficients here to convert a token count into a duration.
"""

from __future__ import annotations

from dataclasses import dataclass

from saga.cache.manager import CacheManager
from saga.cache.policies import build_policy
from saga.cache.ttl import ToolLatencyEstimator, ToolTTLPolicy
from saga.core.types import Worker


@dataclass
class ClusterConfig:
    """Tunable cluster description.

    Defaults reflect 64x A100-80GB with TP=4 (16 workers, 4 GPUs each).
    """

    n_nodes: int = 8
    gpus_per_node: int = 8
    tensor_parallel: int = 4
    kv_capacity_tokens_per_worker: int = 1_500_000
    decode_tokens_per_ms: float = 38.0
    prefill_tokens_per_ms: float = 850.0

    eviction_policy: str = "walru"
    walru_alpha: float = 0.3
    walru_beta: float = 0.5
    walru_gamma: float = 0.2

    ttl_enabled: bool = True
    ttl_percentile: float = 0.95
    ttl_max_ms: float = 300_000.0
    pressure_low: float = 0.7
    pressure_high: float = 0.9

    # Per-miss stall: when an admit is a miss, the disturbed batch pays
    # this extra latency on top of the raw prefill cost (paper §2.2 "2-8x
    # latency overhead per step" --- the disturbance dominates the
    # prefill cost on a busy worker). Default calibrated from the paper's
    # Figure 1 latency breakdown.
    cache_miss_stall_ms: float = 280.0
    # Concurrent-batch multiplier: simulates PagedAttention's continuous
    # batching by treating each session as if it shares the worker with
    # this many peers. Drives the regen cost proportional to concurrency.
    concurrent_batch_size: int = 16

    # CPU-DRAM offload tier (paper §5.4).
    dram_tier_enabled: bool = False
    dram_capacity_tokens_per_worker: int = 4_000_000
    dram_contention: bool = False

    @property
    def n_workers(self) -> int:
        total_gpus = self.n_nodes * self.gpus_per_node
        return max(1, total_gpus // max(1, self.tensor_parallel))


@dataclass
class Cluster:
    """The materialized cluster: workers + per-worker cache manager."""

    workers: list[Worker]
    cache_managers: list[CacheManager]
    estimator: ToolLatencyEstimator
    config: ClusterConfig

    def worker_by_id(self, worker_id: int) -> Worker:
        return self.workers[worker_id]

    def cache_for(self, worker_id: int) -> CacheManager:
        return self.cache_managers[worker_id]


def build_cluster(cfg: ClusterConfig) -> Cluster:
    """Construct workers and a cache manager per worker."""
    n_workers = cfg.n_workers
    workers: list[Worker] = []
    managers: list[CacheManager] = []
    estimator = ToolLatencyEstimator()
    ttl_policy = (
        ToolTTLPolicy(
            estimator=estimator,
            percentile=cfg.ttl_percentile,
            ttl_max_ms=cfg.ttl_max_ms,
            pressure_low=cfg.pressure_low,
            pressure_high=cfg.pressure_high,
        )
        if cfg.ttl_enabled
        else None
    )

    for wid in range(n_workers):
        node_id = wid // max(1, cfg.gpus_per_node // cfg.tensor_parallel)
        first_gpu = (wid * cfg.tensor_parallel) % cfg.gpus_per_node
        gpu_indices = tuple(
            (first_gpu + i) % cfg.gpus_per_node for i in range(cfg.tensor_parallel)
        )
        workers.append(
            Worker(
                worker_id=wid,
                node_id=node_id,
                gpu_indices=gpu_indices,
                kv_capacity_tokens=cfg.kv_capacity_tokens_per_worker,
                decode_tokens_per_ms=cfg.decode_tokens_per_ms,
                prefill_tokens_per_ms=cfg.prefill_tokens_per_ms,
            )
        )
        policy_kwargs: dict[str, float] = {}
        if cfg.eviction_policy == "walru":
            policy_kwargs = {
                "alpha": cfg.walru_alpha,
                "beta": cfg.walru_beta,
                "gamma": cfg.walru_gamma,
            }
        policy = build_policy(cfg.eviction_policy, **policy_kwargs)
        base_mgr = CacheManager(
            worker_id=wid,
            capacity_tokens=cfg.kv_capacity_tokens_per_worker,
            policy=policy,
            ttl_policy=ttl_policy,
            pressure_low=cfg.pressure_low,
            pressure_high=cfg.pressure_high,
        )
        if cfg.dram_tier_enabled:
            from saga.cache.dram_tier import TieredCacheManager

            base_mgr = TieredCacheManager(  # type: ignore[assignment]
                base=base_mgr,
                dram_capacity_tokens=cfg.dram_capacity_tokens_per_worker,
                contended=cfg.dram_contention,
            )
        managers.append(
            base_mgr  # type: ignore[arg-type]
        )

    return Cluster(
        workers=workers,
        cache_managers=managers,
        estimator=estimator,
        config=cfg,
    )
