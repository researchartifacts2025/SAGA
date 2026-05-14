"""KV-cache management.

Policies (LRU, LRU+Prefix, WA-LRU, Belady oracle), tool-call-aware TTL, and
the per-worker cache manager that ties them together.
"""

from saga.cache.dram_tier import DRAMPool, SwapTimeModel, TieredCacheManager
from saga.cache.manager import CacheDecision, CacheManager
from saga.cache.policies import (
    BeladyOracle,
    EvictionPolicy,
    LRUPolicy,
    PrefixLRUPolicy,
    WALRUPolicy,
    build_policy,
)
from saga.cache.ttl import ToolLatencyEstimator, ToolTTLPolicy


__all__ = [
    "BeladyOracle",
    "CacheDecision",
    "CacheManager",
    "DRAMPool",
    "EvictionPolicy",
    "LRUPolicy",
    "PrefixLRUPolicy",
    "SwapTimeModel",
    "TieredCacheManager",
    "ToolLatencyEstimator",
    "ToolTTLPolicy",
    "WALRUPolicy",
    "build_policy",
]
