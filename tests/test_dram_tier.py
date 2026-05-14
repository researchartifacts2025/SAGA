"""Tests for the CPU-DRAM offload tier."""

from __future__ import annotations

import pytest

from saga.cache.dram_tier import DRAMPool, SwapTimeModel, TieredCacheManager
from saga.cache.manager import CacheManager
from saga.cache.policies import LRUPolicy
from saga.core.types import KVCacheEntry


@pytest.mark.unit
def test_swap_time_scales_with_token_count() -> None:
    model = SwapTimeModel()
    t1 = model.transfer_ms(1000)
    t2 = model.transfer_ms(10000)
    assert t2 > t1
    assert model.transfer_ms(0) == 0.0


@pytest.mark.unit
def test_dram_pool_admit_and_evict() -> None:
    pool = DRAMPool(capacity_tokens=500)
    e1 = KVCacheEntry("s1", 0, 200, 0.0, 0.0)
    e2 = KVCacheEntry("s2", 0, 200, 1.0, 1.0)
    pool.admit(e1, now=0.0)
    pool.admit(e2, now=1.0)
    assert pool.used_tokens == 400
    e3 = KVCacheEntry("s3", 0, 200, 2.0, 2.0)
    pool.admit(e3, now=2.0)
    # s1 (LRU) should have been evicted to make room
    assert not pool.contains("s1")
    assert pool.contains("s3")


@pytest.mark.unit
def test_tiered_manager_swaps_in_on_dram_hit() -> None:
    base = CacheManager(worker_id=0, capacity_tokens=300, policy=LRUPolicy())
    tier = TieredCacheManager(base=base, dram_capacity_tokens=1000)

    tier.admit("s1", 200, now=0.0)
    assert tier.hbm_contains("s1")

    # Force a HBM eviction of s1 by admitting two more sessions.
    tier.admit("s2", 200, now=10.0)
    tier.admit("s3", 200, now=20.0)
    assert not tier.hbm_contains("s1")
    assert tier.dram_contains("s1")

    # Now bring s1 back via swap-in.
    decision = tier.admit("s1", 200, now=30.0)
    assert decision.hit is True
    assert tier.cumulative_swap_ms > 0.0


@pytest.mark.unit
def test_tiered_manager_records_stats() -> None:
    base = CacheManager(worker_id=0, capacity_tokens=300, policy=LRUPolicy())
    tier = TieredCacheManager(base=base, dram_capacity_tokens=600)
    tier.admit("s1", 200, now=0.0)
    tier.admit("s2", 200, now=1.0)
    tier.admit("s3", 200, now=2.0)
    s = tier.stats()
    assert "dram_used_fraction" in s
    assert "cumulative_swap_ms" in s
