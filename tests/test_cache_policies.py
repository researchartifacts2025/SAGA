"""Tests for the eviction policies and the cache manager."""

from __future__ import annotations

import pytest

from saga.cache.manager import CacheManager
from saga.cache.policies import (
    BeladyOracle,
    LRUPolicy,
    PolicyContext,
    PrefixLRUPolicy,
    WALRUPolicy,
    build_policy,
)
from saga.core.aeg import build_linear_aeg
from saga.core.types import KVCacheEntry, ToolType


@pytest.mark.unit
def test_build_policy_resolves_known_names() -> None:
    assert isinstance(build_policy("lru"), LRUPolicy)
    assert isinstance(build_policy("walru"), WALRUPolicy)
    assert isinstance(build_policy("lru_prefix"), PrefixLRUPolicy)
    assert isinstance(build_policy("belady"), BeladyOracle)
    with pytest.raises(ValueError):
        build_policy("nonsense")


@pytest.mark.unit
def test_lru_picks_oldest() -> None:
    mgr = CacheManager(worker_id=0, capacity_tokens=1000, policy=LRUPolicy())
    mgr.admit("s1", 200, now=0.0)
    mgr.admit("s2", 200, now=10.0)
    mgr.admit("s3", 200, now=20.0)

    # Add a big entry to force one eviction
    mgr.admit("s4", 600, now=30.0)
    assert not mgr.contains("s1")
    assert mgr.contains("s2")
    assert mgr.contains("s3")
    assert mgr.contains("s4")


@pytest.mark.unit
def test_walru_protects_predicted_reuse() -> None:
    aeg = build_linear_aeg(
        graph_id="g",
        n_steps=4,
        tool_types=[ToolType.CODE_EXECUTION] * 4,
        prompt_tokens_est=200,
        output_tokens_est=50,
        observation_tokens_est=20,
    )
    mgr = CacheManager(worker_id=0, capacity_tokens=500, policy=WALRUPolicy())
    # s1 has high reuse (early in the graph), s2 has low (terminal node)
    mgr.admit("s1", 200, now=0.0)
    mgr.register_aeg("s1", aeg, node=1)

    aeg_done = build_linear_aeg(
        graph_id="g2",
        n_steps=4,
        tool_types=[ToolType.CODE_EXECUTION] * 4,
        prompt_tokens_est=200,
        output_tokens_est=50,
        observation_tokens_est=20,
    )
    mgr.admit("s2", 200, now=5.0)
    mgr.register_aeg("s2", aeg_done, node=3)  # terminal node -> 0 reuse

    # Force an eviction; the policy should prefer evicting s2 (terminal).
    mgr.admit("s3", 200, now=10.0)
    assert mgr.contains("s1")
    assert not mgr.contains("s2")


@pytest.mark.unit
def test_belady_evicts_farthest_future_access() -> None:
    mgr = CacheManager(worker_id=0, capacity_tokens=500, policy=BeladyOracle())
    mgr.admit("s1", 200, now=0.0)
    mgr.admit("s2", 200, now=1.0)

    mgr.set_future_accesses("s1", [50.0])
    mgr.set_future_accesses("s2", [200.0])

    mgr.admit("s3", 200, now=2.0)
    assert mgr.contains("s1")
    assert not mgr.contains("s2")


@pytest.mark.unit
def test_admit_growing_entry_does_not_evict_itself() -> None:
    mgr = CacheManager(worker_id=0, capacity_tokens=600, policy=LRUPolicy())
    mgr.admit("s1", 200, now=0.0)
    mgr.admit("s1", 300, now=1.0)
    mgr.admit("s1", 500, now=2.0)
    assert mgr.contains("s1")
    assert mgr.get("s1").n_tokens == 500


@pytest.mark.unit
def test_policy_context_max_normalization() -> None:
    entries = [
        KVCacheEntry("a", 0, n_tokens=100, last_access_time=0.0, creation_time=0.0),
        KVCacheEntry("b", 0, n_tokens=500, last_access_time=5.0, creation_time=0.0),
    ]
    ctx = PolicyContext()
    ctx.with_max(entries, now=10.0)
    assert ctx.tau_max_ms == 10.0
    assert ctx.size_max_tokens == 500
