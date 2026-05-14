"""Tests for the optional native acceleration layer.

These tests exercise both the C++ path (when the compiled extension is
available) and the pure-Python fallback. Behaviour must be identical
either way.
"""

from __future__ import annotations

import pytest

from saga import is_native_available, native_build_info
from saga.cache.manager import CacheManager
from saga.cache.policies import BeladyOracle, WALRUPolicy
from saga.core.aeg import build_linear_aeg
from saga.core.types import ToolType
from saga.native import NativeSessionTable, walru_select_victim


@pytest.mark.unit
def test_native_build_info_runs() -> None:
    info = native_build_info()
    assert isinstance(info, str)
    assert len(info) > 0


@pytest.mark.unit
def test_walru_native_or_fallback_picks_terminal_first() -> None:
    aeg_active = build_linear_aeg(
        graph_id="g",
        n_steps=5,
        tool_types=[ToolType.CODE_EXECUTION] * 5,
        prompt_tokens_est=200,
        output_tokens_est=50,
        observation_tokens_est=20,
    )
    aeg_done = build_linear_aeg(
        graph_id="g2",
        n_steps=5,
        tool_types=[ToolType.CODE_EXECUTION] * 5,
        prompt_tokens_est=200,
        output_tokens_est=50,
        observation_tokens_est=20,
    )

    for use_native in (True, False):
        mgr = CacheManager(
            worker_id=0,
            capacity_tokens=500,
            policy=WALRUPolicy(use_native=use_native),
        )
        mgr.admit("s1", 200, now=0.0)
        mgr.register_aeg("s1", aeg_active, node=1)
        mgr.admit("s2", 200, now=5.0)
        mgr.register_aeg("s2", aeg_done, node=4)

        mgr.admit("s3", 200, now=10.0)
        assert mgr.contains("s1")
        assert not mgr.contains("s2")


@pytest.mark.unit
def test_belady_native_or_fallback_picks_farthest_future() -> None:
    for use_native in (True, False):
        mgr = CacheManager(
            worker_id=0,
            capacity_tokens=500,
            policy=BeladyOracle(use_native=use_native),
        )
        mgr.admit("s1", 200, now=0.0)
        mgr.admit("s2", 200, now=1.0)
        mgr.set_future_accesses("s1", [50.0])
        mgr.set_future_accesses("s2", [200.0])
        mgr.admit("s3", 200, now=2.0)
        assert mgr.contains("s1")
        assert not mgr.contains("s2")


@pytest.mark.unit
def test_native_session_table_basic() -> None:
    table = NativeSessionTable()
    assert table.get("s1") is None
    table.set("s1", 7)
    assert table.get("s1") == 7
    assert len(table) >= 1
    table.set("s1", 11)
    assert table.get("s1") == 11
    table.erase("s1")
    assert table.get("s1") is None


@pytest.mark.unit
def test_walru_low_level_helper() -> None:
    class _E:
        def __init__(self, sid, tokens, t):
            self.session_id = sid
            self.n_tokens = tokens
            self.last_access_time = t
            self.pinned = False

    entries = [_E("a", 200, 0.0), _E("b", 200, 10.0)]
    idx = walru_select_victim(
        entries,
        now=20.0,
        tau_max=20.0,
        size_max=200,
        alpha=0.3,
        beta=0.5,
        gamma=0.2,
        reuse_lookup=lambda sid: 0.0,
    )
    # entry 'a' has older last_access -> more evictable -> idx 0
    assert idx == 0


@pytest.mark.unit
def test_is_native_available_returns_bool() -> None:
    assert isinstance(is_native_available(), bool)
