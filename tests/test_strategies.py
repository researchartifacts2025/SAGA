"""Tests for the BFS/DFS/Hybrid queue strategies."""

from __future__ import annotations

import pytest

from saga.scheduler.strategies import (
    BFSStrategy,
    DFSStrategy,
    HybridStrategy,
    build_strategy,
)


@pytest.mark.unit
def test_bfs_pops_oldest() -> None:
    q = ["a", "b", "c"]
    s = BFSStrategy()
    assert s.pick(q) == "a"
    assert q == ["b", "c"]


@pytest.mark.unit
def test_dfs_pops_newest() -> None:
    q = ["a", "b", "c"]
    s = DFSStrategy()
    assert s.pick(q) == "c"
    assert q == ["a", "b"]


@pytest.mark.unit
def test_hybrid_picks_highest_score() -> None:
    q = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 5.0, "c": 2.0}
    s = HybridStrategy()
    assert s.pick(q, scoring_fn=scores.__getitem__) == "b"


@pytest.mark.unit
def test_empty_queue_returns_none() -> None:
    for strat in (BFSStrategy(), DFSStrategy(), HybridStrategy()):
        assert strat.pick([]) is None


@pytest.mark.unit
def test_build_strategy_factory() -> None:
    assert isinstance(build_strategy("bfs"), BFSStrategy)
    assert isinstance(build_strategy("dfs"), DFSStrategy)
    assert isinstance(build_strategy("hybrid"), HybridStrategy)
    with pytest.raises(ValueError):
        build_strategy("nonsense")
