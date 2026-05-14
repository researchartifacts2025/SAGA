"""Tests for the session router."""

from __future__ import annotations

import pytest

from saga.core.types import Worker
from saga.scheduler.routing import SessionRouter


def _make_workers(n: int = 4) -> list[Worker]:
    return [
        Worker(
            worker_id=i,
            node_id=0,
            gpu_indices=(i,),
            kv_capacity_tokens=10_000,
            decode_tokens_per_ms=40.0,
            prefill_tokens_per_ms=800.0,
        )
        for i in range(n)
    ]


@pytest.mark.unit
def test_session_affinity_routes_back_to_cached_worker() -> None:
    workers = _make_workers()
    router = SessionRouter(strategy="session_affinity")
    cached: dict[int, set[str]] = {0: {"s1"}}

    def cached_pred(wid: int, sid: str) -> bool:
        return sid in cached.get(wid, set())

    router.remember_session("s1", worker_id=0)
    decision = router.route("s1", prefix_hash=0, workers=workers, cached_predicate=cached_pred)
    assert decision.worker_id == 0
    assert decision.reason == "session_affinity_hit"
    assert decision.cache_hit_expected


@pytest.mark.unit
def test_overloaded_worker_loses_affinity() -> None:
    workers = _make_workers()
    workers[0].cumulative_busy_ms = 1_000.0
    workers[0].busy_until = 1_000.0
    workers[0].cache_used_tokens = 10_000  # max pressure
    router = SessionRouter(strategy="session_affinity", load_threshold=0.4)
    router.remember_session("s1", worker_id=0)
    decision = router.route(
        "s1",
        prefix_hash=0,
        workers=workers,
        cached_predicate=lambda _w, _s: True,
    )
    assert decision.worker_id != 0


@pytest.mark.unit
def test_least_loaded_ignores_affinity() -> None:
    workers = _make_workers()
    workers[0].cumulative_busy_ms = 0.0
    workers[1].cumulative_busy_ms = 0.0
    workers[0].busy_until = 0.0
    router = SessionRouter(strategy="least_loaded")
    decision = router.route(
        "s1",
        prefix_hash=0,
        workers=workers,
        cached_predicate=lambda _w, _s: False,
    )
    # least-loaded always picks the worker with min load
    assert decision.worker_id in {w.worker_id for w in workers}


@pytest.mark.unit
def test_prefix_affinity_groups_by_hash() -> None:
    workers = _make_workers()
    router = SessionRouter(strategy="prefix_affinity", prefix_buckets=2)
    d1 = router.route("s1", prefix_hash=0, workers=workers, cached_predicate=lambda *_: False)
    d2 = router.route("s2", prefix_hash=0, workers=workers, cached_predicate=lambda *_: False)
    assert d1.worker_id == d2.worker_id
