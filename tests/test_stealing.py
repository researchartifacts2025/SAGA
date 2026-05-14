"""Tests for work stealing."""

from __future__ import annotations

import pytest

from saga.core.types import Worker
from saga.scheduler.stealing import WorkStealer
from saga.utils.seeds import RNG


def _w(wid: int, queue_depth: int, busy: float = 0.0) -> Worker:
    w = Worker(
        worker_id=wid,
        node_id=0,
        gpu_indices=(wid,),
        kv_capacity_tokens=10_000,
        decode_tokens_per_ms=40.0,
        prefill_tokens_per_ms=800.0,
    )
    w.queue_depth = queue_depth
    w.cumulative_busy_ms = busy
    w.busy_until = busy
    return w


@pytest.mark.unit
def test_no_steal_when_load_balanced() -> None:
    workers = [_w(i, queue_depth=2) for i in range(4)]
    queues = {i: ["a", "b"] for i in range(4)}
    stealer = WorkStealer(t_idle_ms=100.0, r_max=2.0)
    actions = stealer.step(now=0.0, workers=workers, queues=queues, rng=RNG(seed=0))
    assert actions == []


@pytest.mark.unit
def test_steal_fires_when_idle_long_enough() -> None:
    workers = [_w(0, queue_depth=0), _w(1, queue_depth=5, busy=200.0)]
    queues = {0: [], 1: ["s1", "s2", "s3", "s4", "s5"]}
    stealer = WorkStealer(t_idle_ms=50.0, r_max=10.0)
    stealer.step(now=0.0, workers=workers, queues=queues, rng=RNG(seed=0))
    actions = stealer.step(now=100.0, workers=workers, queues=queues, rng=RNG(seed=0))
    assert any(a.success for a in actions)


@pytest.mark.unit
def test_migration_sampler_within_bounds() -> None:
    stealer = WorkStealer(migration_mean_ms=230.0, migration_p95_ms=890.0)
    rng = RNG(seed=42)
    samples = [stealer.sample_migration_ms(rng) for _ in range(200)]
    assert min(samples) > 0.0
    # The empirical mean should be in the right ballpark.
    mean = sum(samples) / len(samples)
    assert 100.0 < mean < 1500.0
