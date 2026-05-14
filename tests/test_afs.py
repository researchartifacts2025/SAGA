"""Tests for Agent Fair Share."""

from __future__ import annotations

import pytest

from saga.core.types import Task
from saga.fairness.afs import AFSScheduler


def _task(tenant: str, task_id: str, expected_ms: float = 10_000.0) -> Task:
    return Task(
        task_id=f"{tenant}/{task_id}",
        tenant_id=tenant,
        workload_kind="test",
        submit_time=0.0,
        n_steps=10,
        aeg_id="g",
        expected_tct_ms=expected_ms,
    )


@pytest.mark.unit
def test_priority_zero_when_no_pending() -> None:
    afs = AFSScheduler()
    assert afs.priority("tenant_x", now=0.0) == 0.0


@pytest.mark.unit
def test_higher_workload_means_higher_priority_at_same_deadline() -> None:
    afs = AFSScheduler()
    # Both tenants have the same wall-clock deadline (10_000 ms),
    # but "heavy" has more remaining work, so it should have higher urgency.
    afs.note_submit("light", _task("light", "1", expected_ms=2_000.0))
    afs.note_submit("heavy", _task("heavy", "1", expected_ms=50_000.0))
    # Override the deadlines to be identical so urgency reflects workload only.
    afs._task_deadlines["light/1"] = 10_000.0
    afs._task_deadlines["heavy/1"] = 10_000.0
    afs.refresh(now=0.0)
    assert afs.priority("heavy", now=0.0) > afs.priority("light", now=0.0)


@pytest.mark.unit
def test_progress_reduces_urgency() -> None:
    afs = AFSScheduler()
    afs.note_submit("a", _task("a", "1", expected_ms=20_000.0))
    afs.refresh(now=0.0)
    before = afs.priority("a", now=0.0)
    afs.note_progress("a", "a/1", gpu_ms=10_000.0)
    afs.refresh(now=0.0)
    after = afs.priority("a", now=0.0)
    assert after < before


@pytest.mark.unit
def test_completion_clears_pending() -> None:
    afs = AFSScheduler()
    t = _task("a", "1", expected_ms=5_000.0)
    afs.note_submit("a", t)
    afs.note_complete("a", t, now=10_000.0)
    afs.refresh(now=10_000.0)
    assert afs.priority("a", now=10_000.0) == 0.0


@pytest.mark.unit
def test_allocation_sums_to_one() -> None:
    afs = AFSScheduler()
    afs.note_submit("a", _task("a", "1"))
    afs.note_submit("b", _task("b", "1"))
    afs.refresh(now=0.0)
    alloc = afs.allocation(now=0.0)
    assert sum(alloc.values()) == pytest.approx(1.0)
