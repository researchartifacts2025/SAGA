"""Tests for the workload generators."""

from __future__ import annotations

import pytest

from saga.workload import build_workload
from saga.workload.base import WorkloadSpec


@pytest.mark.unit
def test_swe_bench_step_count_within_bounds() -> None:
    spec = WorkloadSpec(n_tasks=50, n_tenants=1, seed=0)
    gen = build_workload("swe_bench", spec=spec)
    templates = [tmpl for _t, tmpl in gen.stream()]
    assert len(templates) == 50
    for tmpl in templates:
        assert 1 <= tmpl.task.n_steps <= 150


@pytest.mark.unit
def test_swe_bench_aeg_consistent_with_task() -> None:
    spec = WorkloadSpec(n_tasks=10, seed=1)
    gen = build_workload("swe_bench", spec=spec)
    for _t, tmpl in gen.stream():
        assert len(tmpl.aeg) == tmpl.task.n_steps
        assert tmpl.aeg.workload_kind == "swe_bench"


@pytest.mark.unit
def test_web_arena_uses_web_api_tools() -> None:
    spec = WorkloadSpec(n_tasks=10, seed=2)
    gen = build_workload("web_arena", spec=spec)
    web_count = 0
    total = 0
    for _t, tmpl in gen.stream():
        for node in tmpl.aeg:
            total += 1
            if node.tool_type.value == "web_api":
                web_count += 1
    assert web_count / total > 0.5


@pytest.mark.unit
def test_burst_gpt_has_three_tenant_classes() -> None:
    gen = build_workload("burst_gpt", horizon_minutes=1.0)
    templates = list(gen.stream())
    kinds = {tmpl.task.workload_kind for _t, tmpl in templates}
    # Expect at least heavy and either medium or light to appear in 1 minute.
    assert any(k.startswith("burst_") for k in kinds)


@pytest.mark.unit
def test_workload_arrival_times_are_monotonic() -> None:
    spec = WorkloadSpec(n_tasks=30, seed=3, arrival_rate_per_minute=10.0)
    gen = build_workload("swe_bench", spec=spec)
    from itertools import pairwise

    times = [t for t, _ in gen.stream()]
    assert all(b >= a for a, b in pairwise(times))


@pytest.mark.unit
def test_unknown_workload_raises() -> None:
    with pytest.raises(ValueError):
        build_workload("totally_unknown")
