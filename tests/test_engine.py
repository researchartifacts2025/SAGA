"""End-to-end integration tests for the simulator."""

from __future__ import annotations

import pytest

from saga.presets import get_preset
from saga.sim.engine import EngineConfig, SimulatorEngine
from saga.workload import build_workload
from saga.workload.base import WorkloadSpec


def _build_engine(
    preset_name: str, seed: int = 0, horizon_ms: float = 600_000.0
) -> SimulatorEngine:
    preset = get_preset(preset_name)
    return SimulatorEngine(
        preset.cluster,
        preset.coordinator,
        EngineConfig(seed=seed, horizon_ms=horizon_ms, label=preset_name),
    )


@pytest.mark.integration
def test_engine_completes_simple_run() -> None:
    spec = WorkloadSpec(n_tasks=10, seed=0)
    gen = build_workload("swe_bench", spec=spec)
    templates = [tmpl for _t, tmpl in gen.stream()]

    engine = _build_engine("saga")
    engine.admit(templates)
    result = engine.run()

    assert sum(1 for t in result.tasks if t.is_complete) > 0
    assert result.sim_time_ms > 0.0


@pytest.mark.integration
def test_saga_beats_vllm_on_cache_hit_rate() -> None:
    spec = WorkloadSpec(n_tasks=20, seed=42)
    templates = [tmpl for _t, tmpl in build_workload("swe_bench", spec=spec).stream()]
    saga = _build_engine("saga", seed=42)
    saga.admit(templates)
    res_saga = saga.run()

    spec2 = WorkloadSpec(n_tasks=20, seed=42)
    templates2 = [tmpl for _t, tmpl in build_workload("swe_bench", spec=spec2).stream()]
    vllm = _build_engine("vllm", seed=42)
    vllm.admit(templates2)
    res_vllm = vllm.run()

    # The headline mechanism: SAGA's session affinity keeps the hit rate at
    # least as high as vanilla vLLM's least-loaded routing. The absolute TCT
    # gap depends on per-worker concurrency, which a single-session-per-step
    # simulator only partially captures.
    assert res_saga.cache_hit_rate >= res_vllm.cache_hit_rate
    assert res_saga.tct_seconds() and res_vllm.tct_seconds()


@pytest.mark.integration
def test_engine_deterministic_under_seed() -> None:
    spec = WorkloadSpec(n_tasks=5, seed=7)
    templates = [tmpl for _t, tmpl in build_workload("swe_bench", spec=spec).stream()]

    e1 = _build_engine("saga", seed=11)
    e1.admit(templates)
    r1 = e1.run()

    e2 = _build_engine("saga", seed=11)
    spec2 = WorkloadSpec(n_tasks=5, seed=7)
    templates2 = [tmpl for _t, tmpl in build_workload("swe_bench", spec=spec2).stream()]
    e2.admit(templates2)
    r2 = e2.run()

    assert r1.regenerated_tokens == r2.regenerated_tokens
    assert r1.tokens_admitted == r2.tokens_admitted
