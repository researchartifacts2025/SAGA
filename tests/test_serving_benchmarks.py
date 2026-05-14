"""Tests for the wall-clock benchmark harness and the canonical paper YAML."""

from __future__ import annotations

import pytest

from saga.serving.benchmarks import (
    BenchmarkConfig,
    PaperResults,
    WallClockBenchmark,
    load_paper_results,
)


@pytest.mark.unit
def test_paper_yaml_loads_and_has_expected_keys() -> None:
    paper = load_paper_results()
    assert isinstance(paper, PaperResults)
    for k in ("cluster", "model", "software", "e2e", "competitive_ratio",
              "ablation_swe_bench", "slo_attainment_pct"):
        assert k in paper.raw
    assert paper.raw["model"]["name"] == "Llama-3-70B-Instruct"
    assert paper.raw["cluster"]["total_gpus"] == 64


@pytest.mark.unit
def test_paper_results_e2e_accessors() -> None:
    paper = load_paper_results()
    mean, std = paper.tct("saga", workload="swe_bench")
    assert mean == 203.4
    assert std == 12.8
    assert paper.memory_pct("saga", workload="swe_bench") == 71.3
    assert paper.speedup_over("vllm", workload="swe_bench") == 3.01
    assert paper.geomean_speedup_vs_vllm_apc == 1.64


@pytest.mark.unit
def test_paper_results_competitive_ratio() -> None:
    paper = load_paper_results()
    assert paper.competitive("walru", workload="swe_bench") == 1.31
    assert paper.competitive("lru", workload="swe_bench") == 2.84


@pytest.mark.unit
def test_paper_results_slo_attainment() -> None:
    paper = load_paper_results()
    assert paper.slo_attainment("saga", tenant_class="overall") == 99.2
    assert paper.slo_attainment("vllm", tenant_class="light") == 43.2


@pytest.mark.unit
def test_wall_clock_benchmark_paper_mode_emits_one_per_system_workload() -> None:
    bench = WallClockBenchmark(cfg=BenchmarkConfig(mode="paper"))
    results = bench.run()
    # 7 systems x 2 workloads = 14 rows
    assert len(results) == 14
    assert all(r.source == "paper_yaml" for r in results)
    saga_swe = next(r for r in results if r.system == "saga" and r.workload == "swe_bench")
    assert saga_swe.tct_mean_s == 203.4
    assert saga_swe.tct_std_s == 12.8


@pytest.mark.unit
def test_wall_clock_benchmark_auto_mode_falls_back_to_paper_without_serving_stack() -> None:
    bench = WallClockBenchmark(cfg=BenchmarkConfig(mode="auto"))
    results = bench.run()
    # Without vllm/ray/torch.cuda the auto resolver should pick "paper".
    if any(r.source == "wall_clock_cluster" for r in results):
        # CI has a real cluster -- accept either outcome.
        return
    assert all(r.source == "paper_yaml" for r in results)


@pytest.mark.unit
def test_format_results_returns_one_line_per_result() -> None:
    bench = WallClockBenchmark(cfg=BenchmarkConfig(mode="paper",
                                                    systems=("saga",),
                                                    workloads=("swe_bench",)))
    results = bench.run()
    text = WallClockBenchmark.format(results)
    assert "saga" in text
    assert "TCT=" in text
    assert "paper_yaml" in text
