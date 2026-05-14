"""Tests for tool-call-aware TTL."""

from __future__ import annotations

import math

import pytest

from saga.cache.ttl import (
    ToolLatencyEstimator,
    ToolTTLPolicy,
    fit_lognormal_from_percentiles,
    percentile_of_lognormal,
)
from saga.core.types import ToolType


@pytest.mark.unit
def test_lognormal_percentile_round_trip() -> None:
    mu, sigma = fit_lognormal_from_percentiles(180.0, 2400.0)
    assert percentile_of_lognormal(mu, sigma, 0.5) == pytest.approx(180.0, rel=1e-3)
    assert percentile_of_lognormal(mu, sigma, 0.95) == pytest.approx(2400.0, rel=5e-2)


@pytest.mark.unit
def test_estimator_updates_distribution() -> None:
    est = ToolLatencyEstimator()
    for _ in range(50):
        est.update(ToolType.CODE_EXECUTION, 500.0)
    mu, sigma = est.fit(ToolType.CODE_EXECUTION)
    assert mu == pytest.approx(math.log(500.0), abs=0.5)
    assert sigma > 0.0


@pytest.mark.unit
def test_pressure_scales_ttl_down() -> None:
    est = ToolLatencyEstimator()
    policy = ToolTTLPolicy(estimator=est, percentile=0.95, ttl_max_ms=300_000.0)
    base = policy.compute_ttl_ms(ToolType.CODE_EXECUTION, used_fraction=0.5)
    high = policy.compute_ttl_ms(ToolType.CODE_EXECUTION, used_fraction=0.9)
    assert high < base


@pytest.mark.unit
def test_ttl_clamped_to_max() -> None:
    est = ToolLatencyEstimator()
    policy = ToolTTLPolicy(estimator=est, percentile=0.99, ttl_max_ms=1.0)
    val = policy.compute_ttl_ms(ToolType.WEB_API, used_fraction=0.0)
    assert val <= 1.0


@pytest.mark.unit
def test_pressure_below_low_threshold_is_zero() -> None:
    est = ToolLatencyEstimator()
    policy = ToolTTLPolicy(estimator=est, pressure_low=0.7, pressure_high=0.9)
    assert policy.memory_pressure(0.5) == 0.0
    assert policy.memory_pressure(0.7) == 0.0
    assert policy.memory_pressure(0.8) == pytest.approx(0.5)
    assert policy.memory_pressure(1.0) == 1.0
