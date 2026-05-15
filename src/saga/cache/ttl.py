"""Tool-Call-Aware TTL.

When an agent pauses for a tool call we must decide how long to retain its
KV cache. Retaining too long wastes memory; evicting too early forces
regeneration. SAGA's TTL adapts the retention horizon to:

* per-tool-type latency history (log-normal fit),
* a configurable target percentile (default 95th),
* the current memory pressure on the worker.

The pressure scaling is:

    m = max(0, (used - low) / (high - low))
    pressure_factor = 1 - 0.5 * m
    ttl = min(p95(latency) * pressure_factor, TTL_max)

with low=0.7, high=0.9, TTL_max=300s.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from saga.core.types import ToolType


# ---- empirical defaults from production traces (p50 / p95 / p99 in ms) ----

_TOOL_LATENCY_DEFAULTS: dict[ToolType, tuple[float, float, float]] = {
    ToolType.CODE_EXECUTION: (180.0, 2_400.0, 28_000.0),
    ToolType.FILE_OPERATION: (45.0, 320.0, 1_200.0),
    ToolType.WEB_API: (850.0, 4_500.0, 45_000.0),
    ToolType.DATABASE_QUERY: (120.0, 890.0, 3_500.0),
    ToolType.NONE: (0.0, 0.0, 0.0),
}


def fit_lognormal_from_percentiles(p50: float, p95: float) -> tuple[float, float]:
    """Recover (mu, sigma) of a log-normal with the given p50 and p95.

    For a log-normal, ``p50 = exp(mu)`` and ``p95 = exp(mu + 1.6449 * sigma)``,
    so ``sigma = ln(p95/p50) / 1.6449``.
    """
    if p50 <= 0 or p95 <= 0 or p95 < p50:
        return 0.0, 0.0
    mu = math.log(p50)
    sigma = math.log(p95 / p50) / 1.6449
    return mu, max(sigma, 1e-6)


def percentile_of_lognormal(mu: float, sigma: float, p: float) -> float:
    """Inverse-CDF of a log-normal at probability ``p`` in (0, 1)."""
    if sigma <= 0.0:
        return math.exp(mu)
    z = _inv_phi(p)
    return math.exp(mu + sigma * z)


def _inv_phi(p: float) -> float:
    """Approximate inverse standard-normal CDF (Beasley-Springer-Moro)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


# ---------------------------------------------------------- estimator


@dataclass
class _LatencyState:
    history: deque[float]
    ema_log_mean: float
    ema_log_var: float
    n: int


class ToolLatencyEstimator:
    """Maintain per-tool log-normal latency estimates via exponential averaging.

    Cold-start values come from production-trace defaults; once we have
    observed enough samples the EMA-derived mu/sigma take over.
    """

    def __init__(self, window: int = 256, alpha: float = 0.1) -> None:
        self.window = window
        self.alpha = alpha
        self._state: dict[ToolType, _LatencyState] = {}
        for tt, (p50, p95, _p99) in _TOOL_LATENCY_DEFAULTS.items():
            mu, sigma = fit_lognormal_from_percentiles(max(p50, 1e-3), max(p95, 1e-3))
            self._state[tt] = _LatencyState(
                history=deque(maxlen=window),
                ema_log_mean=mu,
                ema_log_var=sigma * sigma,
                n=0,
            )

    def update(self, tool: ToolType, observed_ms: float) -> None:
        if observed_ms <= 0.0:
            return
        s = self._state.setdefault(
            tool,
            _LatencyState(deque(maxlen=self.window), math.log(max(observed_ms, 1.0)), 1.0, 0),
        )
        s.history.append(observed_ms)
        log_obs = math.log(max(observed_ms, 1.0))
        if s.n == 0:
            s.ema_log_mean = log_obs
            s.ema_log_var = 1.0
        else:
            prev_mean = s.ema_log_mean
            s.ema_log_mean = (1.0 - self.alpha) * prev_mean + self.alpha * log_obs
            s.ema_log_var = (1.0 - self.alpha) * s.ema_log_var + self.alpha * (
                (log_obs - s.ema_log_mean) ** 2
            )
        s.n += 1

    def fit(self, tool: ToolType) -> tuple[float, float]:
        s = self._state.get(tool)
        if s is None:
            return 0.0, 0.0
        return s.ema_log_mean, math.sqrt(max(s.ema_log_var, 1e-6))

    def percentile(self, tool: ToolType, p: float) -> float:
        mu, sigma = self.fit(tool)
        return percentile_of_lognormal(mu, sigma, p)

    def mean(self, tool: ToolType) -> float:
        mu, sigma = self.fit(tool)
        return math.exp(mu + 0.5 * sigma * sigma)


# ------------------------------------------------------------- TTL


class ToolTTLPolicy:
    """Compute a TTL for a session that has just initiated a tool call.

    Algorithm:

        1. Read the configured percentile of the per-tool latency distribution.
        2. Multiply by ``1 - 0.5 * memory_pressure`` where pressure is in [0, 1].
        3. Clamp to ``ttl_max_ms``.
    """

    def __init__(
        self,
        estimator: ToolLatencyEstimator,
        percentile: float = 0.95,
        ttl_max_ms: float = 300_000.0,
        pressure_low: float = 0.7,
        pressure_high: float = 0.9,
        pressure_scale: float = 0.5,
    ) -> None:
        if not (0.0 < percentile < 1.0):
            raise ValueError("percentile must be in (0, 1)")
        if pressure_high <= pressure_low:
            raise ValueError("pressure_high must exceed pressure_low")
        self.estimator = estimator
        self.percentile = percentile
        self.ttl_max_ms = ttl_max_ms
        self.pressure_low = pressure_low
        self.pressure_high = pressure_high
        self.pressure_scale = pressure_scale

    # ------------------------------------------- pressure

    def memory_pressure(self, used_fraction: float) -> float:
        if used_fraction <= self.pressure_low:
            return 0.0
        denom = max(1e-9, self.pressure_high - self.pressure_low)
        return max(0.0, min(1.0, (used_fraction - self.pressure_low) / denom))

    # ------------------------------------------- ttl

    def compute_ttl_ms(self, tool: ToolType, used_fraction: float) -> float:
        if tool == ToolType.NONE:
            return self.ttl_max_ms
        base = self.estimator.percentile(tool, self.percentile)
        pressure = self.memory_pressure(used_fraction)
        factor = 1.0 - self.pressure_scale * pressure
        ttl = base * factor
        return float(min(max(ttl, 0.0), self.ttl_max_ms))
