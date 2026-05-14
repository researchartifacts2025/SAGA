"""Statistical helpers: bootstrap CIs and Welch's t-test."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class TestResult:
    """Outcome of a two-sample hypothesis test."""

    statistic: float
    p_value: float
    df: float

    @property
    def stars(self) -> str:
        if self.p_value < 1e-3:
            return "***"
        if self.p_value < 1e-2:
            return "**"
        if self.p_value < 5e-2:
            return "*"
        return "n.s."


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Two-sided Welch's t-test, unequal variances."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size < 2 or b_arr.size < 2:
        return TestResult(statistic=float("nan"), p_value=1.0, df=0.0)
    result = scipy_stats.ttest_ind(a_arr, b_arr, equal_var=False)
    return TestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=float(getattr(result, "df", 0.0)),
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point, lower, upper) for the requested statistic."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)

    if statistic == "mean":
        agg = np.mean
    elif statistic == "median":
        agg = np.median
    elif statistic.startswith("p"):
        try:
            q = float(statistic[1:])
        except ValueError as exc:
            raise ValueError(f"unknown statistic {statistic!r}") from exc

        def _pct(x: np.ndarray) -> float:
            return float(np.percentile(x, q))

        agg = _pct  # type: ignore[assignment]
    else:
        raise ValueError(f"unsupported statistic {statistic!r}")

    for i in range(n_bootstrap):
        idx = rng.integers(0, arr.size, size=arr.size)
        samples[i] = agg(arr[idx])

    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    point = float(agg(arr))
    return point, low, high


def geomean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return 1.0
    return float(np.exp(np.log(arr).mean()))
