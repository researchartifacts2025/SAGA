"""RNG helpers.

SAGA's simulator is fully deterministic given a seed: workload generation,
tool durations, work-stealing victim selection, and AEG construction all draw
from a single explicit RNG threaded through the call stack. This module
isolates the RNG so the rest of the code never imports ``random`` directly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


_MASK32 = 0xFFFF_FFFF


def hash_to_seed(*parts: object) -> int:
    """Derive a stable 32-bit seed from the given string parts.

    Uses BLAKE2b under the hood; the same arguments always yield the same seed,
    across Python sessions and machines.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little") & _MASK32


def derive_seed(base: int, *parts: object) -> int:
    """Combine a base seed with extra parts to get a child seed."""
    return hash_to_seed(base, *parts)


class RNG:
    """A thin wrapper around ``numpy.random.Generator``.

    The wrapper restricts the API to the distributions actually used by SAGA
    so calls are easy to grep for and to mock in tests.
    """

    __slots__ = ("_gen", "seed")

    def __init__(self, seed: int) -> None:
        self.seed = int(seed) & _MASK32
        self._gen = np.random.default_rng(self.seed)

    # ---------------- atoms

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self._gen.uniform(low, high))

    def randint(self, low: int, high: int) -> int:
        return int(self._gen.integers(low=low, high=high, endpoint=False))

    def normal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return float(self._gen.normal(mean, sigma))

    def lognormal(self, mean_log: float, sigma_log: float) -> float:
        return float(self._gen.lognormal(mean_log, sigma_log))

    def exponential(self, scale: float) -> float:
        return float(self._gen.exponential(scale))

    def poisson(self, lam: float) -> int:
        return int(self._gen.poisson(lam))

    def gamma(self, shape: float, scale: float) -> float:
        return float(self._gen.gamma(shape, scale))

    # --------------- choices

    def choice(self, options: Sequence[object], probs: Sequence[float] | None = None) -> object:
        if not options:
            raise ValueError("choice over empty sequence")
        if probs is None:
            idx = int(self._gen.integers(0, len(options)))
            return options[idx]
        probs_arr = np.asarray(probs, dtype=np.float64)
        total = float(probs_arr.sum())
        if total <= 0.0:
            idx = int(self._gen.integers(0, len(options)))
            return options[idx]
        probs_arr = probs_arr / total
        idx = int(self._gen.choice(len(options), p=probs_arr))
        return options[idx]

    def shuffle(self, items: list[object]) -> None:
        self._gen.shuffle(items)

    # --------------- child RNGs

    def fork(self, *parts: object) -> RNG:
        return RNG(derive_seed(self.seed, *parts))

    def __repr__(self) -> str:
        return f"RNG(seed={self.seed})"
