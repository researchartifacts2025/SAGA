"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from saga.utils.seeds import RNG


@pytest.fixture
def rng() -> RNG:
    return RNG(seed=42)
