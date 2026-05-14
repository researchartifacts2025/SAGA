"""Tests for the Agent Execution Graph data structure."""

from __future__ import annotations

import pytest

from saga.core.aeg import build_branching_aeg, build_linear_aeg
from saga.core.types import ToolType


@pytest.mark.unit
def test_linear_aeg_basic_shape() -> None:
    aeg = build_linear_aeg(
        graph_id="t1",
        n_steps=5,
        tool_types=[ToolType.CODE_EXECUTION] * 5,
        prompt_tokens_est=2000,
        output_tokens_est=200,
        observation_tokens_est=300,
    )
    assert len(aeg) == 5
    assert aeg.is_terminal(4)
    assert not aeg.is_terminal(0)
    assert aeg.most_likely_successor(0) == 1
    assert aeg.most_likely_successor(4) is None


@pytest.mark.unit
def test_aeg_reuse_prediction_in_range() -> None:
    aeg = build_linear_aeg(
        graph_id="t2",
        n_steps=3,
        tool_types=[ToolType.FILE_OPERATION] * 3,
        prompt_tokens_est=1000,
        output_tokens_est=100,
        observation_tokens_est=200,
    )
    reuse = aeg.predict_reuse(0, cached_tokens=1000)
    assert 0.0 <= reuse <= 1.0
    assert reuse > 0.5  # cached >> obs_est


@pytest.mark.unit
def test_aeg_remaining_work_monotonic() -> None:
    aeg = build_linear_aeg(
        graph_id="t3",
        n_steps=10,
        tool_types=[ToolType.WEB_API] * 10,
        prompt_tokens_est=2000,
        output_tokens_est=300,
        observation_tokens_est=400,
    )
    pre_full, dec_full = aeg.remaining_work_tokens(0)
    pre_half, dec_half = aeg.remaining_work_tokens(5)
    assert pre_full > pre_half
    assert dec_full > dec_half


@pytest.mark.unit
def test_branching_aeg_has_internal_edges() -> None:
    aeg = build_branching_aeg(
        graph_id="b1",
        depth=2,
        branching=2,
        tool_type=ToolType.WEB_API,
        prompt_tokens_est=1000,
        output_tokens_est=100,
        observation_tokens_est=200,
    )
    # depth 2, branch 2 -> 1 + 2 + 4 = 7 nodes
    assert len(aeg) == 7
    assert len(aeg.edges) == 6
    assert aeg.is_terminal(len(aeg) - 1)


@pytest.mark.unit
def test_linear_path_walks_to_terminal() -> None:
    aeg = build_linear_aeg(
        graph_id="t4",
        n_steps=4,
        tool_types=[ToolType.CODE_EXECUTION] * 4,
        prompt_tokens_est=1000,
        output_tokens_est=100,
        observation_tokens_est=200,
    )
    path = aeg.linear_path()
    assert path == [0, 1, 2, 3]
