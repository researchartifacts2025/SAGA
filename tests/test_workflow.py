"""Tests for the framework-hint parser and pattern-inference engine."""

from __future__ import annotations

import pytest

from saga.core.types import ToolType
from saga.workflow import (
    FrameworkHint,
    FrameworkHintParser,
    PatternInferenceEngine,
)


@pytest.mark.unit
def test_framework_hint_to_aeg_round_trip() -> None:
    parser = FrameworkHintParser()
    hints = [
        FrameworkHint(0, "read_file", 2000, 200, observation_tokens=120),
        FrameworkHint(1, "shell", 2200, 180, observation_tokens=400),
        FrameworkHint(2, "write_file", 2400, 200, observation_tokens=80),
        FrameworkHint(3, "finish", 2400, 50, observation_tokens=0),
    ]
    aeg = parser.to_aeg(hints, graph_id="t/0")
    assert len(aeg) == 4
    assert aeg.is_terminal(3)
    assert aeg.node(0).tool_type == ToolType.FILE_OPERATION
    assert aeg.node(1).tool_type == ToolType.CODE_EXECUTION


@pytest.mark.unit
def test_langchain_callback_parsing() -> None:
    parser = FrameworkHintParser()
    events = [
        {"event": "on_chain_start", "tool": "shell", "prompt_tokens": 100, "output_tokens": 10},
        {"event": "on_tool_end", "observation_tokens": 50},
        {"event": "on_chain_start", "tool": "read_file", "prompt_tokens": 110, "output_tokens": 12},
        {"event": "on_tool_end", "observation_tokens": 60},
    ]
    hints = parser.from_langchain_callbacks(events)
    assert len(hints) == 2
    assert hints[0].tool_name == "shell"
    assert hints[1].tool_name == "read_file"


@pytest.mark.unit
def test_pattern_inference_cold_start_guard() -> None:
    engine = PatternInferenceEngine(cold_start_tasks=5)
    assert not engine.has_warmed_up("agent_a")
    assert engine.infer_aeg("agent_a", graph_id="g", n_steps=3) is None


@pytest.mark.unit
def test_pattern_inference_warmup_and_predict() -> None:
    engine = PatternInferenceEngine(cold_start_tasks=3, theta_conf=0.5)
    for _ in range(5):
        engine.observe_session(
            "swe_agent",
            [ToolType.CODE_EXECUTION, ToolType.FILE_OPERATION, ToolType.CODE_EXECUTION],
        )
    assert engine.has_warmed_up("swe_agent")
    aeg = engine.infer_aeg("swe_agent", graph_id="g", n_steps=4)
    assert aeg is not None
    assert len(aeg) == 4
    assert aeg.is_terminal(3)


@pytest.mark.unit
def test_pattern_inference_accuracy_estimate() -> None:
    engine = PatternInferenceEngine(cold_start_tasks=1, theta_conf=0.6)
    engine.observe_session("a", [ToolType.WEB_API] * 10)
    acc = engine.accuracy_estimate("a")
    # All transitions are WEB_API -> WEB_API with P=1.0
    assert acc > 0.9
