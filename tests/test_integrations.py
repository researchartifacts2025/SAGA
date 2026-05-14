"""Tests for the LangChain / AutoGen / CrewAI bridges."""

from __future__ import annotations

import pytest

from saga.core.types import ToolType
from saga.integrations import AutoGenAdapter, CrewAIAdapter, LangChainAdapter
from saga.workflow.pattern import PatternInferenceEngine


@pytest.mark.unit
def test_langchain_adapter_records_and_emits_aeg() -> None:
    adapter = LangChainAdapter(agent_type="swe_agent")
    rid = adapter.record_chain_start(prompt_tokens=2_400)
    adapter.record_llm_end(rid, output_tokens=180)
    adapter.record_tool_end(rid, tool_name="shell", observation_tokens=350)

    rid2 = adapter.record_chain_start(prompt_tokens=2_500)
    adapter.record_llm_end(rid2, output_tokens=200)
    adapter.record_tool_end(rid2, tool_name="read_file", observation_tokens=120)

    aeg = adapter.emit_aeg(graph_id="t/0")
    assert len(aeg) == 2
    assert aeg.node(0).tool_type == ToolType.CODE_EXECUTION
    assert aeg.node(1).tool_type == ToolType.FILE_OPERATION


@pytest.mark.unit
def test_langchain_adapter_feeds_pattern_engine() -> None:
    engine = PatternInferenceEngine(cold_start_tasks=1, theta_conf=0.5)
    adapter = LangChainAdapter(agent_type="agent_x", pattern_engine=engine)
    for i in range(3):
        rid = adapter.record_chain_start(prompt_tokens=1_000 + i)
        adapter.record_llm_end(rid, output_tokens=100)
        adapter.record_tool_end(rid, tool_name="shell", observation_tokens=200)
    adapter.emit_aeg()
    assert engine.has_warmed_up("agent_x")


@pytest.mark.unit
def test_autogen_adapter_consumes_messages() -> None:
    adapter = AutoGenAdapter(agent_type="autogen_test")
    msgs = [
        {"role": "assistant", "content": "running tests", "tool_calls": [{"name": "shell"}]},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "reading file", "tool_calls": [{"name": "read_file"}]},
    ]
    aeg = adapter.build_aeg(msgs, graph_id="ag/0")
    assert len(aeg) == 3
    assert aeg.node(0).tool_type == ToolType.CODE_EXECUTION


@pytest.mark.unit
def test_crewai_adapter_consumes_trace() -> None:
    adapter = CrewAIAdapter(agent_type="crew_test")
    trace = [
        {"step_index": 0, "tool": "shell", "prompt_tokens": 1500, "output_tokens": 100, "observation_tokens": 300},
        {"step_index": 1, "tool": "browser", "prompt_tokens": 1600, "output_tokens": 120, "observation_tokens": 800},
    ]
    aeg = adapter.build_aeg(trace, graph_id="cr/0")
    assert len(aeg) == 2
    assert aeg.node(1).tool_type == ToolType.WEB_API


@pytest.mark.unit
def test_langchain_attach_raises_without_framework() -> None:
    adapter = LangChainAdapter()
    try:
        import langchain  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            adapter.attach()
    else:
        # LangChain is installed; attach should succeed.
        handler = adapter.attach()
        assert handler is not None
