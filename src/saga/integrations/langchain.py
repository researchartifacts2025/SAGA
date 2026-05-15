"""LangChain bridge.

LangChain exposes a ``BaseCallbackHandler`` interface with hooks for every
chain / tool / LLM event. We provide :class:`LangChainAdapter` that:

1. Captures ``on_tool_start`` / ``on_tool_end`` to time tool calls.
2. Captures ``on_llm_start`` / ``on_llm_end`` to time LLM inferences and to
   count prompt / output tokens.
3. Periodically flushes the captured events into an
   :class:`~saga.core.aeg.AgentExecutionGraph` via the
   :class:`~saga.workflow.analyzer.FrameworkHintParser`.

The adapter is itself dependency-free: it only imports LangChain inside
:meth:`attach`, which is the only method that needs the framework class
hierarchy. Tests can drive ``record_*`` directly with synthetic events.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from saga.core.aeg import AgentExecutionGraph
from saga.workflow.analyzer import FrameworkHint, FrameworkHintParser
from saga.workflow.pattern import PatternInferenceEngine


@dataclass
class _RunContext:
    """One in-flight chain invocation."""

    run_id: str
    started_at: float
    tool_name: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    observation_tokens: int = 0


@dataclass
class LangChainAdapter:
    """Bridge LangChain callback events into SAGA's workflow analyzer.

    Args:
        agent_type: a stable agent identifier; used to bucket pattern state.
        parser: optional parser override (default ``FrameworkHintParser()``).
        pattern_engine: optional pattern-inference engine to feed observed
            tool sequences into. When provided, every flushed AEG also
            updates the engine, which can later infer AEGs for tasks that
            arrive without explicit hints.
    """

    agent_type: str = "langchain_default"
    parser: FrameworkHintParser = field(default_factory=FrameworkHintParser)
    pattern_engine: PatternInferenceEngine | None = None

    _active: dict[str, _RunContext] = field(default_factory=dict)
    _hints: list[FrameworkHint] = field(default_factory=list)
    _step_counter: int = 0

    # ---------------------------------------------------- record events

    def record_chain_start(
        self,
        run_id: str | None = None,
        prompt_tokens: int = 0,
    ) -> str:
        rid = run_id or str(uuid.uuid4())
        self._active[rid] = _RunContext(
            run_id=rid,
            started_at=time.monotonic(),
            prompt_tokens=int(prompt_tokens),
        )
        return rid

    def record_llm_end(self, run_id: str, output_tokens: int) -> None:
        ctx = self._active.get(run_id)
        if ctx is None:
            return
        ctx.output_tokens = int(output_tokens)

    def record_tool_end(
        self,
        run_id: str,
        tool_name: str,
        observation_tokens: int = 0,
    ) -> None:
        ctx = self._active.pop(run_id, None)
        if ctx is None:
            return
        ctx.tool_name = tool_name
        ctx.observation_tokens = int(observation_tokens)

        self._hints.append(
            FrameworkHint(
                step_index=self._step_counter,
                tool_name=tool_name,
                prompt_tokens=ctx.prompt_tokens,
                output_tokens=ctx.output_tokens,
                observation_tokens=ctx.observation_tokens,
            )
        )
        self._step_counter += 1

    # -------------------------------------------------------- emit AEG

    def emit_aeg(self, graph_id: str | None = None) -> AgentExecutionGraph:
        """Materialize the captured hints and reset the buffer.

        The resulting AEG is also fed to the pattern engine when one is
        attached, so subsequent unhinted tasks of the same ``agent_type``
        will draw on these observations.
        """
        if graph_id is None:
            graph_id = f"{self.agent_type}/{uuid.uuid4().hex[:8]}"
        aeg = self.parser.to_aeg(self._hints, graph_id=graph_id, workload_kind=self.agent_type)

        if self.pattern_engine is not None and self._hints:
            tools = [h.tool_name for h in self._hints]
            self.pattern_engine.observe_session(self.agent_type, tools)

        self._hints.clear()
        self._step_counter = 0
        return aeg

    # --------------------------------------------------------- attach

    def attach(self) -> Any:
        """Return a LangChain-compatible callback handler.

        Raises:
            ImportError: when LangChain is not installed.
        """
        try:
            from langchain.callbacks.base import (
                BaseCallbackHandler,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise ImportError(
                "LangChain is not installed; use the record_* methods directly for unit tests."
            ) from exc

        bridge = self

        class _Handler(BaseCallbackHandler):  # type: ignore[misc, valid-type]
            def on_llm_start(self, serialized, prompts, run_id=None, **kw):
                tokens = sum(len(p) // 4 for p in prompts)
                bridge.record_chain_start(str(run_id), prompt_tokens=tokens)

            def on_llm_end(self, response, run_id=None, **kw):
                tokens = sum(
                    len(getattr(g, "text", "")) // 4
                    for gens in getattr(response, "generations", [])
                    for g in gens
                )
                bridge.record_llm_end(str(run_id), output_tokens=tokens)

            def on_tool_end(self, output, run_id=None, name="", **kw):
                bridge.record_tool_end(
                    str(run_id),
                    tool_name=name or "tool",
                    observation_tokens=len(str(output)) // 4,
                )

        return _Handler()
