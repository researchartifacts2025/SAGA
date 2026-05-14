"""CrewAI bridge.

CrewAI's verbose run mode emits per-step trace entries (``tool``,
``prompt_tokens``, ``output_tokens``, ``observation_tokens``). The adapter
consumes these traces and materializes an AEG; like the LangChain and
AutoGen adapters it is dependency-free at import time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from saga.core.aeg import AgentExecutionGraph
from saga.workflow.analyzer import FrameworkHintParser
from saga.workflow.pattern import PatternInferenceEngine


@dataclass
class CrewAIAdapter:
    """Convert a CrewAI per-step trace into an AEG."""

    agent_type: str = "crewai_default"
    parser: FrameworkHintParser = field(default_factory=FrameworkHintParser)
    pattern_engine: PatternInferenceEngine | None = None

    def build_aeg(
        self,
        trace: Iterable[dict[str, object]],
        graph_id: str = "crewai",
    ) -> AgentExecutionGraph:
        hints = self.parser.from_crewai_trace(list(trace))
        aeg = self.parser.to_aeg(hints, graph_id=graph_id, workload_kind=self.agent_type)
        if self.pattern_engine is not None and hints:
            self.pattern_engine.observe_session(
                self.agent_type,
                [h.tool_name for h in hints],
            )
        return aeg
