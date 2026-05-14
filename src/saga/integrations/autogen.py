"""AutoGen bridge.

AutoGen logs agent-to-agent messages as JSON-shaped dicts with ``role``,
``content``, and optional ``tool_calls`` / ``tool_call_id`` fields. The
adapter scans the log to recover a hint sequence, then materializes an AEG.

Tested without AutoGen installed; the framework dependency is optional.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from saga.core.aeg import AgentExecutionGraph
from saga.workflow.analyzer import FrameworkHintParser
from saga.workflow.pattern import PatternInferenceEngine


@dataclass
class AutoGenAdapter:
    """Convert an AutoGen message log into an AEG and optionally feed pattern state."""

    agent_type: str = "autogen_default"
    parser: FrameworkHintParser = field(default_factory=FrameworkHintParser)
    pattern_engine: PatternInferenceEngine | None = None

    def build_aeg(
        self,
        messages: Iterable[dict[str, object]],
        graph_id: str = "autogen",
    ) -> AgentExecutionGraph:
        hints = self.parser.from_autogen_messages(list(messages))
        aeg = self.parser.to_aeg(hints, graph_id=graph_id, workload_kind=self.agent_type)
        if self.pattern_engine is not None and hints:
            self.pattern_engine.observe_session(
                self.agent_type,
                [h.tool_name for h in hints],
            )
        return aeg

    def attach(self):  # type: ignore[no-untyped-def]
        """Return a callable that AutoGen's ``register_reply`` accepts.

        The callable runs *after* a reply is generated and records the
        tool-call information. It returns ``False`` so AutoGen continues
        its normal reply pipeline.
        """
        bridge = self

        def _hook(sender, recipient, message, **kwargs):
            bridge.build_aeg([message])
            return False, None

        return _hook
