"""Framework-hint parser.

Real LangChain / AutoGen / CrewAI deployments expose orchestration metadata
through callbacks or message logs. The shapes are framework-specific:

* LangChain emits ``on_tool_start`` / ``on_tool_end`` events.
* AutoGen logs agent-to-agent messages as JSON lines.
* CrewAI emits a per-step trace.

We define a uniform :class:`FrameworkHint` shape and a parser that turns a
list of hints into a fully-populated :class:`AgentExecutionGraph`. This is
the **tier-A** observability path; for tier-B, see ``pattern.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from saga.core.aeg import AEGEdge, AEGNode, AgentExecutionGraph
from saga.core.types import ToolType


_TOOL_NAME_MAP: dict[str, ToolType] = {
    "shell": ToolType.CODE_EXECUTION,
    "python": ToolType.CODE_EXECUTION,
    "run_test": ToolType.CODE_EXECUTION,
    "exec": ToolType.CODE_EXECUTION,
    "read_file": ToolType.FILE_OPERATION,
    "write_file": ToolType.FILE_OPERATION,
    "ls": ToolType.FILE_OPERATION,
    "fetch_url": ToolType.WEB_API,
    "browser": ToolType.WEB_API,
    "http_get": ToolType.WEB_API,
    "sql": ToolType.DATABASE_QUERY,
    "query": ToolType.DATABASE_QUERY,
}


def tool_type_of(name: str) -> ToolType:
    """Best-effort mapping from a tool's display name to its type."""
    key = name.lower().strip()
    if key in _TOOL_NAME_MAP:
        return _TOOL_NAME_MAP[key]
    for fragment, tt in _TOOL_NAME_MAP.items():
        if fragment in key:
            return tt
    return ToolType.NONE


@dataclass
class FrameworkHint:
    """One step of agent execution as exposed by a framework callback.

    ``observation_tokens`` is taken from the framework's reported message
    body; if unavailable the parser falls back to an empirical default.
    """

    step_index: int
    tool_name: str
    prompt_tokens: int
    output_tokens: int
    observation_tokens: int = 0
    next_step_hint: int | None = None


@dataclass
class FrameworkHintParser:
    """Convert a sequence of :class:`FrameworkHint` into an AEG.

    Three frameworks are supported via :meth:`from_langchain_callbacks`,
    :meth:`from_autogen_messages`, and :meth:`from_crewai_trace`. Each
    accepts the framework-native shape and normalizes to ``FrameworkHint``.
    """

    default_observation_tokens: int = 300
    terminal_step_marker: str = "finish"

    # ------------------------------------------------------- normalization

    def from_langchain_callbacks(
        self, events: Iterable[dict[str, object]]
    ) -> list[FrameworkHint]:
        """Parse a list of ``{event, tool, prompt_tokens, ...}`` callback dicts."""
        hints: list[FrameworkHint] = []
        idx = 0
        cur: dict[str, object] = {}
        for ev in events:
            kind = str(ev.get("event", ""))
            if kind == "on_chain_start":
                cur = dict(ev)
                cur["__idx"] = idx
                idx += 1
            elif kind == "on_tool_end":
                tool = str(cur.get("tool", ev.get("tool", "")))
                hints.append(
                    FrameworkHint(
                        step_index=int(cur.get("__idx", idx - 1)),
                        tool_name=tool,
                        prompt_tokens=int(cur.get("prompt_tokens", 0)),
                        output_tokens=int(cur.get("output_tokens", 0)),
                        observation_tokens=int(
                            ev.get("observation_tokens", self.default_observation_tokens)
                        ),
                    )
                )
        return hints

    def from_autogen_messages(
        self, messages: Iterable[dict[str, object]]
    ) -> list[FrameworkHint]:
        """Parse AutoGen-style ``{role, content, tool_calls, ...}`` messages."""
        hints: list[FrameworkHint] = []
        for i, msg in enumerate(messages):
            tool_calls = msg.get("tool_calls") or []
            tool_name = ""
            if isinstance(tool_calls, list) and tool_calls:
                first = tool_calls[0]
                if isinstance(first, dict):
                    tool_name = str(first.get("name", ""))
            content = str(msg.get("content", ""))
            hints.append(
                FrameworkHint(
                    step_index=i,
                    tool_name=tool_name,
                    prompt_tokens=int(msg.get("prompt_tokens", len(content) // 4)),
                    output_tokens=int(msg.get("output_tokens", 0)),
                    observation_tokens=int(
                        msg.get("observation_tokens", self.default_observation_tokens)
                    ),
                )
            )
        return hints

    def from_crewai_trace(
        self, trace: Iterable[dict[str, object]]
    ) -> list[FrameworkHint]:
        """Parse CrewAI per-step trace entries."""
        hints: list[FrameworkHint] = []
        for i, step in enumerate(trace):
            hints.append(
                FrameworkHint(
                    step_index=int(step.get("step_index", i)),
                    tool_name=str(step.get("tool", "")),
                    prompt_tokens=int(step.get("prompt_tokens", 0)),
                    output_tokens=int(step.get("output_tokens", 0)),
                    observation_tokens=int(
                        step.get("observation_tokens", self.default_observation_tokens)
                    ),
                )
            )
        return hints

    # ----------------------------------------------------- build AEG

    def to_aeg(
        self,
        hints: Sequence[FrameworkHint],
        graph_id: str,
        workload_kind: str = "framework_hint",
        termination_prob: float = 0.05,
    ) -> AgentExecutionGraph:
        """Materialize an AEG from a sequence of hints.

        Hints are taken in order; transitions get probability ``1 -
        termination_prob`` except the final edge which terminates.
        """
        if not hints:
            return AgentExecutionGraph(
                graph_id=graph_id,
                nodes=[],
                edges=[],
                workload_kind=workload_kind,
                termination_prob=termination_prob,
            )

        nodes = [
            AEGNode(
                index=i,
                tool_type=tool_type_of(h.tool_name),
                prompt_tokens_est=max(1, h.prompt_tokens),
                output_tokens_est=max(1, h.output_tokens),
                observation_tokens_est=max(
                    1, h.observation_tokens or self.default_observation_tokens
                ),
                is_terminal=(i == len(hints) - 1)
                or h.tool_name.lower() == self.terminal_step_marker,
            )
            for i, h in enumerate(hints)
        ]
        edges: list[AEGEdge] = []
        for i in range(len(hints) - 1):
            edges.append(AEGEdge(src=i, dst=i + 1, probability=1.0 - termination_prob))

        return AgentExecutionGraph(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            workload_kind=workload_kind,
            termination_prob=termination_prob,
        )
