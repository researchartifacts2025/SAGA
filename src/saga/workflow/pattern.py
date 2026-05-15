"""Pattern-based AEG inference.

When framework hints are not available, SAGA observes the *request stream*
and infers an AEG. The algorithm:

1.  Group requests by session identifier.
2.  Within each session, label each step with its observed tool type.
3.  Maintain a transition-count matrix C[a, b] = number of times tool ``a``
    is followed by tool ``b`` within any session.
4.  Convert to a transition probability matrix P[a, b] = C[a, b] / sum_b C[a, b].
5.  Keep edges where ``P[a, b] >= theta_conf`` (default 0.7).
6.  Emit a per-tool AEG whose edges reflect the kept probabilities; in
    practice a single linear AEG with the modal successor at each node is
    sufficient for ReAct-style traces.

For a fresh agent type with fewer than ``cold_start_tasks = 30`` observed
sessions, pattern inference is suspended and the system serves the workload
as a request-level workload. The paper reports 87 % accuracy and a 12-18 %
TCT degradation versus explicit hints in the implicit-trace regime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from saga.core.aeg import AEGEdge, AEGNode, AgentExecutionGraph
from saga.core.types import ToolType
from saga.workflow.analyzer import tool_type_of


@dataclass
class PatternState:
    """Running per-agent-type pattern state.

    The state is intentionally small and lock-free-friendly: a transition
    count matrix and a session-step counter.
    """

    agent_type: str
    counts: dict[tuple[ToolType, ToolType], int] = field(default_factory=dict)
    tool_total: dict[ToolType, int] = field(default_factory=dict)
    sessions_observed: int = 0
    steps_observed: int = 0

    def observe_transition(self, src: ToolType, dst: ToolType) -> None:
        key = (src, dst)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.tool_total[src] = self.tool_total.get(src, 0) + 1
        self.steps_observed += 1

    def observe_session_complete(self) -> None:
        self.sessions_observed += 1

    def transition_prob(self, src: ToolType, dst: ToolType) -> float:
        total = self.tool_total.get(src, 0)
        if total == 0:
            return 0.0
        return self.counts.get((src, dst), 0) / total

    def best_successor(self, src: ToolType) -> tuple[ToolType, float] | None:
        candidates = [(dst, self.counts.get((src, dst), 0)) for dst in ToolType]
        candidates = [(dst, c) for dst, c in candidates if c > 0]
        if not candidates:
            return None
        dst, count = max(candidates, key=lambda x: x[1])
        total = self.tool_total.get(src, 0)
        if total == 0:
            return None
        return dst, count / total


@dataclass
class PatternInferenceEngine:
    """Maintain per-agent-type pattern state and emit inferred AEGs.

    Args:
        theta_conf: confidence threshold below which we treat a transition
            as noise (default 0.7, per paper).
        cold_start_tasks: number of sessions to observe before pattern
            inference activates (default 30).
        default_prompt_tokens: prompt-token estimate when only tool types
            are known (default 3000, midpoint of SWE-bench profile).
        default_output_tokens: output-token estimate (default 250).
        default_observation_tokens: observation-token estimate (default 300).
    """

    theta_conf: float = 0.7
    cold_start_tasks: int = 30
    default_prompt_tokens: int = 3_000
    default_output_tokens: int = 250
    default_observation_tokens: int = 300

    _state: dict[str, PatternState] = field(default_factory=dict)

    # ------------------------------------------------------------- observe

    def observe_session(
        self,
        agent_type: str,
        tool_sequence: Iterable[ToolType | str],
    ) -> None:
        """Update transition statistics from one completed session."""
        state = self._state.setdefault(agent_type, PatternState(agent_type=agent_type))
        tools = [t if isinstance(t, ToolType) else tool_type_of(str(t)) for t in tool_sequence]
        for src, dst in zip(tools, tools[1:], strict=False):
            state.observe_transition(src, dst)
        state.observe_session_complete()

    def has_warmed_up(self, agent_type: str) -> bool:
        state = self._state.get(agent_type)
        return state is not None and state.sessions_observed >= self.cold_start_tasks

    def accuracy_estimate(self, agent_type: str) -> float:
        """Per-paper formula: fraction of transitions with ``P >= theta_conf``."""
        state = self._state.get(agent_type)
        if state is None or state.steps_observed == 0:
            return 0.0
        kept = 0
        for (src, dst), count in state.counts.items():
            total = state.tool_total.get(src, 1)
            if count / total >= self.theta_conf:
                kept += count
        return kept / state.steps_observed

    def state_for(self, agent_type: str) -> PatternState | None:
        return self._state.get(agent_type)

    # --------------------------------------------------------------- infer

    def infer_aeg(
        self,
        agent_type: str,
        graph_id: str,
        n_steps: int,
    ) -> AgentExecutionGraph | None:
        """Emit an inferred AEG; ``None`` if cold-start has not finished."""
        if not self.has_warmed_up(agent_type):
            return None

        state = self._state[agent_type]
        # Choose the modal starting tool (or NONE).
        if state.tool_total:
            start_tool = max(state.tool_total.items(), key=lambda x: x[1])[0]
        else:
            start_tool = ToolType.NONE

        nodes: list[AEGNode] = []
        edges: list[AEGEdge] = []
        cur = start_tool
        for i in range(n_steps):
            is_terminal = i == n_steps - 1
            nodes.append(
                AEGNode(
                    index=i,
                    tool_type=cur,
                    prompt_tokens_est=self.default_prompt_tokens,
                    output_tokens_est=self.default_output_tokens,
                    observation_tokens_est=self.default_observation_tokens,
                    is_terminal=is_terminal,
                )
            )
            if is_terminal:
                break

            best = state.best_successor(cur)
            if best is None or best[1] < self.theta_conf:
                # No high-confidence successor: stay on current tool.
                nxt, prob = cur, 1.0 - 1.0 / max(2.0, n_steps - i)
            else:
                nxt, prob = best
            edges.append(AEGEdge(src=i, dst=i + 1, probability=prob))
            cur = nxt

        return AgentExecutionGraph(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            workload_kind=f"inferred_{agent_type}",
            termination_prob=1.0 / max(1.0, n_steps),
        )
