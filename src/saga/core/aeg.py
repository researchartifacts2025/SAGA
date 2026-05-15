"""Agent Execution Graph (AEG).

An AEG is the structural description of an agent workflow:
``G = (V, E, P, phi)`` with nodes V (LLM inference steps), directed edges E
(execution dependencies), a transition-probability function P, and a
tool-type assignment phi. The cache and scheduler modules consume an AEG to
predict future reuse and to compute remaining work.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from saga.core.types import ToolType


@dataclass(frozen=True)
class AEGEdge:
    """A directed edge between two AEG nodes with a transition probability."""

    src: int
    dst: int
    probability: float


@dataclass
class AEGNode:
    """An LLM inference step inside an AEG."""

    index: int
    tool_type: ToolType
    prompt_tokens_est: int
    output_tokens_est: int
    is_terminal: bool = False
    observation_tokens_est: int = 0


@dataclass
class AgentExecutionGraph:
    """An agent workflow as a directed graph of LLM steps.

    The dominant pattern (ReAct linear chains) has a single successor per node
    with a near-1 transition probability and a small termination probability;
    branching agents (tree-of-thought) produce multi-successor nodes whose
    probabilities sum to 1.
    """

    graph_id: str
    nodes: list[AEGNode]
    edges: list[AEGEdge]
    workload_kind: str = "generic"
    termination_prob: float = 0.05

    _succ_index: dict[int, list[AEGEdge]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._succ_index = {}
        for edge in self.edges:
            self._succ_index.setdefault(edge.src, []).append(edge)

    # ------------------------------------------------------------------ basic

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[AEGNode]:
        return iter(self.nodes)

    def node(self, idx: int) -> AEGNode:
        return self.nodes[idx]

    # ------------------------------------------------------------ successors

    def successors(self, idx: int) -> Sequence[AEGEdge]:
        return self._succ_index.get(idx, [])

    def successor_indices(self, idx: int) -> list[int]:
        return [e.dst for e in self.successors(idx)]

    def is_terminal(self, idx: int) -> bool:
        return self.nodes[idx].is_terminal or not self.successors(idx)

    def most_likely_successor(self, idx: int) -> int | None:
        succ = self.successors(idx)
        if not succ:
            return None
        return max(succ, key=lambda e: e.probability).dst

    # --------------------------------------------------------------- reuse

    def predict_reuse(self, current_node: int, cached_tokens: int) -> float:
        """Estimate the probability that the session's KV cache will be reused.

        For each outgoing edge ``current_node -> u`` we weight the prefix
        overlap by the transition probability. The overlap estimate for a
        ReAct chain is ``cached / (cached + obs_est)`` because the next step's
        prompt is the current context plus the new observation. The result
        is clamped to ``[0, 1]``.
        """
        succ = self.successors(current_node)
        if not succ or cached_tokens <= 0:
            return 0.0

        reuse = 0.0
        for edge in succ:
            obs_est = max(1, self.nodes[edge.dst].observation_tokens_est)
            overlap = cached_tokens / float(cached_tokens + obs_est)
            reuse += edge.probability * overlap

        return max(0.0, min(1.0, reuse))

    # ----------------------------------------------------- remaining work

    def remaining_work_tokens(
        self,
        current_node: int,
        max_depth: int = 64,
    ) -> tuple[int, int]:
        """Estimate the (prefill, decode) token totals from ``current_node``.

        Walks the most-likely path forward until a terminal node or
        ``max_depth`` is reached. The decode cost is summed directly; the
        prefill cost is summed only over the *new* tokens added each step
        because the cached prefix should not require re-prefilling.
        """
        prefill_tokens = 0
        decode_tokens = 0
        node_idx: int | None = current_node
        depth = 0

        while node_idx is not None and depth < max_depth:
            node = self.nodes[node_idx]
            new_prompt_tokens = max(1, node.observation_tokens_est)
            prefill_tokens += new_prompt_tokens
            decode_tokens += max(1, node.output_tokens_est)

            if self.is_terminal(node_idx):
                break

            node_idx = self.most_likely_successor(node_idx)
            depth += 1

        return prefill_tokens, decode_tokens

    # ------------------------------------------------------ trace helpers

    def linear_path(self) -> list[int]:
        """The most-likely linear path from node 0 to a terminal node."""
        path: list[int] = []
        idx: int | None = 0
        seen: set[int] = set()
        while idx is not None and idx not in seen:
            path.append(idx)
            seen.add(idx)
            if self.is_terminal(idx):
                break
            idx = self.most_likely_successor(idx)
        return path

    def expected_step_count(self) -> float:
        """The expected number of steps under the termination probability."""
        if self.termination_prob <= 0.0:
            return float(len(self.nodes))
        return min(float(len(self.nodes)), 1.0 / self.termination_prob)


# -------------------------------------------------------------- factories


def build_linear_aeg(
    graph_id: str,
    n_steps: int,
    tool_types: Iterable[ToolType],
    prompt_tokens_est: int,
    output_tokens_est: int,
    observation_tokens_est: int,
    termination_prob: float = 0.05,
    workload_kind: str = "generic",
) -> AgentExecutionGraph:
    """Build a linear AEG (the dominant ReAct pattern).

    ``tool_types`` provides the per-step tool category; if shorter than
    ``n_steps`` the last value is repeated.
    """
    tools = list(tool_types)
    if not tools:
        tools = [ToolType.NONE]

    nodes: list[AEGNode] = []
    for i in range(n_steps):
        tool = tools[i] if i < len(tools) else tools[-1]
        nodes.append(
            AEGNode(
                index=i,
                tool_type=tool,
                prompt_tokens_est=prompt_tokens_est,
                output_tokens_est=output_tokens_est,
                observation_tokens_est=observation_tokens_est,
                is_terminal=(i == n_steps - 1),
            )
        )

    edges: list[AEGEdge] = []
    for i in range(n_steps - 1):
        edges.append(AEGEdge(src=i, dst=i + 1, probability=1.0 - termination_prob))

    return AgentExecutionGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        workload_kind=workload_kind,
        termination_prob=termination_prob,
    )


def build_branching_aeg(
    graph_id: str,
    depth: int,
    branching: int,
    tool_type: ToolType,
    prompt_tokens_est: int,
    output_tokens_est: int,
    observation_tokens_est: int,
    workload_kind: str = "generic",
) -> AgentExecutionGraph:
    """Build a balanced tree AEG (tree-of-thought pattern)."""
    nodes: list[AEGNode] = []
    edges: list[AEGEdge] = []

    counter = 0

    def add_subtree(level: int, parent: int | None) -> None:
        nonlocal counter
        idx = counter
        counter += 1
        is_leaf = level == depth
        nodes.append(
            AEGNode(
                index=idx,
                tool_type=tool_type,
                prompt_tokens_est=prompt_tokens_est,
                output_tokens_est=output_tokens_est,
                observation_tokens_est=observation_tokens_est,
                is_terminal=is_leaf,
            )
        )
        if parent is not None:
            edges.append(AEGEdge(src=parent, dst=idx, probability=1.0 / branching))
        if not is_leaf:
            for _ in range(branching):
                add_subtree(level + 1, idx)

    add_subtree(0, None)

    return AgentExecutionGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        workload_kind=workload_kind,
        termination_prob=0.0,
    )
