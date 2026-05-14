"""BFS / DFS / Hybrid execution strategies.

Three strategies for choosing which session to dequeue next at a worker:

* ``BFS`` (Breadth-First): admit every session's *first* step before any
  session's second step. Maximizes batch fullness and throughput at the
  cost of high cache eviction (sessions interleave on the cache).
* ``DFS`` (Depth-First): finish one session entirely before starting the
  next. Minimizes eviction but serializes work.
* ``Hybrid`` (SAGA): pick the highest-AFS session whose cache is already on
  this worker; fall back to least-recently-used cached session. Balances
  TCT and throughput.

The strategy plugs into the local worker queue policy and is selected via
:class:`saga.scheduler.coordinator.CoordinatorConfig.queue_strategy`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable


class QueueStrategy(ABC):
    """Pick the next session to run from a worker's pending queue."""

    name: str = "base"

    @abstractmethod
    def pick(
        self,
        queue: list[str],
        scoring_fn: Callable[[str], float] | None = None,
    ) -> str | None:
        """Return the chosen session id, or ``None`` if the queue is empty."""


class BFSStrategy(QueueStrategy):
    """Strict FIFO: oldest pending request first.

    All sessions advance one step before any session takes its second step.
    """

    name = "bfs"

    def pick(
        self,
        queue: list[str],
        scoring_fn=None,
    ) -> str | None:
        if not queue:
            return None
        return queue.pop(0)


class DFSStrategy(QueueStrategy):
    """Strict LIFO: newest pending request first, meaning whichever session
    just returned from a tool call gets priority over fresh arrivals.

    Minimizes context switches and cache eviction, at the cost of
    head-of-line blocking for new sessions.
    """

    name = "dfs"

    def pick(
        self,
        queue: list[str],
        scoring_fn=None,
    ) -> str | None:
        if not queue:
            return None
        return queue.pop(-1)


class HybridStrategy(QueueStrategy):
    """SAGA's hybrid: pick by ``scoring_fn`` (typically AFS urgency).

    Without a scoring function this falls back to BFS.
    """

    name = "hybrid"

    def pick(
        self,
        queue: list[str],
        scoring_fn=None,
    ) -> str | None:
        if not queue:
            return None
        if scoring_fn is None:
            return queue.pop(0)
        best_idx = 0
        best_score = scoring_fn(queue[0])
        for i, sid in enumerate(queue[1:], start=1):
            s = scoring_fn(sid)
            if s > best_score:
                best_score = s
                best_idx = i
        return queue.pop(best_idx)


def build_strategy(name: str) -> QueueStrategy:
    name = name.lower()
    if name == "bfs":
        return BFSStrategy()
    if name == "dfs":
        return DFSStrategy()
    if name in ("hybrid", "saga"):
        return HybridStrategy()
    raise ValueError(f"unknown queue strategy: {name!r}")


def queue_strategies(scope: Iterable[str] | None = None) -> list[str]:
    available = ["bfs", "dfs", "hybrid"]
    if scope is None:
        return available
    return [s for s in available if s in set(scope)]
