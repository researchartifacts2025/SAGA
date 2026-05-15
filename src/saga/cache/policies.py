"""KV-cache eviction policies.

Four policies are implemented under a common interface so the simulator and
the competitive-ratio analysis can swap them transparently:

* ``LRUPolicy``       --- standard least-recently-used.
* ``PrefixLRUPolicy`` --- LRU with a prefix-aware bonus (vLLM v0.4.2+ style).
* ``WALRUPolicy``     --- SAGA's workflow-aware LRU using AEG predictions.
* ``BeladyOracle``    --- offline policy with full future knowledge, used to
                          compute competitive ratios.

All policies operate on a flat list of ``KVCacheEntry`` objects belonging to
one worker; cluster-wide coordination is the scheduler's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saga.core.types import KVCacheEntry


if TYPE_CHECKING:
    from saga.core.aeg import AgentExecutionGraph


# ---------------------------------------------------------------- base


@dataclass
class EvictionDecision:
    """A scored eviction candidate.

    ``score`` is sorted ascending; the lowest-score entry is evicted first.
    """

    session_id: str
    score: float


class EvictionPolicy(ABC):
    """Abstract base class for cache-eviction policies."""

    name: str = "base"

    @abstractmethod
    def score(
        self,
        entry: KVCacheEntry,
        now: float,
        ctx: PolicyContext,
    ) -> float:
        """Return an eviction-priority score; *lower* is more evictable."""

    def select_victim(
        self,
        entries: Iterable[KVCacheEntry],
        now: float,
        ctx: PolicyContext,
    ) -> KVCacheEntry | None:
        """Pick the entry with the lowest score; ``None`` if all are pinned."""
        best: KVCacheEntry | None = None
        best_score = float("inf")
        for entry in entries:
            if entry.pinned:
                continue
            s = self.score(entry, now, ctx)
            if s < best_score:
                best_score = s
                best = entry
        return best


@dataclass
class PolicyContext:
    """Read-only context handed to every policy on each scoring call.

    Bundles the global state a policy might need without forcing each policy
    to pull the same set of arguments. AEG-aware policies read ``aeg_for``
    and ``current_node``; oracle policies read ``future_accesses``.
    """

    aeg_for: dict[str, AgentExecutionGraph] = field(default_factory=dict)
    current_node: dict[str, int] = field(default_factory=dict)
    tau_max_ms: float = 1.0
    size_max_tokens: int = 1
    future_accesses: dict[str, list[float]] = field(default_factory=dict)
    weights: tuple[float, float, float] = (0.3, 0.5, 0.2)

    def with_max(self, entries: Iterable[KVCacheEntry], now: float) -> PolicyContext:
        """Update ``tau_max`` and ``size_max`` from the live entry set."""
        tau = 1.0
        size = 1
        for e in entries:
            tau = max(tau, now - e.last_access_time)
            size = max(size, e.n_tokens)
        self.tau_max_ms = tau
        self.size_max_tokens = size
        return self


# ----------------------------------------------------------------- LRU


class LRUPolicy(EvictionPolicy):
    """Standard LRU.

    Sorts purely by last-access time. Captures the behavior of pre-v0.4 vLLM
    and most generic KV-cache implementations.
    """

    name = "lru"

    def score(self, entry: KVCacheEntry, now: float, ctx: PolicyContext) -> float:
        return entry.last_access_time


# --------------------------------------------------------- LRU + Prefix


class PrefixLRUPolicy(EvictionPolicy):
    """LRU with a shared-prefix bonus.

    Approximates Automatic Prefix Caching as it ships in vLLM v0.4.2+:
    entries that share a long prefix with a hot session get a recency bonus
    so they survive eviction longer. Captures the qualitative behavior used
    in the LRU+Prefix row of the competitive-ratio comparison.
    """

    name = "lru_prefix"

    def __init__(self, prefix_bonus_ms: float = 2_000.0) -> None:
        self.prefix_bonus_ms = prefix_bonus_ms

    def score(self, entry: KVCacheEntry, now: float, ctx: PolicyContext) -> float:
        bonus = 0.0
        aeg = ctx.aeg_for.get(entry.session_id)
        if aeg is not None:
            cur = ctx.current_node.get(entry.session_id, entry.aeg_node_index)
            reuse = aeg.predict_reuse(cur, entry.n_tokens)
            if reuse > 0.5:
                bonus = self.prefix_bonus_ms
        return entry.last_access_time + bonus


# ----------------------------------------------------------- WA-LRU


class WALRUPolicy(EvictionPolicy):
    """Workflow-Aware LRU.

    Eviction priority is computed as a weighted sum of three normalized
    factors: recency, predicted reuse, and size:

        P_evict(s) = alpha * R(s) + beta * (1 - P_reuse(s)) + gamma * S(s)

    Larger ``P_evict`` means the entry is *more* evictable; we negate so the
    common ``select_victim`` (which picks the *smallest* score) returns the
    correct victim. AEG predictions come from the per-session AEG held in
    ``ctx.aeg_for`` and current node in ``ctx.current_node``.
    """

    name = "walru"

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        use_native: bool = True,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_native = use_native

    def _reuse_for(self, sid: str, ctx: PolicyContext) -> float:
        aeg = ctx.aeg_for.get(sid)
        if aeg is None:
            return 0.0
        entry = ctx._entry_index.get(sid) if hasattr(ctx, "_entry_index") else None
        node = ctx.current_node.get(sid, entry.aeg_node_index if entry else 0)
        cached = entry.n_tokens if entry else 0
        return aeg.predict_reuse(node, cached)

    def score(self, entry: KVCacheEntry, now: float, ctx: PolicyContext) -> float:
        tau_max = max(1.0, ctx.tau_max_ms)
        size_max = max(1, ctx.size_max_tokens)

        recency_norm = min(1.0, (now - entry.last_access_time) / tau_max)
        size_norm = min(1.0, entry.n_tokens / float(size_max))

        reuse = 0.0
        aeg = ctx.aeg_for.get(entry.session_id)
        if aeg is not None:
            cur = ctx.current_node.get(entry.session_id, entry.aeg_node_index)
            reuse = aeg.predict_reuse(cur, entry.n_tokens)

        p_evict = self.alpha * recency_norm + self.beta * (1.0 - reuse) + self.gamma * size_norm
        return -p_evict

    def select_victim(
        self,
        entries: Iterable[KVCacheEntry],
        now: float,
        ctx: PolicyContext,
    ) -> KVCacheEntry | None:
        if not self.use_native:
            return super().select_victim(entries, now, ctx)

        from saga.native import is_native_available, walru_select_victim

        if not is_native_available():
            return super().select_victim(entries, now, ctx)

        ent_list = [e for e in entries if not e.pinned]
        if not ent_list:
            return None

        def reuse_lookup(sid: str) -> float:
            aeg = ctx.aeg_for.get(sid)
            if aeg is None:
                return 0.0
            for e in ent_list:
                if e.session_id == sid:
                    cur = ctx.current_node.get(sid, e.aeg_node_index)
                    return aeg.predict_reuse(cur, e.n_tokens)
            return 0.0

        idx = walru_select_victim(
            ent_list,
            now=now,
            tau_max=ctx.tau_max_ms,
            size_max=ctx.size_max_tokens,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            reuse_lookup=reuse_lookup,
        )
        if idx < 0 or idx >= len(ent_list):
            return None
        return ent_list[idx]


# ----------------------------------------------------- Belady Oracle


class BeladyOracle(EvictionPolicy):
    """Optimal offline policy.

    Evicts the entry whose *next* access is farthest in the future (or never).
    Requires per-session future-access timestamps; ``PolicyContext.future_accesses``
    holds a sorted list of times per session that the harness produces by
    replaying the trace ahead of the policy run.
    """

    name = "belady"

    def __init__(self, use_native: bool = True) -> None:
        self.use_native = use_native

    def score(self, entry: KVCacheEntry, now: float, ctx: PolicyContext) -> float:
        future = ctx.future_accesses.get(entry.session_id, [])
        for t in future:
            if t > now:
                return -t
        return -float("inf")

    def select_victim(
        self,
        entries: Iterable[KVCacheEntry],
        now: float,
        ctx: PolicyContext,
    ) -> KVCacheEntry | None:
        if not self.use_native:
            return super().select_victim(entries, now, ctx)
        from saga.native import belady_select_victim, is_native_available

        if not is_native_available():
            return super().select_victim(entries, now, ctx)
        ent_list = [e for e in entries if not e.pinned]
        if not ent_list:
            return None
        future_lists = [ctx.future_accesses.get(e.session_id, []) for e in ent_list]
        idx = belady_select_victim(ent_list, now=now, future_accesses=future_lists)
        if idx < 0 or idx >= len(ent_list):
            return None
        return ent_list[idx]


# ---------------------------------------------------------- factory


def build_policy(name: str, **kwargs: float) -> EvictionPolicy:
    """Construct a policy by name; useful for config-driven instantiation."""
    name = name.lower()
    if name in ("lru",):
        return LRUPolicy()
    if name in ("lru_prefix", "prefix_lru", "apc"):
        return PrefixLRUPolicy(**kwargs)
    if name in ("walru", "wa_lru", "wa-lru", "saga"):
        return WALRUPolicy(**kwargs)
    if name in ("belady", "opt", "oracle"):
        return BeladyOracle()
    raise ValueError(f"unknown eviction policy: {name!r}")
