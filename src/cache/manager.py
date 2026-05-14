"""Per-worker KV-cache manager.

Owns the live cache entries for one worker and exposes three operations:

* ``ensure_capacity`` --- free space by evicting under the current policy
  until ``required_tokens`` fit inside ``kv_capacity_tokens``.
* ``admit``           --- insert / update an entry, refreshing its TTL.
* ``touch``           --- mark an entry as recently used (no size change).

The manager is policy-agnostic; the policy is injected at construction so the
simulator can run LRU, LRU+Prefix, WA-LRU, and Belady against the same
plumbing for fair comparison.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saga.cache.policies import EvictionPolicy, PolicyContext
from saga.cache.ttl import ToolTTLPolicy
from saga.core.types import KVCacheEntry, ToolType


if TYPE_CHECKING:
    from saga.core.aeg import AgentExecutionGraph


@dataclass
class CacheDecision:
    """The result of an ``admit`` call.

    ``hit`` is true if the session was cached and reusable at admit time.
    ``evicted`` lists the entries removed to make space.
    ``regenerated_tokens`` is the number of prefix tokens that the worker had
    to re-prefill on this admit (zero on a hit, the full prefix on a miss).
    """

    hit: bool
    evicted: list[KVCacheEntry] = field(default_factory=list)
    regenerated_tokens: int = 0


class CacheManager:
    """KV-cache pool for a single worker."""

    def __init__(
        self,
        worker_id: int,
        capacity_tokens: int,
        policy: EvictionPolicy,
        ttl_policy: ToolTTLPolicy | None = None,
        pressure_low: float = 0.7,
        pressure_high: float = 0.9,
    ) -> None:
        self.worker_id = worker_id
        self.capacity_tokens = max(1, int(capacity_tokens))
        self.policy = policy
        self.ttl_policy = ttl_policy
        self.pressure_low = pressure_low
        self.pressure_high = pressure_high

        self._entries: dict[str, KVCacheEntry] = {}
        self._used_tokens = 0

        self._aeg_for: dict[str, AgentExecutionGraph] = {}
        self._current_node: dict[str, int] = {}
        self._future_accesses: dict[str, list[float]] = {}

        self.n_admits = 0
        self.n_hits = 0
        self.n_evictions = 0
        self.n_expired_evictions = 0
        self.regenerated_tokens_total = 0
        self.tokens_admitted_total = 0

    # ------------------------------------------------------ accounting

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @property
    def used_fraction(self) -> float:
        return self._used_tokens / float(self.capacity_tokens)

    def entries(self) -> Iterable[KVCacheEntry]:
        return self._entries.values()

    def contains(self, session_id: str) -> bool:
        entry = self._entries.get(session_id)
        return entry is not None

    def get(self, session_id: str) -> KVCacheEntry | None:
        return self._entries.get(session_id)

    # --------------------------------------------------- workflow info

    def register_aeg(self, session_id: str, aeg: AgentExecutionGraph, node: int = 0) -> None:
        self._aeg_for[session_id] = aeg
        self._current_node[session_id] = node

    def update_node(self, session_id: str, node: int) -> None:
        if session_id in self._current_node:
            self._current_node[session_id] = node

    def set_future_accesses(self, session_id: str, times: list[float]) -> None:
        """Inform an oracle policy of all future access times for a session."""
        self._future_accesses[session_id] = times

    def forget(self, session_id: str) -> None:
        """Drop all state for a session (terminated task)."""
        self._aeg_for.pop(session_id, None)
        self._current_node.pop(session_id, None)
        self._future_accesses.pop(session_id, None)
        entry = self._entries.pop(session_id, None)
        if entry is not None:
            self._used_tokens -= entry.n_tokens
            self._used_tokens = max(0, self._used_tokens)

    # --------------------------------------------------- TTL handling

    def set_ttl_for_tool_call(
        self,
        session_id: str,
        tool: ToolType,
        now: float,
    ) -> float | None:
        """Set the TTL on ``session_id``'s entry for an upcoming tool call.

        Returns the TTL deadline (ms) if set, else ``None``. If no TTL policy
        is configured the entry's TTL is left at infinity (LRU semantics).
        """
        if self.ttl_policy is None:
            return None
        entry = self._entries.get(session_id)
        if entry is None:
            return None
        ttl_ms = self.ttl_policy.compute_ttl_ms(tool, self.used_fraction)
        entry.ttl_deadline = now + ttl_ms
        return entry.ttl_deadline

    def clear_ttl(self, session_id: str) -> None:
        entry = self._entries.get(session_id)
        if entry is not None:
            entry.ttl_deadline = float("inf")

    # --------------------------------------------------- eviction loop

    def _build_context(self, now: float) -> PolicyContext:
        ctx = PolicyContext(
            aeg_for=self._aeg_for,
            current_node=self._current_node,
            future_accesses=self._future_accesses,
        )
        ctx.with_max(self._entries.values(), now)
        return ctx

    def expire(self, now: float) -> list[KVCacheEntry]:
        """Evict any entries whose TTL has elapsed *and* the pool is under pressure.

        TTL acts as a hint that an entry is no longer worth protecting; under no
        memory pressure we leave the entry alone (avoiding gratuitous misses).
        Under pressure we drop expired entries first to satisfy ``ensure_capacity``.
        """
        if self.used_fraction <= self.pressure_low:
            return []
        expired: list[KVCacheEntry] = []
        for sid, entry in list(self._entries.items()):
            if entry.is_expired(now):
                self._entries.pop(sid, None)
                self._used_tokens -= entry.n_tokens
                self._used_tokens = max(0, self._used_tokens)
                expired.append(entry)
        self.n_expired_evictions += len(expired)
        self.n_evictions += len(expired)
        return expired

    def ensure_capacity(self, required_tokens: int, now: float) -> list[KVCacheEntry]:
        """Evict until the pool has at least ``required_tokens`` of headroom."""
        if required_tokens > self.capacity_tokens:
            raise ValueError(
                f"required tokens {required_tokens} exceeds capacity {self.capacity_tokens}"
            )
        evicted: list[KVCacheEntry] = []
        if self._used_tokens + required_tokens <= self.capacity_tokens:
            return evicted

        ctx = self._build_context(now)

        while self._used_tokens + required_tokens > self.capacity_tokens:
            victim = self.policy.select_victim(self._entries.values(), now, ctx)
            if victim is None:
                break
            self._entries.pop(victim.session_id, None)
            self._used_tokens -= victim.n_tokens
            self._used_tokens = max(0, self._used_tokens)
            evicted.append(victim)
            self.n_evictions += 1

        return evicted

    # --------------------------------------------------------- admit

    def admit(
        self,
        session_id: str,
        new_token_count: int,
        now: float,
        aeg_node_index: int = 0,
    ) -> CacheDecision:
        """Bring a session's cache to ``new_token_count`` tokens.

        On a hit, the entry grows by the difference between the new and old
        token counts; on a miss the entry is created and the full prefix is
        counted as regenerated.
        """
        self.n_admits += 1
        existing = self._entries.get(session_id)
        decision = CacheDecision(hit=existing is not None)

        if existing is not None and new_token_count <= existing.n_tokens:
            existing.last_access_time = now
            existing.aeg_node_index = aeg_node_index
            existing.ttl_deadline = float("inf")
            self.n_hits += 1
            return decision

        if existing is None:
            need = new_token_count
            decision.regenerated_tokens = new_token_count
        else:
            need = new_token_count - existing.n_tokens
            decision.regenerated_tokens = 0

        if existing is None:
            evicted = self.ensure_capacity(need, now)
        else:
            self._used_tokens -= existing.n_tokens
            evicted = self.ensure_capacity(need, now)
            self._used_tokens += existing.n_tokens

        decision.evicted = evicted

        if existing is None:
            entry = KVCacheEntry(
                session_id=session_id,
                worker_id=self.worker_id,
                n_tokens=new_token_count,
                last_access_time=now,
                creation_time=now,
                aeg_node_index=aeg_node_index,
            )
            self._entries[session_id] = entry
            self._used_tokens += new_token_count
        else:
            growth = new_token_count - existing.n_tokens
            existing.n_tokens = new_token_count
            existing.last_access_time = now
            existing.aeg_node_index = aeg_node_index
            existing.ttl_deadline = float("inf")
            self._used_tokens += growth
            self.n_hits += 1

        self.tokens_admitted_total += new_token_count
        self.regenerated_tokens_total += decision.regenerated_tokens
        return decision

    # ---------------------------------------------------------- touch

    def touch(self, session_id: str, now: float) -> None:
        entry = self._entries.get(session_id)
        if entry is not None:
            entry.last_access_time = now

    # ----------------------------------------------------------- pin

    def pin(self, session_id: str, value: bool = True) -> None:
        entry = self._entries.get(session_id)
        if entry is not None:
            entry.pinned = value

    # --------------------------------------------------------- stats

    def stats(self) -> dict[str, float]:
        hit_rate = self.n_hits / max(1, self.n_admits)
        return {
            "n_admits": self.n_admits,
            "n_hits": self.n_hits,
            "n_evictions": self.n_evictions,
            "n_expired_evictions": self.n_expired_evictions,
            "hit_rate": hit_rate,
            "regenerated_tokens": self.regenerated_tokens_total,
            "tokens_admitted": self.tokens_admitted_total,
            "used_fraction": self.used_fraction,
        }
