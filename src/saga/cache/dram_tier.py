"""CPU-DRAM offload tier.

A complementary architecture to HBM-only retention: evicted KV-cache entries
are moved to host DRAM rather than discarded, so re-prefill is needed only
when both tiers miss. PCIe Gen4 ×16 sustains ≈25 GB/s in practice, giving a
≈430 ms one-way transfer for the 10.7 GB Llama-3-70B 32K-context cache.

This module models the second tier:

* :class:`DRAMPool` holds offloaded entries with a configured byte capacity.
* :class:`SwapTimeModel` charges per-byte PCIe latency for swap-in / swap-out.
* :class:`TieredCacheManager` wraps a base :class:`CacheManager` and routes
  evictions through the DRAM pool first.

The HBM-only path remains the default; the tiered path activates when
``ClusterConfig.dram_tier_enabled = True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from saga.cache.manager import CacheDecision, CacheManager
from saga.core.types import KVCacheEntry


# PCIe Gen4 x16 sustained: ~25 GB/s (paper §5.4).
_PCIE_BANDWIDTH_BYTES_PER_MS = 25_000_000.0
# Bytes per KV-cache token at FP16, Llama-3-70B GQA n_kv=8, L=80, d_h=128:
# 2 (head) * 8 (kv) * 80 (L) * 128 (d_h) * 2 bytes = 327,680 B/token.
# Reduce by tensor-parallel slicing (TP=4) -> ~81,920 B/token per worker.
_BYTES_PER_TOKEN_PER_WORKER = 81_920.0


@dataclass
class SwapTimeModel:
    """PCIe transfer cost as a function of bytes."""

    bandwidth_bytes_per_ms: float = _PCIE_BANDWIDTH_BYTES_PER_MS
    contention_factor: float = 0.5
    """Sustained PCIe under multi-tenant contention is roughly half of peak
    (paper §5.4 (2))."""

    bytes_per_token: float = _BYTES_PER_TOKEN_PER_WORKER

    def transfer_ms(self, n_tokens: int, contended: bool = False) -> float:
        if n_tokens <= 0:
            return 0.0
        bytes_total = float(n_tokens) * self.bytes_per_token
        bw = self.bandwidth_bytes_per_ms
        if contended:
            bw *= self.contention_factor
        return bytes_total / max(1.0, bw)


@dataclass
class DRAMPool:
    """A bounded host-DRAM pool of offloaded entries.

    Eviction within the DRAM pool itself is plain LRU.
    """

    capacity_tokens: int
    _entries: dict[str, KVCacheEntry] = field(default_factory=dict)
    _used_tokens: int = 0

    n_swaps_in: int = 0
    n_swaps_out: int = 0
    n_dram_evictions: int = 0

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @property
    def used_fraction(self) -> float:
        if self.capacity_tokens <= 0:
            return 0.0
        return self._used_tokens / float(self.capacity_tokens)

    def contains(self, session_id: str) -> bool:
        return session_id in self._entries

    def get(self, session_id: str) -> KVCacheEntry | None:
        return self._entries.get(session_id)

    def admit(self, entry: KVCacheEntry, now: float) -> list[KVCacheEntry]:
        """Bring ``entry`` into DRAM, evicting LRU entries to make room.

        Returns the entries fully discarded (DRAM also full).
        """
        evicted: list[KVCacheEntry] = []
        while self._used_tokens + entry.n_tokens > self.capacity_tokens and self._entries:
            oldest_sid = min(self._entries, key=lambda s: self._entries[s].last_access_time)
            old = self._entries.pop(oldest_sid)
            self._used_tokens -= old.n_tokens
            self._used_tokens = max(0, self._used_tokens)
            self.n_dram_evictions += 1
            evicted.append(old)

        existing = self._entries.pop(entry.session_id, None)
        if existing is not None:
            self._used_tokens -= existing.n_tokens

        entry.last_access_time = now
        self._entries[entry.session_id] = entry
        self._used_tokens += entry.n_tokens
        self.n_swaps_out += 1
        return evicted

    def evict(self, session_id: str) -> KVCacheEntry | None:
        e = self._entries.pop(session_id, None)
        if e is not None:
            self._used_tokens -= e.n_tokens
            self._used_tokens = max(0, self._used_tokens)
            self.n_swaps_in += 1
        return e

    def stats(self) -> dict[str, float]:
        return {
            "dram_used_tokens": self._used_tokens,
            "dram_used_fraction": self.used_fraction,
            "dram_n_swaps_in": self.n_swaps_in,
            "dram_n_swaps_out": self.n_swaps_out,
            "dram_n_evictions": self.n_dram_evictions,
            "dram_n_resident": len(self._entries),
        }


class TieredCacheManager:
    """Two-tier cache wrapper: HBM (base) + DRAM (this).

    The two-tier hit semantics:

    * **HBM hit**: cheap path, no swap charge.
    * **DRAM hit**: swap-in cost (PCIe transfer), then HBM admit.
    * **Miss**: full re-prefill on the next inference step.

    The wrapper preserves the :class:`CacheManager` interface for
    ``ensure_capacity`` and ``admit`` while routing evictions through DRAM.
    """

    def __init__(
        self,
        base: CacheManager,
        dram_capacity_tokens: int,
        swap_model: SwapTimeModel | None = None,
        contended: bool = False,
    ) -> None:
        self.base = base
        self.dram = DRAMPool(capacity_tokens=dram_capacity_tokens)
        self.swap_model = swap_model or SwapTimeModel()
        self.contended = contended

        self.cumulative_swap_ms = 0.0

    # ------------------------------------------------- delegated state

    @property
    def used_tokens(self) -> int:
        return self.base.used_tokens

    @property
    def used_fraction(self) -> float:
        return self.base.used_fraction

    def contains(self, session_id: str) -> bool:
        return self.base.contains(session_id) or self.dram.contains(session_id)

    def hbm_contains(self, session_id: str) -> bool:
        return self.base.contains(session_id)

    def dram_contains(self, session_id: str) -> bool:
        return self.dram.contains(session_id)

    # ----------------------------------------------------------- admit

    def admit(
        self,
        session_id: str,
        new_token_count: int,
        now: float,
        aeg_node_index: int = 0,
    ) -> CacheDecision:
        """Admit into HBM. DRAM-resident entries are restored via swap-in."""
        dram_hit = self.dram.contains(session_id)
        if dram_hit:
            dram_entry = self.dram.evict(session_id)
            if dram_entry is not None:
                swap_ms = self.swap_model.transfer_ms(dram_entry.n_tokens, contended=self.contended)
                self.cumulative_swap_ms += swap_ms
                dram_entry.last_access_time = now
                self.base._entries[session_id] = dram_entry
                self.base._used_tokens += dram_entry.n_tokens

        decision = self.base.admit(
            session_id=session_id,
            new_token_count=new_token_count,
            now=now,
            aeg_node_index=aeg_node_index,
        )

        if decision.evicted:
            for victim in decision.evicted:
                swap_ms = self.swap_model.transfer_ms(victim.n_tokens, contended=self.contended)
                self.cumulative_swap_ms += swap_ms
                self.dram.admit(victim, now)

        if dram_hit:
            decision.hit = True
            decision.regenerated_tokens = 0
        return decision

    # ----------------------------------------------------- pass-through

    def touch(self, session_id: str, now: float) -> None:
        self.base.touch(session_id, now)
        if self.dram.contains(session_id):
            e = self.dram.get(session_id)
            if e is not None:
                e.last_access_time = now

    def forget(self, session_id: str) -> None:
        self.base.forget(session_id)
        self.dram.evict(session_id)

    def expire(self, now: float) -> list[KVCacheEntry]:
        return self.base.expire(now)

    # ---------------------------------------------------------- stats

    def stats(self) -> dict[str, float]:
        s = dict(self.base.stats())
        s.update(self.dram.stats())
        s["cumulative_swap_ms"] = self.cumulative_swap_ms
        return s
