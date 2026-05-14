"""Session-affinity routing.

A new request from session ``s`` should ideally land on the worker that still
holds ``s``'s KV cache. Pure affinity, however, lets popular sessions create
hotspots. SAGA balances the two with:

    route(r) = w*_s  if load(w*_s) < theta and cached(w*_s, s)
             = argmin_w load(w)  otherwise

where ``w*_s`` is the worker that currently caches ``s``. ``theta = 0.8``
reserves 20% headroom for spikes.

We support three strategies for ablations:

* ``session_affinity`` --- the SAGA default.
* ``prefix_affinity``  --- bucket by prefix hash (vLLM PrefixCacheAffinityRouter).
* ``least_loaded``     --- ignore session identity entirely.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from saga.core.types import Worker


@dataclass
class RoutingDecision:
    """The chosen worker plus a short reason string for instrumentation."""

    worker_id: int
    reason: str
    cache_hit_expected: bool


class SessionRouter:
    """Implements the cache-aware routing rule with three strategies."""

    def __init__(
        self,
        strategy: str = "session_affinity",
        load_threshold: float = 0.8,
        prefix_buckets: int = 64,
    ) -> None:
        valid = {"session_affinity", "prefix_affinity", "least_loaded"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid!r}; got {strategy!r}")
        self.strategy = strategy
        self.load_threshold = load_threshold
        self.prefix_buckets = max(1, prefix_buckets)

        self._session_to_worker: dict[str, int] = {}
        self._prefix_to_worker: dict[int, int] = {}

    # --------------------------------------------------- affinity map

    def remember_session(self, session_id: str, worker_id: int) -> None:
        self._session_to_worker[session_id] = worker_id

    def forget_session(self, session_id: str) -> None:
        self._session_to_worker.pop(session_id, None)

    def known_worker(self, session_id: str) -> int | None:
        return self._session_to_worker.get(session_id)

    # ------------------------------------------------------ routing

    def route(
        self,
        session_id: str,
        prefix_hash: int,
        workers: Sequence[Worker],
        cached_predicate: CachedPredicate,
    ) -> RoutingDecision:
        if not workers:
            raise ValueError("at least one worker required")

        if self.strategy == "least_loaded":
            best = _least_loaded(workers)
            self._session_to_worker[session_id] = best.worker_id
            return RoutingDecision(
                worker_id=best.worker_id,
                reason="least_loaded",
                cache_hit_expected=False,
            )

        if self.strategy == "prefix_affinity":
            bucket = prefix_hash % self.prefix_buckets
            preferred = self._prefix_to_worker.get(bucket)
            if preferred is not None:
                preferred_worker = _find(workers, preferred)
                if (
                    preferred_worker is not None
                    and _load_of(preferred_worker) < self.load_threshold
                ):
                    self._session_to_worker[session_id] = preferred
                    return RoutingDecision(
                        worker_id=preferred,
                        reason="prefix_affinity",
                        cache_hit_expected=False,
                    )
            best = _least_loaded(workers)
            self._prefix_to_worker[bucket] = best.worker_id
            self._session_to_worker[session_id] = best.worker_id
            return RoutingDecision(
                worker_id=best.worker_id,
                reason="prefix_affinity_fallback",
                cache_hit_expected=False,
            )

        # session_affinity
        preferred = self._session_to_worker.get(session_id)
        if preferred is not None:
            preferred_worker = _find(workers, preferred)
            if preferred_worker is not None:
                load = _load_of(preferred_worker)
                cached = cached_predicate(preferred, session_id)
                if cached and load < self.load_threshold:
                    return RoutingDecision(
                        worker_id=preferred,
                        reason="session_affinity_hit",
                        cache_hit_expected=True,
                    )
        best = _least_loaded(workers)
        self._session_to_worker[session_id] = best.worker_id
        return RoutingDecision(
            worker_id=best.worker_id,
            reason=("session_affinity_overloaded" if preferred is not None else "session_affinity_new"),
            cache_hit_expected=False,
        )


# ----------------------------------------------------------- helpers


CachedPredicate = "callable"


def _load_of(worker: Worker) -> float:
    util = worker.utilization
    mem = worker.memory_pressure
    return 0.5 * util + 0.5 * mem + 0.01 * worker.queue_depth


def _find(workers: Iterable[Worker], worker_id: int) -> Worker | None:
    for w in workers:
        if w.worker_id == worker_id:
            return w
    return None


def _least_loaded(workers: Sequence[Worker]) -> Worker:
    return min(workers, key=_load_of)
