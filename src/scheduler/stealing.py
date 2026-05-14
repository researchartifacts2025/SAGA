"""Randomized work stealing.

Triggered when (a) a worker's queue has been empty for ``T_idle = 100ms`` or
(b) the load ratio between the most- and least-loaded workers exceeds
``R_max = 2.0x``. The stealer picks a victim uniformly at random from
overloaded workers, asks for that worker's oldest pending session, and the
victim's KV cache is migrated to the thief.

The Blumofe--Leiserson bound (work_steal_bound) assumes zero migration cost;
we charge a non-zero migration latency drawn from a configurable distribution
so the engine accounts for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from saga.core.types import Worker
from saga.utils.seeds import RNG


@dataclass
class StealOutcome:
    """The result of a single steal attempt."""

    success: bool
    thief_id: int
    victim_id: int
    session_id: str | None
    migration_ms: float
    reason: str


class WorkStealer:
    """Cluster-wide work-stealing coordinator.

    The stealer is consulted once per coordinator epoch. It does not migrate
    work itself; it returns a list of ``StealOutcome`` actions that the engine
    applies (so the engine remains the single owner of cache state).
    """

    def __init__(
        self,
        t_idle_ms: float = 100.0,
        r_max: float = 2.0,
        migration_mean_ms: float = 230.0,
        migration_p95_ms: float = 890.0,
        max_steals_per_epoch: int = 4,
    ) -> None:
        self.t_idle_ms = t_idle_ms
        self.r_max = r_max
        self.migration_mean_ms = migration_mean_ms
        self.migration_p95_ms = migration_p95_ms
        self.max_steals_per_epoch = max_steals_per_epoch

        self._idle_since: dict[int, float] = {}
        self.attempts = 0
        self.successes = 0

    # ----------------------------------------------------- migration

    def sample_migration_ms(self, rng: RNG) -> float:
        """Sample a migration latency.

        We use a log-normal parameterized so that its median equals
        ``migration_mean_ms`` and its 95th percentile equals
        ``migration_p95_ms``.
        """
        import math

        if self.migration_p95_ms <= self.migration_mean_ms:
            return self.migration_mean_ms
        mu = math.log(self.migration_mean_ms)
        sigma = math.log(self.migration_p95_ms / self.migration_mean_ms) / 1.6449
        return rng.lognormal(mu, max(sigma, 1e-6))

    # ----------------------------------------------------------- decide

    def step(
        self,
        now: float,
        workers: Sequence[Worker],
        queues: dict[int, list[str]],
        rng: RNG,
    ) -> list[StealOutcome]:
        """Choose work-stealing actions for the current epoch."""
        if not workers:
            return []

        for w in workers:
            if w.queue_depth == 0:
                self._idle_since.setdefault(w.worker_id, now)
            else:
                self._idle_since.pop(w.worker_id, None)

        loads = [_load_score(w) for w in workers]
        load_max = max(loads)
        load_min = max(min(loads), 1e-6)
        ratio = load_max / load_min

        actions: list[StealOutcome] = []

        if ratio < self.r_max and not any(
            (now - self._idle_since.get(w.worker_id, now)) > self.t_idle_ms
            for w in workers if w.queue_depth == 0
        ):
            return actions

        overloaded = sorted(
            (w for w in workers if _load_score(w) > load_min * self.r_max
             or w.queue_depth >= 2),
            key=_load_score,
            reverse=True,
        )
        idle = [
            w for w in workers
            if w.queue_depth == 0
            and (now - self._idle_since.get(w.worker_id, now)) > self.t_idle_ms
        ]
        if not overloaded and ratio >= self.r_max:
            overloaded = [max(workers, key=_load_score)]
        if not idle and ratio >= self.r_max:
            idle = [min(workers, key=_load_score)]

        rng_local = rng.fork("stealer", now)

        n_actions = 0
        for thief in idle:
            if not overloaded:
                break
            if n_actions >= self.max_steals_per_epoch:
                break
            self.attempts += 1
            victim_idx = rng_local.randint(0, len(overloaded))
            victim = overloaded[victim_idx]
            queue = queues.get(victim.worker_id, [])
            if not queue:
                actions.append(
                    StealOutcome(
                        success=False,
                        thief_id=thief.worker_id,
                        victim_id=victim.worker_id,
                        session_id=None,
                        migration_ms=0.0,
                        reason="victim_empty",
                    )
                )
                overloaded.pop(victim_idx)
                continue
            session_id = queue[0]
            migration_ms = self.sample_migration_ms(rng_local)
            self.successes += 1
            n_actions += 1
            actions.append(
                StealOutcome(
                    success=True,
                    thief_id=thief.worker_id,
                    victim_id=victim.worker_id,
                    session_id=session_id,
                    migration_ms=migration_ms,
                    reason="ratio" if ratio >= self.r_max else "idle",
                )
            )
            self._idle_since.pop(thief.worker_id, None)
            if victim.queue_depth <= 1:
                overloaded.pop(victim_idx)

        return actions

    # ------------------------------------------------------ telemetry

    def stats(self) -> dict[str, float]:
        return {
            "steal_attempts": self.attempts,
            "steal_successes": self.successes,
            "steal_success_rate": self.successes / max(1, self.attempts),
        }


def _load_score(w: Worker) -> float:
    return 0.5 * w.utilization + 0.5 * w.memory_pressure + 0.05 * w.queue_depth
