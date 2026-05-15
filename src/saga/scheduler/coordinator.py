"""Global coordinator.

Holds cluster-wide state: worker list, per-session affinity (via the router),
queues per worker, fairness counters (via AFS), and a periodic ``tick`` that
runs every 100ms (the scheduling epoch). The coordinator is intentionally a
*plain object*: the simulator drives time forward and calls ``tick`` and
``submit`` directly. In a real deployment the same methods would sit behind
gRPC.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from saga.core.types import Session, SessionState, Task, Worker
from saga.fairness.afs import AFSScheduler
from saga.scheduler.routing import RoutingDecision, SessionRouter
from saga.scheduler.stealing import StealOutcome, WorkStealer
from saga.scheduler.strategies import QueueStrategy, build_strategy
from saga.utils.seeds import RNG


@dataclass
class CoordinatorConfig:
    """Tunable knobs for the global coordinator."""

    epoch_ms: float = 100.0
    routing_strategy: str = "session_affinity"
    queue_strategy: str = "hybrid"
    load_threshold: float = 0.8
    enable_work_stealing: bool = True
    enable_afs: bool = True
    afs_preempt_threshold_ms: float = 500.0
    t_idle_ms: float = 100.0
    r_max: float = 2.0
    migration_mean_ms: float = 230.0
    migration_p95_ms: float = 890.0


@dataclass
class _WorkerView:
    """Lightweight read view of a worker for routing decisions."""

    worker: Worker
    queue: list[str] = field(default_factory=list)


class GlobalCoordinator:
    """Cluster-wide scheduling state, separate from any single worker."""

    def __init__(
        self,
        workers: Sequence[Worker],
        cfg: CoordinatorConfig | None = None,
        rng: RNG | None = None,
    ) -> None:
        self.cfg = cfg or CoordinatorConfig()
        self.rng = rng or RNG(seed=42)

        self._workers: dict[int, _WorkerView] = {w.worker_id: _WorkerView(w) for w in workers}

        self.router = SessionRouter(
            strategy=self.cfg.routing_strategy,
            load_threshold=self.cfg.load_threshold,
        )
        self.stealer = WorkStealer(
            t_idle_ms=self.cfg.t_idle_ms,
            r_max=self.cfg.r_max,
            migration_mean_ms=self.cfg.migration_mean_ms,
            migration_p95_ms=self.cfg.migration_p95_ms,
        )
        self.afs = AFSScheduler() if self.cfg.enable_afs else None
        self.queue_strategy: QueueStrategy = build_strategy(self.cfg.queue_strategy)

        self._sessions: dict[str, Session] = {}
        self._cached_predicates: dict[int, set[str]] = {wid: set() for wid in self._workers}
        self.last_epoch_ms: float = 0.0

    # ---------------------------------------------------- registration

    def register_session(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        if self.afs is not None:
            self.afs.note_submit(session.tenant_id, session.task)

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def all_workers(self) -> list[Worker]:
        return [v.worker for v in self._workers.values()]

    def queue_for(self, worker_id: int) -> list[str]:
        return self._workers[worker_id].queue

    # ----------------------------------------------------- cache view

    def mark_cached(self, worker_id: int, session_id: str) -> None:
        self._cached_predicates.setdefault(worker_id, set()).add(session_id)

    def mark_uncached(self, worker_id: int, session_id: str) -> None:
        self._cached_predicates.get(worker_id, set()).discard(session_id)

    def _cached(self, worker_id: int, session_id: str) -> bool:
        return session_id in self._cached_predicates.get(worker_id, set())

    # ------------------------------------------------------ routing

    def route(self, session: Session) -> RoutingDecision:
        prefix_hash = _prefix_hash(session.task)
        decision = self.router.route(
            session_id=session.session_id,
            prefix_hash=prefix_hash,
            workers=self.all_workers(),
            cached_predicate=self._cached,
        )
        view = self._workers[decision.worker_id]
        view.queue.append(session.session_id)
        view.worker.queue_depth = len(view.queue)
        session.worker_id = decision.worker_id
        session.state = SessionState.PENDING
        return decision

    # ------------------------------------------------------- queues

    def dequeue(self, worker_id: int) -> str | None:
        view = self._workers[worker_id]
        if not view.queue:
            return None

        if self.afs is not None and self.cfg.queue_strategy == "hybrid":
            sid = self.queue_strategy.pick(
                view.queue,
                scoring_fn=lambda s: self.afs_priority(s, now=self.last_epoch_ms),  # type: ignore[arg-type]
            )
        else:
            sid = self.queue_strategy.pick(view.queue)
        if sid is None:
            return None
        view.worker.queue_depth = len(view.queue)
        view.worker.current_session_id = sid
        return sid

    def requeue(self, worker_id: int, session_id: str, front: bool = False) -> None:
        view = self._workers[worker_id]
        if front:
            view.queue.insert(0, session_id)
        else:
            view.queue.append(session_id)
        view.worker.queue_depth = len(view.queue)

    def remove_from_queue(self, worker_id: int, session_id: str) -> bool:
        view = self._workers[worker_id]
        try:
            view.queue.remove(session_id)
        except ValueError:
            return False
        view.worker.queue_depth = len(view.queue)
        return True

    # ----------------------------------------------------------- AFS

    def afs_priority(self, session_id: str, now: float) -> float:
        if self.afs is None:
            return 1.0
        sess = self._sessions.get(session_id)
        if sess is None:
            return 1.0
        return self.afs.priority(sess.tenant_id, now)

    # ------------------------------------------------------- tick

    def tick(self, now: float) -> list[StealOutcome]:
        """Run a coordinator epoch: refresh AFS and (optionally) work-steal."""
        self.last_epoch_ms = now

        if self.afs is not None:
            self.afs.refresh(now)

        if not self.cfg.enable_work_stealing:
            return []

        queues = {wid: list(view.queue) for wid, view in self._workers.items()}
        actions = self.stealer.step(now, self.all_workers(), queues, self.rng)

        applied: list[StealOutcome] = []
        for action in actions:
            if not action.success or action.session_id is None:
                applied.append(action)
                continue
            if self.remove_from_queue(action.victim_id, action.session_id):
                self.requeue(action.thief_id, action.session_id, front=True)
                self.router.remember_session(action.session_id, action.thief_id)
                sess = self._sessions.get(action.session_id)
                if sess is not None:
                    sess.worker_id = action.thief_id
                    sess.task.n_migrations += 1
                self._workers[action.thief_id].worker.n_steals_in += 1
                self._workers[action.victim_id].worker.n_steals_out += 1
                self._workers[action.thief_id].worker.n_migrations_received += 1
                applied.append(action)
            else:
                applied.append(
                    StealOutcome(
                        success=False,
                        thief_id=action.thief_id,
                        victim_id=action.victim_id,
                        session_id=action.session_id,
                        migration_ms=0.0,
                        reason="stale",
                    )
                )
        return applied

    # ---------------------------------------------------- accounting

    def note_completion(self, session: Session, now: float) -> None:
        if self.afs is not None:
            self.afs.note_complete(session.tenant_id, session.task, now)
        self.router.forget_session(session.session_id)
        for cached_set in self._cached_predicates.values():
            cached_set.discard(session.session_id)

    def stats(self) -> dict[str, float]:
        d = {
            "n_workers": len(self._workers),
            "n_sessions": len(self._sessions),
            "last_epoch_ms": self.last_epoch_ms,
        }
        d.update(self.stealer.stats())
        if self.afs is not None:
            d.update(self.afs.stats())
        return d


# ---------------------------------------------------------- helpers


def _prefix_hash(task: Task) -> int:
    """A stable 32-bit hash of the task's prefix-defining fields.

    Two tasks from the same workload kind and tenant share a prefix bucket;
    this approximates the way real systems group requests by system prompt.
    """
    h = hashlib.blake2b(digest_size=4)
    h.update(task.workload_kind.encode("utf-8"))
    h.update(b"|")
    h.update(task.tenant_id.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")
