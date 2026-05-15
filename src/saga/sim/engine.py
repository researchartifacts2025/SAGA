"""Discrete-event simulator engine.

Time is in milliseconds. The engine pulls events from an ``EventQueue``,
dispatches them by ``EventKind``, and updates state through the cluster,
coordinator, and cache manager. The model below is deliberately coarse: the
goal is faithful *relative* behavior across policies on agent traces, not
microsecond-accurate kernel timing.

Inference cost model (per step):

    prefill_ms = max(new_prompt_tokens, 1) / prefill_tokens_per_ms
    decode_ms  = max(output_tokens, 1)    / decode_tokens_per_ms

On a cold miss, the *whole* current context must be prefilled; on a hit only
the delta (the new observation) must be prefilled. The cache manager reports
which case applies via ``CacheDecision``.

Tool durations are taken verbatim from the ground-truth ``ToolPlan`` produced
by the workload generator.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from saga.core.aeg import AgentExecutionGraph
from saga.core.types import Session, SessionState, Task
from saga.scheduler.coordinator import CoordinatorConfig, GlobalCoordinator
from saga.sim.cluster import Cluster, ClusterConfig, build_cluster
from saga.sim.events import Event, EventKind, EventQueue
from saga.utils.logging import get_logger
from saga.utils.seeds import RNG
from saga.workload.base import AgentTaskTemplate, ToolPlan


log = get_logger("saga.sim")


# --------------------------------------------------------- result types


@dataclass
class WorkerSnapshot:
    """A snapshot of one worker at end of simulation."""

    worker_id: int
    cumulative_busy_ms: float
    n_steals_in: int
    n_steals_out: int
    memory_pressure: float
    cache_hit_rate: float
    regenerated_tokens: int


@dataclass
class SimulationResult:
    """The full result of one simulator run."""

    sim_time_ms: float
    config_label: str
    seed: int

    tasks: list[Task]
    workers: list[WorkerSnapshot]

    n_cache_admits: int = 0
    n_cache_hits: int = 0
    n_evictions: int = 0
    n_expired: int = 0
    n_steals: int = 0
    n_migrations: int = 0
    regenerated_tokens: int = 0
    tokens_admitted: int = 0

    coord_stats: dict[str, float] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        return self.n_cache_hits / max(1, self.n_cache_admits)

    @property
    def regeneration_ratio(self) -> float:
        return self.regenerated_tokens / max(1, self.tokens_admitted)

    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.is_complete]

    def tct_seconds(self) -> list[float]:
        return [t.tct_ms / 1_000.0 for t in self.completed_tasks()]

    def throughput_per_min(self) -> float:
        completed = sum(1 for t in self.tasks if t.is_complete)
        if self.sim_time_ms <= 0:
            return 0.0
        return completed * 60_000.0 / self.sim_time_ms

    def mean_memory_utilization(self) -> float:
        if not self.workers:
            return 0.0
        return sum(w.memory_pressure for w in self.workers) / len(self.workers)


# ------------------------------------------------------- engine class


@dataclass
class EngineConfig:
    """Engine knobs that are not part of cluster/coordinator config."""

    seed: int = 42
    horizon_ms: float = 600_000.0
    enable_speculative_prefetch: bool = True
    label: str = "saga"


class SimulatorEngine:
    """Top-level driver tying together cluster, coordinator, and workloads."""

    def __init__(
        self,
        cluster_cfg: ClusterConfig,
        coord_cfg: CoordinatorConfig,
        engine_cfg: EngineConfig,
    ) -> None:
        self.cluster_cfg = cluster_cfg
        self.coord_cfg = coord_cfg
        self.engine_cfg = engine_cfg
        self.rng = RNG(engine_cfg.seed)
        self.cluster: Cluster = build_cluster(cluster_cfg)
        self.coordinator = GlobalCoordinator(
            workers=self.cluster.workers,
            cfg=coord_cfg,
            rng=self.rng.fork("coord"),
        )
        self.queue = EventQueue()

        self._sessions: dict[str, Session] = {}
        self._aegs: dict[str, AgentExecutionGraph] = {}
        self._tool_plans: dict[str, ToolPlan] = {}
        self._tasks: list[Task] = []

        self._now: float = 0.0

    # ----------------------------------------------- admission

    def admit(self, templates: Iterable[AgentTaskTemplate]) -> None:
        for tmpl in templates:
            self._aegs[tmpl.task.aeg_id] = tmpl.aeg
            self._tool_plans[tmpl.task.task_id] = tmpl.tool_plan
            session = Session(
                session_id=tmpl.task.task_id,
                task=tmpl.task,
            )
            self._sessions[session.session_id] = session
            self._tasks.append(tmpl.task)
            if self.coordinator.afs is not None:
                self.coordinator.afs.ensure_tenant(tmpl.task.tenant_id, weight=tmpl.tenant_weight)
            self.queue.push(
                tmpl.task.submit_time,
                EventKind.ARRIVAL,
                session_id=session.session_id,
                payload={"task_id": tmpl.task.task_id},
            )

    def schedule_epoch_ticks(self) -> None:
        epoch = self.coord_cfg.epoch_ms
        t = epoch
        while t <= self.engine_cfg.horizon_ms:
            self.queue.push(t, EventKind.EPOCH_TICK)
            t += epoch

    # ----------------------------------------------------- run

    def run(self) -> SimulationResult:
        self.schedule_epoch_ticks()
        while self.queue:
            ev = self.queue.pop()
            if ev.time > self.engine_cfg.horizon_ms:
                break
            self._now = ev.time
            self._dispatch(ev)

        # finalize incomplete tasks (mark as failed for accounting)
        for task in self._tasks:
            if not task.is_complete:
                task.completion_time = -1.0

        snapshots = [
            WorkerSnapshot(
                worker_id=w.worker_id,
                cumulative_busy_ms=w.cumulative_busy_ms,
                n_steals_in=w.n_steals_in,
                n_steals_out=w.n_steals_out,
                memory_pressure=self.cluster.cache_for(w.worker_id).used_fraction,
                cache_hit_rate=self.cluster.cache_for(w.worker_id).stats()["hit_rate"],
                regenerated_tokens=int(
                    self.cluster.cache_for(w.worker_id).stats()["regenerated_tokens"]
                ),
            )
            for w in self.cluster.workers
        ]

        result = SimulationResult(
            sim_time_ms=self._now,
            config_label=self.engine_cfg.label,
            seed=self.engine_cfg.seed,
            tasks=list(self._tasks),
            workers=snapshots,
            coord_stats=self.coordinator.stats(),
        )

        for mgr in self.cluster.cache_managers:
            s = mgr.stats()
            result.n_cache_admits += int(s["n_admits"])
            result.n_cache_hits += int(s["n_hits"])
            result.n_evictions += int(s["n_evictions"])
            result.n_expired += int(s["n_expired_evictions"])
            result.regenerated_tokens += int(s["regenerated_tokens"])
            result.tokens_admitted += int(s["tokens_admitted"])

        result.n_steals = int(self.coordinator.stealer.successes)
        result.n_migrations = sum(w.n_migrations_received for w in self.cluster.workers)

        return result

    # ----------------------------------------------- dispatch

    def _dispatch(self, ev: Event) -> None:
        if ev.kind == EventKind.ARRIVAL:
            self._on_arrival(ev)
        elif ev.kind == EventKind.INFERENCE_START:
            self._on_inference_start(ev)
        elif ev.kind == EventKind.INFERENCE_END:
            self._on_inference_end(ev)
        elif ev.kind == EventKind.TOOL_END:
            self._on_tool_end(ev)
        elif ev.kind == EventKind.EPOCH_TICK:
            self._on_epoch_tick(ev)
        elif ev.kind == EventKind.TASK_COMPLETE:
            self._on_task_complete(ev)
        elif ev.kind == EventKind.MIGRATION_END:
            self._on_migration_end(ev)
        elif ev.kind == EventKind.CACHE_EXPIRE:
            self._on_cache_expire(ev)

    # -------- arrival

    def _on_arrival(self, ev: Event) -> None:
        sid = ev.session_id or ""
        session = self._sessions[sid]
        session.task.start_time = self._now
        session.state = SessionState.PENDING
        self.coordinator.register_session(session)
        decision = self.coordinator.route(session)
        log.debug("arrival sid=%s -> w%d (%s)", sid, decision.worker_id, decision.reason)
        self._try_dispatch(decision.worker_id)

    # -------- dispatch from worker queue to inference

    def _try_dispatch(self, worker_id: int) -> None:
        worker = self.cluster.worker_by_id(worker_id)
        if worker.busy_until > self._now:
            return
        sid = self.coordinator.dequeue(worker_id)
        if sid is None:
            return
        self._start_inference(sid, worker_id)

    def _start_inference(self, sid: str, worker_id: int) -> None:
        session = self._sessions[sid]
        aeg = self._aegs[session.task.aeg_id]
        node = aeg.node(session.aeg_node_index)
        worker = self.cluster.worker_by_id(worker_id)
        mgr = self.cluster.cache_for(worker_id)
        session.worker_id = worker_id
        session.state = SessionState.RUNNING

        mgr.register_aeg(sid, aeg, session.aeg_node_index)

        if session.context_tokens == 0:
            new_context = max(1, node.prompt_tokens_est)
        else:
            new_context = session.context_tokens + node.observation_tokens_est

        decision = mgr.admit(
            session_id=sid,
            new_token_count=new_context,
            now=self._now,
            aeg_node_index=session.aeg_node_index,
        )
        session.context_tokens = new_context
        session.cached_tokens = new_context

        # adjust eviction accounting on the task
        for ev_entry in decision.evicted:
            session.task.n_cache_evictions += 1
            self.coordinator.mark_uncached(worker_id, ev_entry.session_id)

        # prefill cost
        if decision.hit:
            prefill_tokens = max(1, node.observation_tokens_est)
            miss_stall_ms = 0.0
        else:
            prefill_tokens = new_context
            session.task.n_regenerated_tokens += decision.regenerated_tokens
            miss_stall_ms = self.cluster_cfg.cache_miss_stall_ms

        prefill_ms = prefill_tokens / max(1.0, worker.prefill_tokens_per_ms)
        decode_ms = max(1, node.output_tokens_est) / max(1.0, worker.decode_tokens_per_ms)
        duration = prefill_ms + decode_ms + miss_stall_ms

        worker.busy_until = self._now + duration
        worker.cumulative_busy_ms += duration
        worker.current_session_id = sid
        worker.sessions_seen.add(sid)
        session.task.n_inference_calls += 1

        if self.coordinator.afs is not None:
            self.coordinator.afs.note_progress(session.tenant_id, session.task.task_id, duration)

        self.queue.push(
            self._now + duration,
            EventKind.INFERENCE_END,
            session_id=sid,
            worker_id=worker_id,
        )

        self.coordinator.mark_cached(worker_id, sid)
        self.coordinator.router.remember_session(sid, worker_id)

    def _on_inference_start(self, ev: Event) -> None:
        # Reserved for future use (e.g., scheduling-induced delays).
        return

    # -------- inference end

    def _on_inference_end(self, ev: Event) -> None:
        sid = ev.session_id or ""
        worker_id = ev.worker_id or 0
        session = self._sessions[sid]
        aeg = self._aegs[session.task.aeg_id]
        worker = self.cluster.worker_by_id(worker_id)
        worker.current_session_id = None
        node = aeg.node(session.aeg_node_index)

        session.n_steps_completed += 1
        session.last_step_completion = self._now

        # tool call?
        plan = self._tool_plans[session.task.task_id]
        tool_ms = (
            plan.durations_ms[session.aeg_node_index]
            if session.aeg_node_index < len(plan.durations_ms)
            else 0.0
        )

        if (
            aeg.is_terminal(session.aeg_node_index)
            or session.n_steps_completed >= session.task.n_steps
        ):
            self.queue.push(self._now, EventKind.TASK_COMPLETE, session_id=sid, worker_id=worker_id)
            self._try_dispatch(worker_id)
            return

        if tool_ms > 0.0:
            session.state = SessionState.WAITING_TOOL
            session.tool_release_time = self._now + tool_ms
            session.task.n_tool_calls += 1

            mgr = self.cluster.cache_for(worker_id)
            mgr.set_ttl_for_tool_call(sid, node.tool_type, self._now)

            self.cluster.estimator.update(node.tool_type, tool_ms)

            self.queue.push(
                self._now + tool_ms,
                EventKind.TOOL_END,
                session_id=sid,
                worker_id=worker_id,
            )
        else:
            # No tool call -- advance the AEG node and re-route.
            session.aeg_node_index += 1
            session.state = SessionState.PENDING
            decision = self.coordinator.route(session)
            self._try_dispatch(decision.worker_id)
            if decision.worker_id != worker_id:
                self._try_dispatch(worker_id)
            return

        # Speculative prefetch: pin and pre-extend the cache for the most-likely
        # successor so that, if it arrives before the tool returns, the prefill
        # cost is amortized into the otherwise-idle gap rather than paid on
        # admission. We pin the entry during the tool gap so eviction cannot
        # rob the prefetched prefix; the pin is cleared on tool_end.
        if self.engine_cfg.enable_speculative_prefetch:
            nxt = aeg.most_likely_successor(session.aeg_node_index)
            if nxt is not None:
                mgr = self.cluster.cache_for(worker_id)
                successor_obs = (
                    plan.observation_tokens[session.aeg_node_index]
                    if session.aeg_node_index < len(plan.observation_tokens)
                    else aeg.node(nxt).observation_tokens_est
                )
                target_tokens = session.context_tokens + max(1, successor_obs)
                mgr.admit(
                    session_id=sid,
                    new_token_count=target_tokens,
                    now=self._now,
                    aeg_node_index=nxt,
                )
                mgr.pin(sid, True)
                self.coordinator.mark_cached(worker_id, sid)

        self._try_dispatch(worker_id)

    # -------- tool end

    def _on_tool_end(self, ev: Event) -> None:
        sid = ev.session_id or ""
        session = self._sessions[sid]
        prior_worker = session.worker_id

        # advance to the next AEG node
        session.aeg_node_index += 1
        session.state = SessionState.PENDING

        # context grew by the tool observation
        plan = self._tool_plans[session.task.task_id]
        idx = session.aeg_node_index - 1
        if 0 <= idx < len(plan.observation_tokens):
            session.context_tokens += plan.observation_tokens[idx]

        # clear the TTL on the prior worker; the next step re-enters routing,
        # so a non-affinity strategy may send the session elsewhere (and miss).
        prior_mgr = self.cluster.cache_for(prior_worker)
        prior_mgr.clear_ttl(sid)
        prior_mgr.pin(sid, False)

        decision = self.coordinator.route(session)
        self._try_dispatch(decision.worker_id)
        if decision.worker_id != prior_worker:
            self._try_dispatch(prior_worker)

    # -------- task complete

    def _on_task_complete(self, ev: Event) -> None:
        sid = ev.session_id or ""
        session = self._sessions[sid]
        session.state = SessionState.FINISHED
        session.task.completion_time = self._now
        session.task.succeeded = True
        self.coordinator.note_completion(session, self._now)
        mgr = self.cluster.cache_for(session.worker_id)
        mgr.forget(sid)

    # -------- migration

    def _on_migration_end(self, ev: Event) -> None:
        # Migration is modeled as a one-shot cache transfer; the cache map on
        # the destination is already updated by the coordinator; we just
        # release any "in-flight" markers here.
        return

    # -------- cache expire

    def _on_cache_expire(self, ev: Event) -> None:
        worker_id = ev.worker_id or 0
        mgr = self.cluster.cache_for(worker_id)
        mgr.expire(self._now)

    # -------- epoch tick

    def _on_epoch_tick(self, ev: Event) -> None:
        actions = self.coordinator.tick(self._now)
        for action in actions:
            if not action.success or action.session_id is None:
                continue
            # Charge migration latency to the thief's queue.
            thief = self.cluster.worker_by_id(action.thief_id)
            thief.busy_until = max(thief.busy_until, self._now + action.migration_ms)
            thief.cumulative_busy_ms += action.migration_ms
            self.queue.push(
                self._now + action.migration_ms,
                EventKind.MIGRATION_END,
                session_id=action.session_id,
                worker_id=action.thief_id,
            )

        # Periodic TTL sweep across all workers.
        for mgr in self.cluster.cache_managers:
            mgr.expire(self._now)

        # Wake up idle workers whose queues just got something.
        for w in self.cluster.workers:
            if w.busy_until <= self._now and w.queue_depth > 0:
                self._try_dispatch(w.worker_id)


# ------------------------------------------------------- helpers


def stream_templates(
    arrivals: Iterable[tuple[float, AgentTaskTemplate]],
) -> Iterator[AgentTaskTemplate]:
    for _t, tmpl in arrivals:
        yield tmpl
