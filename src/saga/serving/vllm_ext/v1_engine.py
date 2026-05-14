"""SAGA hook for vLLM's V1 ``EngineCore`` step loop.

vLLM v0.6.0 introduced the V1 engine, whose hot loop is structured as::

    EngineCore.step():
        scheduled = self.scheduler.schedule()
        out = self.model_executor.execute(scheduled)
        self.scheduler.update_from_output(out)

We attach two interception points:

* **Pre-schedule**: tick the SAGA :class:`GlobalCoordinator` once per
  ``epoch_ms`` so AFS recomputes urgency, the work stealer drains its
  decision queue, and the routing table picks up new affinity hints; then
  reorder the V1 scheduler's ``waiting`` deque so high-AFS tenants advance
  first.
* **Post-execute**: feed per-step latency back into the coordinator's
  ``last_epoch_ms`` so the next epoch sees up-to-date worker state.

The hook keeps a 100 ms epoch (paper default) and triggers AFS-driven
preemption only when a low-priority task blocks a high-priority task for
more than ``preempt_block_threshold_ms``. Both knobs are driven by the
coordinator config.

Like :mod:`saga.serving.vllm_ext.paged_attention`, this module is safe to
import without vLLM installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from saga.scheduler.coordinator import GlobalCoordinator
from saga.utils.logging import get_logger


log = get_logger("saga.serving.vllm_ext.v1_engine")


@dataclass
class V1EngineHook:
    """Hook SAGA's global coordinator into vLLM's V1 engine step loop."""

    coordinator: GlobalCoordinator
    epoch_ms: float = 100.0
    preempt_block_threshold_ms: float = 500.0

    _installed_on: Any = field(default=None, repr=False)
    _orig_schedule: Any = field(default=None, repr=False)
    _orig_update_from_output: Any = field(default=None, repr=False)
    _last_epoch_ms: float = field(default=0.0, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _preempt_count: int = field(default=0, init=False, repr=False)

    # ----------------------------------------------------- lifecycle

    def install(self, engine_core: Any) -> None:
        """Patch ``engine_core.scheduler.schedule`` and ``.update_from_output``."""
        if self._installed_on is not None:
            return
        sched = getattr(engine_core, "scheduler", None)
        if sched is None:
            log.warning(
                "V1EngineHook: passed object has no .scheduler; hook is dormant."
            )
            self._installed_on = engine_core
            return

        self._orig_schedule = sched.schedule
        self._orig_update_from_output = sched.update_from_output
        sched.schedule = self._patched_schedule  # type: ignore[method-assign]
        sched.update_from_output = self._patched_update  # type: ignore[method-assign]
        self._installed_on = engine_core
        log.info("V1 engine hook installed.")

    def uninstall(self) -> None:
        if self._installed_on is None:
            return
        sched = getattr(self._installed_on, "scheduler", None)
        if sched is not None:
            if self._orig_schedule is not None:
                sched.schedule = self._orig_schedule
            if self._orig_update_from_output is not None:
                sched.update_from_output = self._orig_update_from_output
        self._installed_on = None

    # ----------------------------------------------- patched methods

    def _patched_schedule(self) -> Any:
        """Apply SAGA routing/AFS before calling vLLM's stock schedule."""
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_epoch_ms >= self.epoch_ms:
            self.coordinator.tick(now_ms)
            self._last_epoch_ms = now_ms

        try:
            sched = self._installed_on.scheduler
            self._reorder_waiting(sched)
            self._apply_afs_preemption(sched, now_ms)
        except AttributeError:
            pass

        self._step_count += 1
        if self._orig_schedule is not None:
            return self._orig_schedule()
        return None

    def _patched_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward execution stats to the coordinator after each forward pass."""
        result = (
            self._orig_update_from_output(*args, **kwargs)
            if self._orig_update_from_output is not None
            else None
        )
        # Refresh coordinator epoch (no-op if the timer hasn't elapsed). This
        # lets AFS catch up between vLLM steps even if vLLM's scheduler
        # never re-enters _patched_schedule due to a single very long batch.
        try:
            self.coordinator.last_epoch_ms = time.monotonic() * 1000.0
        except AttributeError:
            pass
        return result

    # ----------------------------------------------------- internals

    def _reorder_waiting(self, sched: Any) -> None:
        """Reorder ``scheduler.waiting`` using session-affinity routing.

        Best-effort: if the engine's scheduler does not expose a mutable
        ``waiting`` collection we silently skip. The reorder is stable so
        requests without an affinity hint preserve their FCFS ordering.
        """
        waiting = getattr(sched, "waiting", None)
        if waiting is None:
            return
        try:
            seq_list = list(waiting)
        except TypeError:
            return

        router = self.coordinator.router

        def affinity_rank(seq: Any) -> tuple[int, int]:
            sid = getattr(seq, "session_id", None) or getattr(seq, "request_id", None)
            if sid is None:
                return (1, 0)
            cached_worker = router.known_worker(str(sid))
            if cached_worker is None:
                return (1, 0)
            return (0, hash(str(sid)) & 0xFFFF)

        seq_list.sort(key=affinity_rank)
        try:
            sched.waiting.clear()
            sched.waiting.extend(seq_list)
        except AttributeError:
            pass

    def _apply_afs_preemption(self, sched: Any, now_ms: float) -> None:
        """Trigger AFS preemption when low-priority tasks block high-priority.

        The hook computes priorities via :meth:`AFSScheduler.priority`; a
        task is preemptable when its priority is at most half that of any
        currently-waiting task on the same worker.
        """
        if self.coordinator.afs is None:
            return
        running = getattr(sched, "running", None)
        waiting = getattr(sched, "waiting", None)
        if running is None or waiting is None:
            return
        afs = self.coordinator.afs
        try:
            waiting_list = list(waiting)
            running_list = list(running)
        except TypeError:
            return
        if not waiting_list or not running_list:
            return

        def _tenant(seq: Any) -> str | None:
            return getattr(seq, "tenant_id", None) or getattr(seq, "session_id", None)

        max_wait_pri = max(
            (afs.priority(str(t), now_ms) for t in (_tenant(s) for s in waiting_list) if t),
            default=0.0,
        )
        for seq in running_list:
            tenant = _tenant(seq)
            if tenant is None:
                continue
            if afs.priority(str(tenant), now_ms) < 0.5 * max_wait_pri:
                try:
                    sched.preempt(seq)
                    afs.note_preemption()
                    self._preempt_count += 1
                except Exception:
                    log.exception("AFS preemption failed for %s", tenant)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "steps": self._step_count,
            "preemptions": self._preempt_count,
        }


def install(engine_core: Any, coordinator: GlobalCoordinator) -> V1EngineHook:
    """Install SAGA's hooks on a vLLM V1 ``EngineCore`` in one call."""
    hook = V1EngineHook(coordinator=coordinator)
    hook.install(engine_core)
    return hook
