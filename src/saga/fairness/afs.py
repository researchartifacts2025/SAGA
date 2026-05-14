"""Agent Fair Share (AFS).

Per-tenant urgency is

    urgency_i(t) = (W_i - S_i(t)) / (deadline_i - t)

where ``W_i`` is the tenant's total estimated workload (sum of expected TCTs
of pending tasks), ``S_i(t)`` is cumulative service received, and the deadline
is taken to be ``1.5 * expected_tct`` per the SLO definition. Tenants with
higher urgency get higher priority; the coordinator allocates worker capacity
in proportion to urgency.

The Lyapunov-drift analysis (theorem_afs) gives a high-probability
completion-time bound; this module implements the allocation rule and exposes
the priority that the scheduler uses to break ties.
"""

from __future__ import annotations

from dataclasses import dataclass

from saga.core.types import Task


_SLO_MULTIPLIER = 1.5
_EPS = 1e-6


@dataclass
class TenantUrgency:
    """Per-tenant accounting for AFS."""

    tenant_id: str
    weight: float = 1.0
    pending_work_ms: float = 0.0
    cumulative_service_ms: float = 0.0
    earliest_deadline_ms: float = float("inf")
    n_pending: int = 0

    def urgency(self, now: float) -> float:
        if self.n_pending == 0 or self.pending_work_ms <= 0.0:
            return 0.0
        deadline_gap = max(self.earliest_deadline_ms - now, _EPS)
        return self.weight * self.pending_work_ms / deadline_gap


class AFSScheduler:
    """The fairness state machine."""

    def __init__(self) -> None:
        self._tenants: dict[str, TenantUrgency] = {}
        self._task_deadlines: dict[str, float] = {}
        self._task_remaining_ms: dict[str, float] = {}
        self.n_preemptions = 0

    # ---------------------------------------------------- registration

    def ensure_tenant(self, tenant_id: str, weight: float = 1.0) -> TenantUrgency:
        t = self._tenants.get(tenant_id)
        if t is None:
            t = TenantUrgency(tenant_id=tenant_id, weight=weight)
            self._tenants[tenant_id] = t
        return t

    def note_submit(self, tenant_id: str, task: Task) -> None:
        t = self.ensure_tenant(tenant_id)
        t.n_pending += 1
        t.pending_work_ms += task.expected_tct_ms
        deadline = task.submit_time + max(task.expected_tct_ms, 1.0) * _SLO_MULTIPLIER
        self._task_deadlines[task.task_id] = deadline
        self._task_remaining_ms[task.task_id] = task.expected_tct_ms
        t.earliest_deadline_ms = min(t.earliest_deadline_ms, deadline)

    def note_progress(self, tenant_id: str, task_id: str, gpu_ms: float) -> None:
        if gpu_ms <= 0.0:
            return
        t = self._tenants.get(tenant_id)
        if t is None:
            return
        t.cumulative_service_ms += gpu_ms
        remaining = self._task_remaining_ms.get(task_id, 0.0) - gpu_ms
        self._task_remaining_ms[task_id] = max(0.0, remaining)
        t.pending_work_ms = max(0.0, t.pending_work_ms - gpu_ms)

    def note_complete(self, tenant_id: str, task: Task, now: float) -> None:
        t = self._tenants.get(tenant_id)
        if t is None:
            return
        t.n_pending = max(0, t.n_pending - 1)
        self._task_remaining_ms.pop(task.task_id, None)
        self._task_deadlines.pop(task.task_id, None)
        if t.n_pending == 0:
            t.earliest_deadline_ms = float("inf")
            t.pending_work_ms = 0.0
        else:
            t.earliest_deadline_ms = self._earliest_deadline(tenant_id)

    def note_preemption(self) -> None:
        self.n_preemptions += 1

    def _earliest_deadline(self, tenant_id: str) -> float:
        earliest = float("inf")
        for tid, deadline in self._task_deadlines.items():
            if tid.startswith(f"{tenant_id}/") and deadline < earliest:
                earliest = deadline
        return earliest

    # --------------------------------------------------------- query

    def priority(self, tenant_id: str, now: float) -> float:
        t = self._tenants.get(tenant_id)
        if t is None:
            return 0.0
        return t.urgency(now)

    def allocation(self, now: float) -> dict[str, float]:
        """Return per-tenant capacity allocation summing to 1.0."""
        urgencies = {tid: t.urgency(now) for tid, t in self._tenants.items()}
        total = sum(urgencies.values())
        if total <= _EPS:
            n = max(1, sum(1 for u in urgencies.values() if u >= 0.0))
            return dict.fromkeys(urgencies, 1.0 / n)
        return {tid: u / total for tid, u in urgencies.items()}

    def refresh(self, now: float) -> None:
        """Recompute earliest-deadlines (called once per epoch)."""
        for tid, tenant in self._tenants.items():
            tenant.earliest_deadline_ms = self._earliest_deadline(tid)
            if tenant.n_pending == 0:
                tenant.earliest_deadline_ms = float("inf")

    # ---------------------------------------------------- preemption

    def should_preempt(
        self,
        low_priority_tenant: str,
        high_priority_tenant: str,
        block_duration_ms: float,
        threshold_ms: float = 500.0,
    ) -> bool:
        if block_duration_ms < threshold_ms:
            return False
        lo = self.priority(low_priority_tenant, now=0.0)
        hi = self.priority(high_priority_tenant, now=0.0)
        return hi > 2.0 * lo

    # ---------------------------------------------------------- stats

    def stats(self) -> dict[str, float]:
        return {
            "n_tenants": len(self._tenants),
            "n_preemptions": self.n_preemptions,
        }

    def tenant_ids(self) -> list[str]:
        return list(self._tenants.keys())

    def tenant(self, tenant_id: str) -> TenantUrgency | None:
        return self._tenants.get(tenant_id)
