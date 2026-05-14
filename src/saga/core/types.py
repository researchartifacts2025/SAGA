"""Domain types shared across SAGA.

The objects here are pure data carriers (no behaviour). They are deliberately
lightweight: a discrete-event simulator instantiates millions of them, so
``slots`` and small attribute counts matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ToolType(str, Enum):
    """Tool categories that an agent step can invoke.

    Each category has its own empirical latency distribution. The values match
    the categories used in the per-tool latency table.
    """

    CODE_EXECUTION = "code_execution"
    FILE_OPERATION = "file_operation"
    WEB_API = "web_api"
    DATABASE_QUERY = "database_query"
    NONE = "none"


class SessionState(str, Enum):
    """Lifecycle state of a session."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation in an agent step.

    ``duration_ms`` is the realized wall-clock duration once the tool completes.
    Predicted duration lives separately in the TTL module.
    """

    tool_type: ToolType
    duration_ms: float
    observation_tokens: int = 0


@dataclass
class KVCacheEntry:
    """A KV-cache entry for one session on one worker.

    Sizes are in tokens; the cache manager converts to bytes using the
    bytes-per-token coefficient set by the cluster config.
    """

    session_id: str
    worker_id: int
    n_tokens: int
    last_access_time: float
    creation_time: float
    aeg_node_index: int = 0
    ttl_deadline: float = float("inf")
    pinned: bool = False

    def is_expired(self, now: float) -> bool:
        return now >= self.ttl_deadline and not self.pinned


@dataclass
class Task:
    """An agent task (a whole program), composed of multiple LLM steps.

    A task is the unit of scheduling under SAGA's program-level abstraction.
    """

    task_id: str
    tenant_id: str
    workload_kind: str
    submit_time: float
    n_steps: int
    aeg_id: str
    expected_tct_ms: float = 0.0

    start_time: float = -1.0
    completion_time: float = -1.0
    n_inference_calls: int = 0
    n_tool_calls: int = 0
    n_cache_evictions: int = 0
    n_migrations: int = 0
    n_regenerated_tokens: int = 0
    tokens_prefilled_initial: int = 0
    succeeded: bool = False

    @property
    def is_complete(self) -> bool:
        return self.completion_time >= 0.0

    @property
    def tct_ms(self) -> float:
        if not self.is_complete:
            return -1.0
        return self.completion_time - self.submit_time

    def met_slo(self, slo_multiplier: float) -> bool:
        if not self.is_complete or self.expected_tct_ms <= 0.0:
            return False
        return self.tct_ms <= slo_multiplier * self.expected_tct_ms


@dataclass
class Session:
    """A live agent session executing on the cluster.

    A session corresponds 1-1 with a Task; the separation lets us treat
    cache/locality bookkeeping (Session) independently from end-to-end
    completion bookkeeping (Task).
    """

    session_id: str
    task: Task
    aeg_node_index: int = 0
    state: SessionState = SessionState.PENDING
    worker_id: int = -1
    cached_tokens: int = 0
    context_tokens: int = 0
    last_step_completion: float = -1.0
    tool_release_time: float = -1.0
    n_steps_completed: int = 0

    @property
    def tenant_id(self) -> str:
        return self.task.tenant_id


@dataclass
class Tenant:
    """A tenant (multi-task workload owner) for fair-share accounting."""

    tenant_id: str
    weight: float = 1.0
    submitted_tasks: int = 0
    completed_tasks: int = 0
    cumulative_service_ms: float = 0.0


@dataclass
class Worker:
    """A single LLM-serving worker (one TP group on a subset of GPUs).

    The cache pool is owned by the worker; the global coordinator never touches
    it directly.
    """

    worker_id: int
    node_id: int
    gpu_indices: tuple[int, ...]
    kv_capacity_tokens: int
    decode_tokens_per_ms: float
    prefill_tokens_per_ms: float

    queue_depth: int = 0
    current_session_id: str | None = None
    cache_used_tokens: int = 0
    busy_until: float = 0.0
    cumulative_busy_ms: float = 0.0
    n_steals_in: int = 0
    n_steals_out: int = 0
    n_migrations_received: int = 0
    sessions_seen: set[str] = field(default_factory=set)

    @property
    def memory_pressure(self) -> float:
        if self.kv_capacity_tokens <= 0:
            return 0.0
        return min(1.0, self.cache_used_tokens / float(self.kv_capacity_tokens))

    @property
    def utilization(self) -> float:
        if self.busy_until <= 0.0:
            return 0.0
        return min(1.0, self.cumulative_busy_ms / max(self.busy_until, 1.0))
