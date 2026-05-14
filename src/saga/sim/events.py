"""Event types and a min-heap priority queue keyed on time."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """The kinds of events the simulator schedules."""

    ARRIVAL = "arrival"
    INFERENCE_START = "inference_start"
    INFERENCE_END = "inference_end"
    TOOL_END = "tool_end"
    EPOCH_TICK = "epoch_tick"
    TASK_COMPLETE = "task_complete"
    MIGRATION_END = "migration_end"
    CACHE_EXPIRE = "cache_expire"


@dataclass(order=False)
class Event:
    """A scheduled event.

    Stored in a heap ordered by ``(time, sequence)`` so events at the same
    time fire in insertion order (FIFO tie-breaking).
    """

    time: float
    kind: EventKind
    session_id: str | None = None
    worker_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    cancelled: bool = False

    def __lt__(self, other: Event) -> bool:
        if self.time != other.time:
            return self.time < other.time
        return self.seq < other.seq

    def cancel(self) -> None:
        self.cancelled = True


class EventQueue:
    """Min-heap priority queue of events."""

    __slots__ = ("_counter", "_heap")

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def push(
        self,
        time: float,
        kind: EventKind,
        session_id: str | None = None,
        worker_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        ev = Event(
            time=time,
            kind=kind,
            session_id=session_id,
            worker_id=worker_id,
            payload=payload or {},
            seq=next(self._counter),
        )
        heapq.heappush(self._heap, ev)
        return ev

    def pop(self) -> Event:
        while self._heap:
            ev = heapq.heappop(self._heap)
            if not ev.cancelled:
                return ev
        raise IndexError("pop from empty event queue")

    def peek_time(self) -> float | None:
        for ev in self._heap:
            if not ev.cancelled:
                return ev.time
        return None
