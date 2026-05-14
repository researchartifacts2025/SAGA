"""Workload generators."""

from saga.workload.base import (
    AgentTaskTemplate,
    WorkloadGenerator,
    WorkloadSpec,
)
from saga.workload.burst_gpt import BurstGPTWorkload
from saga.workload.swe_bench import SWEBenchWorkload
from saga.workload.web_arena import WebArenaWorkload


def build_workload(name: str, **kwargs) -> WorkloadGenerator:
    """Construct a workload generator by name."""
    name = name.lower().replace("-", "_")
    if name in ("swe_bench", "swebench", "swe"):
        return SWEBenchWorkload(**kwargs)
    if name in ("web_arena", "webarena", "web"):
        return WebArenaWorkload(**kwargs)
    if name in ("burst_gpt", "burstgpt", "burst"):
        return BurstGPTWorkload(**kwargs)
    raise ValueError(f"unknown workload: {name!r}")


__all__ = [
    "AgentTaskTemplate",
    "BurstGPTWorkload",
    "SWEBenchWorkload",
    "WebArenaWorkload",
    "WorkloadGenerator",
    "WorkloadSpec",
    "build_workload",
]
