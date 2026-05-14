"""WebArena-shaped browser-agent workload.

WebArena tasks have:

  * mean 18 steps (lighter than SWE-bench)
  * 4-8K prompt tokens per step (page content), 50-200 output tokens
  * mostly web_api tool calls; occasional file or database
"""

from __future__ import annotations

from saga.core.aeg import build_linear_aeg
from saga.core.types import Task, ToolType
from saga.utils.seeds import RNG
from saga.workload.base import (
    AgentTaskTemplate,
    WorkloadGenerator,
    WorkloadSpec,
    default_tool_plan,
)


_TOOL_MIX: tuple[tuple[ToolType, float], ...] = (
    (ToolType.WEB_API, 0.75),
    (ToolType.CODE_EXECUTION, 0.05),
    (ToolType.FILE_OPERATION, 0.15),
    (ToolType.DATABASE_QUERY, 0.05),
)


class WebArenaWorkload(WorkloadGenerator):
    """WebArena-like task templates."""

    name = "web_arena"

    def __init__(
        self,
        spec: WorkloadSpec | None = None,
        mean_steps: float = 18.0,
        max_steps: int = 80,
        prompt_min: int = 4_000,
        prompt_max: int = 8_000,
        output_min: int = 50,
        output_max: int = 200,
        observation_mean_tokens: int = 800,
        expected_tct_ms: float = 80_000.0,
    ) -> None:
        super().__init__(spec or WorkloadSpec(tag="web_arena"))
        self.mean_steps = mean_steps
        self.max_steps = max_steps
        self.prompt_min = prompt_min
        self.prompt_max = prompt_max
        self.output_min = output_min
        self.output_max = output_max
        self.observation_mean_tokens = observation_mean_tokens
        self.expected_tct_ms = expected_tct_ms

    def sample(self, rng: RNG, index: int, tenant_id: str) -> AgentTaskTemplate:
        n_steps = max(1, min(self.max_steps, int(round(rng.gamma(2.0, self.mean_steps / 2.0)))))
        tools: list[ToolType] = []
        opts = [t for t, _ in _TOOL_MIX]
        probs = [p for _, p in _TOOL_MIX]
        for _ in range(n_steps):
            tools.append(rng.choice(opts, probs))  # type: ignore[arg-type]
        prompt = rng.randint(self.prompt_min, self.prompt_max + 1)
        output = rng.randint(self.output_min, self.output_max + 1)
        obs = max(1, int(rng.gamma(2.0, self.observation_mean_tokens / 2.0)))

        aeg = build_linear_aeg(
            graph_id=f"web/{tenant_id}/{index}",
            n_steps=n_steps,
            tool_types=tools,
            prompt_tokens_est=prompt,
            output_tokens_est=output,
            observation_tokens_est=obs,
            termination_prob=1.0 / max(1.0, self.mean_steps),
            workload_kind="web_arena",
        )

        task = Task(
            task_id=f"{tenant_id}/web/{index:04d}",
            tenant_id=tenant_id,
            workload_kind="web_arena",
            submit_time=0.0,
            n_steps=n_steps,
            aeg_id=aeg.graph_id,
            expected_tct_ms=self.expected_tct_ms,
            tokens_prefilled_initial=prompt,
        )
        tool_plan = default_tool_plan(aeg, rng.fork("tool", index))
        return AgentTaskTemplate(task=task, aeg=aeg, tool_plan=tool_plan)
