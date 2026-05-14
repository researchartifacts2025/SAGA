"""SWE-bench-shaped agent workload.

SWE-bench-style coding agents have:

  * mean step count 37, max 150 (long-tailed; Negative-binomial-ish)
  * per step 2-4K prompt tokens, 100-500 output tokens
  * tools drawn from {code_execution, file_operation, web_api} with bias
    toward code/file
  * one tenant by default (single-stream); multi-tenant runs simply duplicate
    the stream

Numbers come from the per-workload table in the paper.
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
    (ToolType.CODE_EXECUTION, 0.45),
    (ToolType.FILE_OPERATION, 0.40),
    (ToolType.WEB_API, 0.10),
    (ToolType.DATABASE_QUERY, 0.05),
)


class SWEBenchWorkload(WorkloadGenerator):
    """SWE-bench-like task templates."""

    name = "swe_bench"

    def __init__(
        self,
        spec: WorkloadSpec | None = None,
        mean_steps: float = 37.0,
        max_steps: int = 150,
        prompt_min: int = 2_000,
        prompt_max: int = 4_000,
        output_min: int = 100,
        output_max: int = 500,
        observation_mean_tokens: int = 350,
        expected_tct_ms: float = 200_000.0,
    ) -> None:
        super().__init__(spec or WorkloadSpec(tag="swe_bench"))
        self.mean_steps = mean_steps
        self.max_steps = max_steps
        self.prompt_min = prompt_min
        self.prompt_max = prompt_max
        self.output_min = output_min
        self.output_max = output_max
        self.observation_mean_tokens = observation_mean_tokens
        self.expected_tct_ms = expected_tct_ms

    def sample(self, rng: RNG, index: int, tenant_id: str) -> AgentTaskTemplate:
        n_steps = self._sample_step_count(rng)
        tools: list[ToolType] = []
        for _ in range(n_steps):
            tools.append(_pick_tool(rng))
        prompt = rng.randint(self.prompt_min, self.prompt_max + 1)
        output = rng.randint(self.output_min, self.output_max + 1)
        obs = max(1, int(rng.gamma(2.0, self.observation_mean_tokens / 2.0)))

        aeg = build_linear_aeg(
            graph_id=f"swe/{tenant_id}/{index}",
            n_steps=n_steps,
            tool_types=tools,
            prompt_tokens_est=prompt,
            output_tokens_est=output,
            observation_tokens_est=obs,
            termination_prob=1.0 / max(1.0, self.mean_steps),
            workload_kind="swe_bench",
        )

        task = Task(
            task_id=f"{tenant_id}/swe/{index:04d}",
            tenant_id=tenant_id,
            workload_kind="swe_bench",
            submit_time=0.0,
            n_steps=n_steps,
            aeg_id=aeg.graph_id,
            expected_tct_ms=self.expected_tct_ms,
            tokens_prefilled_initial=prompt,
        )

        tool_plan = default_tool_plan(aeg, rng.fork("tool", index))
        return AgentTaskTemplate(task=task, aeg=aeg, tool_plan=tool_plan)

    def _sample_step_count(self, rng: RNG) -> int:
        gap = self.mean_steps
        if gap <= 0.0:
            return 1
        sample = int(round(rng.gamma(2.0, gap / 2.0)))
        return max(1, min(self.max_steps, sample))


def _pick_tool(rng: RNG) -> ToolType:
    options = [t for t, _ in _TOOL_MIX]
    probs = [p for _, p in _TOOL_MIX]
    return rng.choice(options, probs)  # type: ignore[return-value]
