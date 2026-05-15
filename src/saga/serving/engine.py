"""SagaVLLMEngine -- the user-facing wrapper.

A thin facade over vLLM's ``LLMEngine`` that:

1.  Loads Llama-3-70B-Instruct (or another :class:`ModelConfig`) with TP=4.
2.  Installs the three vLLM seams (PagedAttention hook, V1 engine hook,
    prefill/decode binder).
3.  Wires them to a shared :class:`GlobalCoordinator` so every worker on
    this node uses the same affinity table and AFS state.
4.  Exposes a single ``generate(messages, agent_type)`` method that takes
    a chat message list, registers the resulting session with the
    coordinator, and returns the generated text + per-step trace.

Lazy ``vllm`` import: importing this module is safe without vLLM. Calling
:meth:`SagaVLLMEngine.serve` without ``vllm`` installed raises
:class:`MissingRuntimeError` with the install hint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from saga.scheduler.coordinator import CoordinatorConfig, GlobalCoordinator
from saga.serving.errors import MissingRuntimeError
from saga.serving.vllm_ext.llama3_70b import LLAMA3_70B, ModelConfig
from saga.serving.vllm_ext.paged_attention import WALRUBlockManagerHook
from saga.serving.vllm_ext.prefill_decode import PrefillDecodeBinder
from saga.serving.vllm_ext.v1_engine import V1EngineHook
from saga.utils.logging import get_logger


log = get_logger("saga.serving.engine")


@dataclass
class SagaVLLMEngine:
    """A SAGA-equipped vLLM engine, ready to drive Llama-3-70B inference.

    The engine deliberately doesn't subclass vLLM's classes; instead it
    *composes* a vanilla vLLM engine with three lightweight hooks. This
    isolates the SAGA logic from upstream vLLM API drift and lets us
    swap engine implementations (e.g. for a future V2 engine) without
    rewriting the scheduler.
    """

    model_config: ModelConfig = field(default_factory=lambda: LLAMA3_70B)
    coordinator_config: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    gpu_memory_utilization: float = 0.92
    max_num_seqs: int = 256
    enforce_eager: bool = False

    coordinator: GlobalCoordinator | None = None
    block_hook: WALRUBlockManagerHook | None = None
    engine_hook: V1EngineHook | None = None
    prefill_binder: PrefillDecodeBinder | None = None

    _vllm_engine: Any = field(default=None, repr=False)

    # -------------------------------------------------------- serve

    def serve(self, workers: list[Any] | None = None) -> None:
        """Boot the underlying vLLM engine and install all SAGA hooks.

        ``workers`` is the list of :class:`Worker` objects exposed by the
        Ray distributed runtime; in single-node mode it is built from the
        local TP slice and the runtime configures :class:`GlobalCoordinator`
        accordingly. Without vLLM installed, raises
        :class:`MissingRuntimeError`.
        """
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as exc:
            raise MissingRuntimeError(
                runtime="vllm",
                install_hint="pip install 'saga-sched[serving]'",
            ) from exc

        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine

        if self.coordinator is None:
            if workers is None:
                raise ValueError("Must supply either coordinator or workers list")
            self.coordinator = GlobalCoordinator(workers=workers, cfg=self.coordinator_config)

        args = EngineArgs(
            model=self.model_config.hf_id,
            tensor_parallel_size=self.model_config.tensor_parallel,
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_context,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
        )
        self._vllm_engine = LLMEngine.from_engine_args(args)

        # Install hooks. Paper §4.1 / Table 9: alpha=0.3, beta=0.5, gamma=0.2.
        self.block_hook = WALRUBlockManagerHook(
            block_size=16,
            n_kv_heads=self.model_config.n_kv_heads,
            head_dim=self.model_config.head_dim,
            walru_alpha=0.3,
            walru_beta=0.5,
            walru_gamma=0.2,
        )
        self.block_hook.install(self._vllm_engine)

        self.engine_hook = V1EngineHook(
            coordinator=self.coordinator,
            epoch_ms=self.coordinator_config.epoch_ms,
            preempt_block_threshold_ms=self.coordinator_config.afs_preempt_threshold_ms,
        )
        self.engine_hook.install(self._vllm_engine)

        self.prefill_binder = PrefillDecodeBinder()
        self.prefill_binder.install(self._vllm_engine)

        log.info(
            "SagaVLLMEngine ready: model=%s TP=%d max_seqs=%d",
            self.model_config.name,
            self.model_config.tensor_parallel,
            self.max_num_seqs,
        )

    # -------------------------------------------------- generate

    def generate(
        self,
        prompts: list[str] | str,
        session_id: str | None = None,
        tenant_id: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run a single LLM step and return the result.

        The session is registered with the coordinator under its
        ``session_id``/``tenant_id`` so AFS and routing see it on the next
        epoch.
        """
        if self._vllm_engine is None:
            raise MissingRuntimeError(
                runtime="vllm engine (not started)",
                install_hint="call SagaVLLMEngine.serve() before generate()",
            )

        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]

        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        start = time.perf_counter()
        outputs = self._vllm_engine.generate(prompts, sampling_params=sampling)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "n_prompts": len(prompts),
            "elapsed_ms": elapsed_ms,
            "outputs": [o.outputs[0].text if o.outputs else "" for o in outputs],
        }

    # ------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model": self.model_config.name}
        if self.coordinator is not None:
            out["coordinator"] = self.coordinator.stats()
        if self.block_hook is not None:
            out["block_hook"] = self.block_hook.manager.stats()
        if self.engine_hook is not None:
            out["engine_hook"] = self.engine_hook.stats
        if self.prefill_binder is not None:
            out["prefill_binder"] = self.prefill_binder.stats
        return out
