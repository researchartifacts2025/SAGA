"""Tests for the vLLM v0.6.0 extension layer (WALRUBlockManagerHook, V1EngineHook).

Runs without vLLM installed -- the hooks exercise their in-process bookkeeping
path. Cluster-only behavior (real GPU eviction kernels) is gated separately.
"""

from __future__ import annotations

import pytest

from saga.core.aeg import build_linear_aeg
from saga.core.types import ToolType, Worker
from saga.scheduler.coordinator import CoordinatorConfig, GlobalCoordinator
from saga.serving.vllm_ext.llama3_70b import (
    LLAMA3_8B,
    LLAMA3_70B,
    ModelConfig,
    assert_paper_invariants,
)
from saga.serving.vllm_ext.paged_attention import WALRUBlockManagerHook
from saga.serving.vllm_ext.prefill_decode import PrefillDecodeBinder
from saga.serving.vllm_ext.v1_engine import V1EngineHook


@pytest.mark.unit
def test_llama3_70b_paper_invariants_hold() -> None:
    assert LLAMA3_70B.tensor_parallel == 4
    assert LLAMA3_70B.n_kv_heads == 8
    assert LLAMA3_70B.head_dim == 128
    assert LLAMA3_70B.n_layers == 80
    assert LLAMA3_70B.max_context == 32_768
    # Should not raise; the assertion validates paper §2.2's ~10.7 GiB number.
    assert_paper_invariants()


@pytest.mark.unit
def test_llama3_kv_bytes_per_token_matches_paper() -> None:
    # GQA Llama-3-70B: 2 (K + V) * 8 (n_kv) * 128 (d_head) * 2 (fp16) = 4096 B/token/layer
    assert LLAMA3_70B.kv_bytes_per_token == 2 * 8 * 128 * 2
    bytes_32k = LLAMA3_70B.kv_bytes_for_context(32_768)
    gib = bytes_32k / (1024**3)
    assert 10.0 <= gib <= 11.5


@pytest.mark.unit
def test_walru_block_manager_hook_admit_and_forget() -> None:
    hook = WALRUBlockManagerHook(capacity_tokens=10_000)
    aeg = build_linear_aeg(
        graph_id="t",
        n_steps=3,
        tool_types=[ToolType.CODE_EXECUTION] * 3,
        prompt_tokens_est=1_000,
        output_tokens_est=200,
        observation_tokens_est=100,
    )
    hook.register_aeg("s1", aeg, node=0)
    out = hook.admit("s1", n_tokens=1_000, now_ms=0.0)
    assert out["hit"] == 0
    assert out["regenerated_tokens"] == 1_000

    # Re-admit at the same node should be a hit.
    out2 = hook.admit("s1", n_tokens=1_000, now_ms=10.0)
    assert out2["hit"] == 1
    assert out2["regenerated_tokens"] == 0

    hook.forget("s1")
    assert not hook.manager.contains("s1")


@pytest.mark.unit
def test_walru_hook_signal_tool_call_sets_ttl() -> None:
    hook = WALRUBlockManagerHook(capacity_tokens=5_000)
    aeg = build_linear_aeg(
        graph_id="t",
        n_steps=2,
        tool_types=[ToolType.WEB_API, ToolType.NONE],
        prompt_tokens_est=500,
        output_tokens_est=100,
        observation_tokens_est=50,
    )
    hook.register_aeg("s1", aeg, node=0)
    hook.admit("s1", n_tokens=500, now_ms=0.0)
    deadline = hook.signal_tool_call("s1", ToolType.WEB_API, now_ms=0.0)
    assert deadline is not None and deadline > 0.0
    hook.signal_tool_return("s1")


@pytest.mark.unit
def test_walru_hook_install_without_vllm_is_idempotent() -> None:
    hook = WALRUBlockManagerHook(capacity_tokens=1_000)
    # No real engine; the hook logs a warning and records the install state.
    hook.install(vllm_engine=object())
    hook.install(vllm_engine=object())  # idempotent
    hook.uninstall()


@pytest.mark.unit
def test_v1_engine_hook_stats_track_steps() -> None:
    workers = [
        Worker(
            worker_id=i,
            node_id=0,
            gpu_indices=(i,),
            kv_capacity_tokens=1_000_000,
            decode_tokens_per_ms=38.0,
            prefill_tokens_per_ms=850.0,
        )
        for i in range(2)
    ]
    coord = GlobalCoordinator(workers=workers, cfg=CoordinatorConfig())
    hook = V1EngineHook(coordinator=coord, epoch_ms=100.0)

    class _FakeSched:
        def __init__(self):
            self.waiting = []
            self.running = []

        def schedule(self):
            return []

        def update_from_output(self, *_a, **_kw):
            return None

    class _FakeEngine:
        def __init__(self):
            self.scheduler = _FakeSched()

    engine = _FakeEngine()
    hook.install(engine)
    # Trigger one schedule + update cycle via the patched methods.
    engine.scheduler.schedule()
    engine.scheduler.update_from_output(None)
    assert hook.stats["steps"] == 1
    hook.uninstall()


@pytest.mark.unit
def test_prefill_decode_binder_install_without_cuda_is_safe() -> None:
    binder = PrefillDecodeBinder()
    binder.install(vllm_executor=object())  # no torch.cuda; warns, no raise
    # Calling prefetch_blocks without CUDA returns 0 bytes copied.
    n = binder.prefetch_blocks(None, None, None, None, None, None, None)
    assert n == 0
    binder.uninstall()


@pytest.mark.unit
def test_model_config_kv_bytes_scales_with_dtype() -> None:
    fp32 = ModelConfig(
        name="dummy",
        hf_id="x",
        n_layers=4,
        n_q_heads=8,
        n_kv_heads=8,
        head_dim=64,
        hidden_size=512,
        max_context=2048,
        tensor_parallel=1,
        dtype="fp32",
    )
    fp16 = ModelConfig(**{**fp32.__dict__, "dtype": "fp16"})
    assert fp32.kv_bytes_per_token == 2 * fp16.kv_bytes_per_token


@pytest.mark.unit
def test_llama3_8b_alternative_config_is_valid() -> None:
    assert LLAMA3_8B.tensor_parallel == 1
    assert LLAMA3_8B.n_layers == 32
    assert LLAMA3_8B.n_kv_heads == 8
