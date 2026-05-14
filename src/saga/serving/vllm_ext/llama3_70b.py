"""Llama-3-70B-Instruct serving configuration.

The paper's primary evaluation model. We pin the exact configuration here so
the wall-clock numbers in Tables 3-10 are reproducible at the configuration
level (the underlying weights are licensed and not redistributed by SAGA).

Key invariants
--------------

* Tensor parallelism: 4 GPUs per instance (so 64 GPUs -> 16 instances, which
  is exactly the worker count assumed by :mod:`saga.serving.distributed`).
* Grouped Query Attention: 8 KV heads, 64 query heads, 128-dim per head.
* KV-cache footprint at 32K context: ~10.7 GB (matches paper Eq. text).
* dtype: FP16 weights and KV cache; FP32 reductions inside attention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Canonical Llama-3 family configuration used by the SAGA paper."""

    name: str
    hf_id: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    max_context: int
    tensor_parallel: int
    dtype: str

    @property
    def kv_bytes_per_token(self) -> int:
        # 2 (K + V) * n_kv_heads * head_dim * dtype_bytes
        dtype_bytes = 2 if self.dtype in ("fp16", "bf16") else 4
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes

    def kv_bytes_for_context(self, ctx_tokens: int) -> int:
        return self.kv_bytes_per_token * self.n_layers * ctx_tokens


LLAMA3_70B = ModelConfig(
    name="Llama-3-70B-Instruct",
    hf_id="meta-llama/Meta-Llama-3-70B-Instruct",
    n_layers=80,
    n_q_heads=64,
    n_kv_heads=8,
    head_dim=128,
    hidden_size=8192,
    max_context=32_768,
    tensor_parallel=4,
    dtype="fp16",
)


LLAMA3_8B = ModelConfig(
    name="Llama-3-8B-Instruct",
    hf_id="meta-llama/Meta-Llama-3-8B-Instruct",
    n_layers=32,
    n_q_heads=32,
    n_kv_heads=8,
    head_dim=128,
    hidden_size=4096,
    max_context=8_192,
    tensor_parallel=1,
    dtype="fp16",
)


def assert_paper_invariants(cfg: ModelConfig = LLAMA3_70B) -> None:
    """Cross-check the model config against the paper's stated numbers."""
    # §2.2 of the paper: ~10.7 GB at 32K context.
    bytes_at_32k = cfg.kv_bytes_for_context(32_768)
    gib = bytes_at_32k / (1024**3)
    if not 10.0 <= gib <= 11.5:
        raise AssertionError(
            f"Llama-3-70B KV footprint at 32K should be ~10.7 GiB; got {gib:.2f} GiB"
        )
    # 16 instances at TP=4 covers the 64-GPU cluster.
    if 64 % cfg.tensor_parallel != 0:
        raise AssertionError("TP degree must divide cluster GPU count (64).")


if __name__ == "__main__":  # pragma: no cover
    assert_paper_invariants()
    print(f"{LLAMA3_70B.name}: {LLAMA3_70B.kv_bytes_for_context(32_768) / 2**30:.2f} GiB / 32K ctx")
