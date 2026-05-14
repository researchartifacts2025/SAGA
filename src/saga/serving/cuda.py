"""Pythonic wrapper over the ``saga._cuda`` extension.

Loaded by :mod:`saga.serving.vllm_ext.prefill_decode` (separate-stream
prefetch), :mod:`saga.serving.distributed.coordinator` (KV-migration
launches), and the wall-clock benchmark harness. On the 64-A100 cluster
every call here dispatches to a real CUDA kernel from
:file:`csrc/cuda/*.cu`; the wrappers return ``int`` byte counts so the
worker can verify the transfer completed.

For development on machines without nvcc/torch.cuda the wrapper degrades
to ``int(0)`` returns and ``is_cuda_available() == False`` so unit tests
import cleanly. Production deployments **must** build the extension::

    pip install "torch>=2.1.2" "pybind11>=2.11"
    python setup_cuda.py build_ext --inplace
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    from saga import _cuda as _ext  # type: ignore[attr-defined]

    _HAS_CUDA = True
except Exception:  # pragma: no cover -- absent on most dev boxes
    _ext = None
    _HAS_CUDA = False


def is_cuda_available() -> bool:
    """Return ``True`` iff the compiled ``saga._cuda`` module loaded."""
    return _HAS_CUDA


def cuda_build_info() -> str:
    if _HAS_CUDA and _ext is not None and hasattr(_ext, "build_info"):
        try:
            return str(_ext.build_info())
        except Exception:  # pragma: no cover
            pass
    return "saga._cuda (not built) -- CPU fallback active"


@dataclass(frozen=True)
class PagedBlockShape:
    """vLLM-compatible paged KV-cache block descriptor.

    Mirrors the C++ struct in ``csrc/cuda/saga_cuda.h``. The Python side
    exposes the same fields so the pybind11 wrapper can read them by name.
    """

    block_size: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2  # fp16/bf16

    @property
    def bytes_per_block(self) -> int:
        return 2 * self.block_size * self.n_kv_heads * self.head_dim * self.dtype_bytes


# --------------------------------------------------------- kernels


def prefetch_blocks(
    src_k: Any,
    src_v: Any,
    dst_k: Any,
    dst_v: Any,
    src_ids: Any,
    dst_ids: Any,
    shape: PagedBlockShape,
    stream_ptr: int = 0,
) -> int:
    """Separate-stream KV-cache prefetch."""
    if not _HAS_CUDA or _ext is None:
        return 0
    return int(
        _ext.prefetch_blocks(
            src_k, src_v, dst_k, dst_v, src_ids, dst_ids, shape, stream_ptr=stream_ptr
        )
    )


def migration_send(
    k_blocks: Any,
    v_blocks: Any,
    src_block_ids: Any,
    peer_rank: int,
    shape: PagedBlockShape,
    stream_ptr: int = 0,
) -> int:
    if not _HAS_CUDA or _ext is None:
        return 0
    return int(
        _ext.migration_send(
            k_blocks, v_blocks, src_block_ids, peer_rank, shape, stream_ptr=stream_ptr
        )
    )


def migration_recv(
    k_blocks: Any,
    v_blocks: Any,
    dst_block_ids: Any,
    peer_rank: int,
    shape: PagedBlockShape,
    stream_ptr: int = 0,
) -> int:
    if not _HAS_CUDA or _ext is None:
        return 0
    return int(
        _ext.migration_recv(
            k_blocks, v_blocks, dst_block_ids, peer_rank, shape, stream_ptr=stream_ptr
        )
    )


def prefix_overlap_batch(
    tokens_cached: Any,
    tokens_succ_flat: Any,
    succ_offsets: Any,
    stream_ptr: int = 0,
) -> Any:
    if not _HAS_CUDA or _ext is None:
        return None
    return _ext.prefix_overlap_batch(
        tokens_cached, tokens_succ_flat, succ_offsets, stream_ptr=stream_ptr
    )


def walru_score(
    recency: Any,
    preuse: Any,
    size_norm: Any,
    pinned: Any,
    alpha: float,
    beta: float,
    gamma: float,
    stream_ptr: int = 0,
) -> tuple[Any, Any] | None:
    if not _HAS_CUDA or _ext is None:
        return None
    return _ext.walru_score(
        recency, preuse, size_norm, pinned, alpha, beta, gamma, stream_ptr=stream_ptr
    )


def compact_pool(
    k_pool: Any,
    v_pool: Any,
    block_table: Any,
    alive_mask: Any,
    shape: PagedBlockShape,
    stream_ptr: int = 0,
) -> int:
    if not _HAS_CUDA or _ext is None:
        return 0
    return int(
        _ext.compact_pool(
            k_pool, v_pool, block_table, alive_mask, shape, stream_ptr=stream_ptr
        )
    )


__all__ = [
    "PagedBlockShape",
    "compact_pool",
    "cuda_build_info",
    "is_cuda_available",
    "migration_recv",
    "migration_send",
    "prefetch_blocks",
    "prefix_overlap_batch",
    "walru_score",
]
