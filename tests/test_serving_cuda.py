"""Tests for the saga.serving.cuda wrapper.

Runs without saga._cuda built; the wrapper should report unavailability
gracefully and every entry point should become a safe no-op.
"""

from __future__ import annotations

import pytest

from saga.serving.cuda import (
    PagedBlockShape,
    compact_pool,
    cuda_build_info,
    is_cuda_available,
    migration_recv,
    migration_send,
    prefetch_blocks,
    prefix_overlap_batch,
    walru_score,
)


@pytest.mark.unit
def test_is_cuda_available_returns_bool() -> None:
    assert isinstance(is_cuda_available(), bool)


@pytest.mark.unit
def test_cuda_build_info_returns_string() -> None:
    info = cuda_build_info()
    assert isinstance(info, str)
    assert len(info) > 0


@pytest.mark.unit
def test_paged_block_shape_bytes_per_block_matches_paper() -> None:
    # Llama-3-70B GQA defaults from the paper: 16 tokens/block * 8 KV heads *
    # 128 head_dim * 2 bytes (fp16) for K + same for V.
    shape = PagedBlockShape()
    assert shape.bytes_per_block == 2 * 16 * 8 * 128 * 2
    assert shape.block_size == 16
    assert shape.n_kv_heads == 8
    assert shape.head_dim == 128
    assert shape.dtype_bytes == 2


@pytest.mark.unit
def test_kernels_return_no_op_when_cuda_unavailable() -> None:
    if is_cuda_available():
        pytest.skip("CUDA build present; this is a fallback-only test")

    shape = PagedBlockShape()
    assert prefetch_blocks(None, None, None, None, None, None, shape) == 0
    assert migration_send(None, None, None, peer_rank=1, shape=shape) == 0
    assert migration_recv(None, None, None, peer_rank=1, shape=shape) == 0
    assert prefix_overlap_batch(None, None, None) is None
    assert walru_score(None, None, None, None, 0.3, 0.5, 0.2) is None
    assert compact_pool(None, None, None, None, shape) == 0


@pytest.mark.unit
def test_paged_block_shape_is_immutable() -> None:
    shape = PagedBlockShape()
    with pytest.raises(Exception):
        shape.block_size = 32  # type: ignore[misc]
