// SAGA CUDA kernel declarations.
//
// All kernels target sm_80+ (A100) but compile down to sm_70 (V100) as well.
// Half-precision support is GQA-safe (n_kv * d_head per block, contiguous in
// d_head) so the layout matches vLLM's PagedAttention v2 block format.
//
// Each kernel comes in two flavors:
//   * `<name>_launch`   --- ABI-stable, host-side launcher invoked from C++.
//   * `<name>_pybind`   --- pybind11 wrapper around the launcher; takes
//                          torch::Tensor and returns torch::Tensor.
//
// Streams: every launcher accepts an explicit cudaStream_t so the prefetch
// path can overlap with the running decode batch (paper §4.3).

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace saga {
namespace cuda {

// -------------------------------------------------------- block layout
//
// vLLM-compatible paged KV-cache block descriptor. A block holds
// `block_size` tokens, each contributing `n_kv * d_head` half-precision
// values for the K matrix and the same for V. The layout is:
//
//     k_cache[block_id, head_id, slot_id, d_head_idx]
//     v_cache[block_id, head_id, slot_id, d_head_idx]
//
// where `block_id` indexes into the global block pool and `slot_id` is the
// position within the block. This matches vLLM's PagedAttention v2 format.

struct PagedBlockShape {
    std::int32_t block_size;    // tokens per block (vLLM default: 16)
    std::int32_t n_kv_heads;    // GQA-friendly: Llama-3-70B has 8
    std::int32_t head_dim;      // 128 for Llama-3-70B
    std::int32_t dtype_bytes;   // 2 for fp16/bf16, 4 for fp32
};

inline std::int64_t bytes_per_block(const PagedBlockShape& s) {
    // 2x for K and V; n_kv_heads * head_dim per token.
    return 2LL
         * static_cast<std::int64_t>(s.block_size)
         * static_cast<std::int64_t>(s.n_kv_heads)
         * static_cast<std::int64_t>(s.head_dim)
         * static_cast<std::int64_t>(s.dtype_bytes);
}

// ---------------------------------------------------- separate-stream prefetch
//
// Move `n_blocks` KV-cache blocks from the source pool (typically HBM on the
// current device) to the destination pool (HBM on the same or a different
// device). Performs an async memcpy and a fence-free dependency edge into
// the consumer stream.
//
// Returns the number of bytes copied. Asynchronous on the supplied stream.

std::int64_t prefetch_blocks_launch(
    const std::uint16_t* src_k,        // [n_src_blocks, n_kv, block_size, d_head], fp16
    const std::uint16_t* src_v,
    std::uint16_t* dst_k,
    std::uint16_t* dst_v,
    const std::int32_t* src_block_ids, // length == n_blocks
    const std::int32_t* dst_block_ids,
    std::int32_t n_blocks,
    const PagedBlockShape& shape,
    void* stream                       // cudaStream_t opaque-typed for ABI
);

// ------------------------------------------------------ KV migration
//
// Cross-device live migration (Llumnix-style). Pipelines NCCL P2P sends with
// the destination's reconstruction of the block-table mapping. The launcher
// returns once the send queue is drained but BEFORE the receive completes;
// the receiver pairs with `migration_recv_launch` to commit.

std::int64_t migration_send_launch(
    const std::uint16_t* k_blocks,
    const std::uint16_t* v_blocks,
    const std::int32_t* src_block_ids,
    std::int32_t n_blocks,
    std::int32_t peer_rank,
    const PagedBlockShape& shape,
    void* stream
);

std::int64_t migration_recv_launch(
    std::uint16_t* k_blocks,
    std::uint16_t* v_blocks,
    const std::int32_t* dst_block_ids,
    std::int32_t n_blocks,
    std::int32_t peer_rank,
    const PagedBlockShape& shape,
    void* stream
);

// ------------------------------------------------------- prefix overlap
//
// Compute the longest common prefix length between two token sequences, on
// device. Used by WA-LRU's `predict_reuse` when the eviction candidate set
// is large (> 1024) and host-side scoring is bottlenecked.

void prefix_overlap_launch(
    const std::int32_t* tokens_a,   // [n_a]
    const std::int32_t* tokens_b,   // [n_b]
    std::int32_t n_a,
    std::int32_t n_b,
    std::int32_t* out_overlap,      // scalar, device pointer
    void* stream
);

// Batched variant: compute overlap between one cached session and `n_succ`
// candidate successors in parallel.

void prefix_overlap_batch_launch(
    const std::int32_t* tokens_cached,
    std::int32_t n_cached,
    const std::int32_t* tokens_succ_flat,  // CSR-flattened
    const std::int32_t* succ_offsets,      // [n_succ + 1]
    std::int32_t n_succ,
    std::int32_t* out_overlap,             // [n_succ]
    void* stream
);

// ----------------------------------------------------- WA-LRU scoring
//
// Compute `P_evict = alpha*R + beta*(1-Preuse) + gamma*S` for every entry
// in parallel. Inputs are device tensors in float32; output is a single
// scalar holding the argmin index. The launcher uses cooperative-group
// reduction so the kernel runs in a single grid launch.

void walru_score_launch(
    const float* recency,        // [n], in [0, 1]
    const float* preuse,         // [n], in [0, 1]
    const float* size_norm,      // [n], in [0, 1]
    const std::uint8_t* pinned,  // [n], 0 or 1
    std::int32_t n,
    float alpha, float beta, float gamma,
    std::int32_t* out_argmin,    // scalar
    float* out_min_score,        // scalar
    void* stream
);

// ----------------------------------------------------- block compaction
//
// Compact a fragmented KV-cache pool. After many evictions, free blocks may
// be scattered through the block table; this kernel moves live blocks to
// the low end of the pool in a single pass so subsequent allocations get
// contiguous memory (paper §6.1, "Implementation: KV Cache Manager").

std::int32_t compact_pool_launch(
    std::uint16_t* k_pool,
    std::uint16_t* v_pool,
    std::int32_t* block_table,         // [n_logical], slot -> physical
    const std::uint8_t* alive_mask,    // [n_physical]
    std::int32_t n_physical,
    const PagedBlockShape& shape,
    void* stream
);

// ------------------------------------------------- TTL-aware paged attention
//
// Three companion kernels (see csrc/cuda/paged_attention_walru.cu) that
// implement the per-block TTL bookkeeping introduced by SAGA. They expose
// the device-side counterpart of Algorithm 1 (tool-call-aware TTL) and the
// PagedAttention victim picker that combines WA-LRU scoring with the TTL
// state in a single grid launch (avoiding the two-pass scoring + scan that
// the host-side fallback uses).
//
// The kernels operate on an opaque ``BlockMeta`` struct laid out as 32-byte
// records --- defined inside the .cu file. Callers obtain a pointer to a
// device buffer of those records via the pybind wrapper.

struct BlockMeta;  // opaque to header consumers

// TTL expiry scan: marks blocks whose deadline has passed and not pinned.
std::int32_t ttl_decay_launch(
    const BlockMeta* blocks,
    std::int64_t now_ms,
    std::int32_t n_blocks,
    std::uint8_t* out_expired_mask,
    std::int32_t* out_n_expired,
    void* stream
);

// Block-cooperative scan that compacts an expired-mask into a free list.
std::int32_t free_list_compact_launch(
    const std::uint8_t* expired_mask,
    std::int32_t n_blocks,
    std::int32_t* out_free_list,
    std::int32_t* out_n_free,
    void* stream
);

// One-launch WA-LRU + TTL victim picker. Pinned blocks score +inf; expired
// blocks score -inf; all others use the standard WA-LRU formula. The
// returned ``argmin`` is the chosen victim's index.
std::int32_t paged_attention_walru_pick_launch(
    const BlockMeta* blocks,
    const float* recency,
    const float* preuse,
    const float* size_norm,
    std::int64_t now_ms,
    std::int32_t n_blocks,
    float alpha,
    float beta,
    float gamma,
    std::int32_t* out_argmin,
    float* out_min_score,
    void* stream
);

}  // namespace cuda
}  // namespace saga
