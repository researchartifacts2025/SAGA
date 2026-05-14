// TTL-aware PagedAttention block table maintenance.
//
// PagedAttention (Kwon et al. 2023) keeps per-sequence KV cache in fixed-size
// blocks indirected through a logical-to-physical block table. SAGA adds two
// pieces of bookkeeping on top:
//
//   1. Per-block TTL deadline (Algorithm 1 in the paper). When the agent
//      enters a tool call, all blocks belonging to that session inherit a
//      TTL = p95(latency_tool) * (1 - 0.5 * memory_pressure). A block whose
//      ``now_ms > ttl_deadline`` is eligible for eviction even if it would
//      otherwise have a high WA-LRU score.
//
//   2. Per-block pin status (set during prefill, cleared on session end).
//      Pinned blocks are never evicted; this matches vLLM's
//      ``BlockSpaceManagerV2`` semantics for the "running" sequence list.
//
// This translation unit hosts three kernels and the host-side dispatch:
//
//   - ttl_decay_kernel: scans the resident block pool, marks blocks whose
//     TTL has expired in ``out_expired_mask``.
//   - free_list_compact_kernel: takes the expired-mask and writes back a
//     compacted free list usable by ``BlockSpaceManagerV2.allocate``.
//   - paged_attention_walru_pick_kernel: combines the WA-LRU score with the
//     TTL state to pick the next victim in one launch (avoiding the
//     two-pass scoring + scan that the host-side fallback uses).
//
// All kernels target sm_80 (A100) and degrade gracefully to sm_70 (V100).
// fp16 KV cache, GQA-safe layout consistent with the rest of csrc/cuda.

#include "saga_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <float.h>

namespace cg = cooperative_groups;

namespace saga {
namespace cuda {

namespace {

constexpr int kThreadsTTL    = 256;
constexpr int kThreadsCompact = 256;
constexpr int kThreadsPick   = 256;

// Per-block bookkeeping used by all three kernels in this file. Keep the
// fields cache-line aligned so warp loads coalesce; the struct is sized to
// exactly 32 bytes which fits two per 64-byte line.
//
// NOTE: ``session_id`` is a hashed session identifier (we don't carry the
// string into device memory). The host side maintains the inverse mapping.

struct __align__(32) BlockMeta {
    std::uint64_t session_hash;
    std::int64_t  ttl_deadline_ms;    // INT64_MAX == "no TTL"
    std::int32_t  last_access_ms_lo;  // ms since epoch, low 32 bits
    std::uint8_t  pinned;
    std::uint8_t  is_free;
    std::uint16_t _pad;
};
static_assert(sizeof(BlockMeta) == 32, "BlockMeta must fit two per cache line");

// ============================================================ ttl_decay
//
// Pure data parallel: each thread inspects one block, writes a 1 if it
// should be evicted (TTL expired AND not pinned), 0 otherwise. Output is a
// flat byte mask so a downstream prefix-sum kernel can compact the free
// list without a second pass over BlockMeta.

__global__ void ttl_decay_kernel(
    const BlockMeta* __restrict__ blocks,
    std::int64_t now_ms,
    std::int32_t n_blocks,
    std::uint8_t* __restrict__ out_expired_mask,
    std::int32_t* __restrict__ out_n_expired)
{
    const std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;

    const BlockMeta m = blocks[idx];
    const bool already_free = m.is_free != 0;
    const bool pinned       = m.pinned != 0;
    const bool ttl_expired  = (m.ttl_deadline_ms != INT64_MAX) &&
                              (m.ttl_deadline_ms <= now_ms);

    const std::uint8_t mark = (!already_free && !pinned && ttl_expired) ? 1u : 0u;
    out_expired_mask[idx] = mark;

    // Tally expired count via warp-aggregated atomics so we don't trip on
    // the global atomic when the mask is dense (every-block-expired regime
    // is exercised by the pressure tests in test_paper_fidelity.py).
    using warp_t = cg::thread_block_tile<32>;
    warp_t warp = cg::tiled_partition<32>(cg::this_thread_block());
    const unsigned mask = __ballot_sync(0xffffffffu, mark);
    if (warp.thread_rank() == 0 && mask != 0u) {
        atomicAdd(out_n_expired, __popc(mask));
    }
}

// ============================================================ compact
//
// Convert the per-block expired bitmask into a compacted free list usable
// by BlockSpaceManagerV2.allocate. We do an exclusive prefix sum of the
// mask across the grid so each thread knows where to write its block id;
// for cluster sizes under ~16K blocks per worker the single-block
// cooperative scan is plenty.

__global__ void free_list_compact_kernel(
    const std::uint8_t* __restrict__ expired_mask,
    std::int32_t n_blocks,
    std::int32_t* __restrict__ out_free_list,
    std::int32_t* __restrict__ out_n_free)
{
    extern __shared__ std::int32_t scratch[];
    cg::thread_block tile = cg::this_thread_block();
    const std::int32_t tid = threadIdx.x;
    const std::int32_t stride = blockDim.x;

    // Cooperative single-block scan: each thread handles ceil(n / stride)
    // mask entries, accumulating into a shared running offset.
    std::int32_t local_count = 0;
    for (std::int32_t i = tid; i < n_blocks; i += stride) {
        local_count += expired_mask[i] ? 1 : 0;
    }
    scratch[tid] = local_count;
    tile.sync();

    // In-place inclusive scan over ``stride`` entries (Kogge-Stone).
    for (std::int32_t off = 1; off < stride; off <<= 1) {
        const std::int32_t v = (tid >= off) ? scratch[tid - off] : 0;
        tile.sync();
        scratch[tid] += v;
        tile.sync();
    }
    const std::int32_t my_inclusive = scratch[tid];
    const std::int32_t my_exclusive = my_inclusive - local_count;

    // Second pass: emit the block ids.
    std::int32_t write_off = my_exclusive;
    for (std::int32_t i = tid; i < n_blocks; i += stride) {
        if (expired_mask[i]) {
            out_free_list[write_off++] = i;
        }
    }

    if (tid == 0) {
        *out_n_free = scratch[stride - 1];
    }
}

// ============================================================ walru_pick
//
// One-launch combined eviction picker:
//
//   1.  Read BlockMeta + the precomputed (recency, preuse, size_norm)
//       triples for each candidate.
//   2.  Compute score = alpha * recency + beta * (1 - preuse) + gamma * size.
//   3.  Pinned blocks contribute +inf. Expired blocks contribute -inf so
//       they win ties with younger non-expired blocks (TTL takes
//       precedence per Algorithm 1).
//   4.  Block reduce the (score, idx) pair and emit argmin to
//       ``out_argmin``.

__global__ void paged_attention_walru_pick_kernel(
    const BlockMeta* __restrict__ blocks,
    const float* __restrict__ recency,
    const float* __restrict__ preuse,
    const float* __restrict__ size_norm,
    std::int64_t now_ms,
    std::int32_t n_blocks,
    float alpha,
    float beta,
    float gamma,
    std::int32_t* __restrict__ out_argmin,
    float* __restrict__ out_min_score)
{
    cg::thread_block tile = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tile);

    const std::int32_t tid = threadIdx.x;
    const std::int32_t stride = blockDim.x;

    float best_score = FLT_MAX;
    std::int32_t best_idx = -1;

    for (std::int32_t i = tid; i < n_blocks; i += stride) {
        const BlockMeta m = blocks[i];
        if (m.is_free) continue;

        float local_score;
        if (m.pinned) {
            local_score = FLT_MAX;
        } else {
            const bool ttl_expired = (m.ttl_deadline_ms != INT64_MAX) &&
                                     (m.ttl_deadline_ms <= now_ms);
            if (ttl_expired) {
                local_score = -FLT_MAX;
            } else {
                const float r = recency[i];
                const float p = preuse[i];
                const float s = size_norm[i];
                const float p_evict = alpha * r + beta * (1.0f - p) + gamma * s;
                local_score = -p_evict;
            }
        }
        if (local_score < best_score) {
            best_score = local_score;
            best_idx = i;
        }
    }

    // Warp-level reduction first (avoids the shared-memory dance for the
    // common case of n_blocks ~= warpSize), then a single block-level
    // reduction over the per-warp partial minima.
    for (int off = 16; off > 0; off >>= 1) {
        float other_score = __shfl_down_sync(0xffffffffu, best_score, off);
        std::int32_t other_idx = __shfl_down_sync(0xffffffffu, best_idx, off);
        if (other_score < best_score) {
            best_score = other_score;
            best_idx   = other_idx;
        }
    }

    __shared__ float    warp_scores[32];
    __shared__ std::int32_t warp_indices[32];
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;

    if (lane_id == 0) {
        warp_scores[warp_id]  = best_score;
        warp_indices[warp_id] = best_idx;
    }
    tile.sync();

    if (warp_id == 0) {
        const int n_warps = (stride + 31) / 32;
        best_score = (lane_id < n_warps) ? warp_scores[lane_id] : FLT_MAX;
        best_idx   = (lane_id < n_warps) ? warp_indices[lane_id] : -1;
        for (int off = 16; off > 0; off >>= 1) {
            float other_score = __shfl_down_sync(0xffffffffu, best_score, off);
            std::int32_t other_idx = __shfl_down_sync(0xffffffffu, best_idx, off);
            if (other_score < best_score) {
                best_score = other_score;
                best_idx   = other_idx;
            }
        }
        if (lane_id == 0) {
            *out_argmin    = best_idx;
            *out_min_score = best_score;
        }
    }
}

}  // anonymous namespace

// ============================================================ launchers

std::int32_t ttl_decay_launch(
    const BlockMeta* blocks,
    std::int64_t now_ms,
    std::int32_t n_blocks,
    std::uint8_t* out_expired_mask,
    std::int32_t* out_n_expired,
    void* stream)
{
    if (n_blocks <= 0) return 0;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    cudaMemsetAsync(out_n_expired, 0, sizeof(std::int32_t), s);

    const int blocks_per_grid = (n_blocks + kThreadsTTL - 1) / kThreadsTTL;
    ttl_decay_kernel<<<blocks_per_grid, kThreadsTTL, 0, s>>>(
        blocks, now_ms, n_blocks, out_expired_mask, out_n_expired);
    return n_blocks;
}

std::int32_t free_list_compact_launch(
    const std::uint8_t* expired_mask,
    std::int32_t n_blocks,
    std::int32_t* out_free_list,
    std::int32_t* out_n_free,
    void* stream)
{
    if (n_blocks <= 0) {
        std::int32_t zero = 0;
        cudaMemcpyAsync(out_n_free, &zero, sizeof(std::int32_t),
                        cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream));
        return 0;
    }
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int shared_bytes = kThreadsCompact * sizeof(std::int32_t);
    free_list_compact_kernel<<<1, kThreadsCompact, shared_bytes, s>>>(
        expired_mask, n_blocks, out_free_list, out_n_free);
    return n_blocks;
}

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
    void* stream)
{
    if (n_blocks <= 0) {
        const std::int32_t neg = -1;
        const float pos_inf = FLT_MAX;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        cudaMemcpyAsync(out_argmin, &neg, sizeof(neg),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(out_min_score, &pos_inf, sizeof(pos_inf),
                        cudaMemcpyHostToDevice, s);
        return 0;
    }
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    paged_attention_walru_pick_kernel<<<1, kThreadsPick, 0, s>>>(
        blocks, recency, preuse, size_norm,
        now_ms, n_blocks,
        alpha, beta, gamma,
        out_argmin, out_min_score);
    return n_blocks;
}

}  // namespace cuda
}  // namespace saga
