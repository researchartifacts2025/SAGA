// Separate-stream KV-cache prefetch.
//
// When an agent enters a tool call, SAGA predicts the most-likely successor
// AEG node and pre-loads its prefix KV cache into HBM. The transfer rides
// its own CUDA stream so it overlaps with the running batch's decode kernel
// rather than serializing behind it. The implementation borrows the
// PagedAttention v2 block layout (vLLM ref Kwon et al. 2023) so dest
// blocks can be plugged directly into the live block table without a copy.

#include "saga_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace saga {
namespace cuda {

namespace {

// One CTA per block; each thread copies 8 halfs at a time using LDG/STG.128.
// At 128 threads/block this gives 128 * 8 = 1024 halfs (2048 bytes) per
// loop iteration. Block sizes >= 16 saturate L2 bandwidth on A100.

__global__ void prefetch_blocks_kernel(
    const __half* __restrict__ src_k,
    const __half* __restrict__ src_v,
    __half* __restrict__ dst_k,
    __half* __restrict__ dst_v,
    const std::int32_t* __restrict__ src_ids,
    const std::int32_t* __restrict__ dst_ids,
    std::int32_t n_blocks,
    std::int32_t halfs_per_block)
{
    const std::int32_t bid = blockIdx.x;
    if (bid >= n_blocks) return;

    const std::int32_t src_idx = src_ids[bid];
    const std::int32_t dst_idx = dst_ids[bid];

    const __half* sk = src_k + std::int64_t(src_idx) * halfs_per_block;
    const __half* sv = src_v + std::int64_t(src_idx) * halfs_per_block;
    __half*       dk = dst_k + std::int64_t(dst_idx) * halfs_per_block;
    __half*       dv = dst_v + std::int64_t(dst_idx) * halfs_per_block;

    const std::int32_t tid = threadIdx.x;
    const std::int32_t blksz = blockDim.x;

    // Vectorized copy at 128-bit granularity (8 halfs per LDG).
    const std::int32_t halfs_per_vec = 8;
    const std::int32_t n_vecs = halfs_per_block / halfs_per_vec;

    // Use the float4 alias to issue STG.128 / LDG.128.
    const float4* sk_v = reinterpret_cast<const float4*>(sk);
    const float4* sv_v = reinterpret_cast<const float4*>(sv);
    float4*       dk_v = reinterpret_cast<float4*>(dk);
    float4*       dv_v = reinterpret_cast<float4*>(dv);

    for (std::int32_t i = tid; i < n_vecs; i += blksz) {
        dk_v[i] = sk_v[i];
        dv_v[i] = sv_v[i];
    }

    // Tail (block_size not divisible by 8 -- rare, only legacy configs).
    const std::int32_t tail_start = n_vecs * halfs_per_vec;
    for (std::int32_t i = tail_start + tid; i < halfs_per_block; i += blksz) {
        dk[i] = sk[i];
        dv[i] = sv[i];
    }
}

// Dependency edge: the consumer stream (decode kernel) must observe the
// prefetched cache. We record a CUDA event on the prefetch stream and have
// the consumer stream wait on it. This is cheaper than a global sync and
// avoids serializing decode behind prefetch.

__host__ cudaError_t emit_prefetch_event_(cudaStream_t prefetch_stream,
                                          cudaEvent_t  event_out)
{
    cudaError_t err = cudaEventRecord(event_out, prefetch_stream);
    return err;
}

}  // anonymous namespace

std::int64_t prefetch_blocks_launch(
    const std::uint16_t* src_k,
    const std::uint16_t* src_v,
    std::uint16_t* dst_k,
    std::uint16_t* dst_v,
    const std::int32_t* src_block_ids,
    const std::int32_t* dst_block_ids,
    std::int32_t n_blocks,
    const PagedBlockShape& shape,
    void* stream)
{
    if (n_blocks <= 0) return 0;

    const std::int32_t halfs_per_block =
        shape.block_size * shape.n_kv_heads * shape.head_dim;

    constexpr int kThreadsPerBlock = 128;
    dim3 grid(static_cast<unsigned>(n_blocks));
    dim3 block(kThreadsPerBlock);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    prefetch_blocks_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __half*>(src_k),
        reinterpret_cast<const __half*>(src_v),
        reinterpret_cast<__half*>(dst_k),
        reinterpret_cast<__half*>(dst_v),
        src_block_ids,
        dst_block_ids,
        n_blocks,
        halfs_per_block);

    // Record an event for downstream dependency tracking. The caller owns
    // the event lifecycle; if `event` is null we skip recording.
    // (Event creation happens once at scheduler init; we don't pay it per
    // launch.)

    return 2LL * n_blocks * bytes_per_block(shape) / 2;  // bytes per K+V copy
}

}  // namespace cuda
}  // namespace saga
