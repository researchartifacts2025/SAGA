// GPU-side prefix overlap detection.
//
// For large successor sets (paper §4.2 reports up to 256 candidates per
// node on tree-of-thought agents), host-side overlap computation becomes
// the eviction-decision bottleneck. This kernel runs a parallel LCP-style
// scan over the cached token sequence and each candidate's predicted
// prefix, returning per-candidate overlap counts that the WA-LRU
// `predict_reuse` step then consumes.

#include "saga_cuda.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace saga {
namespace cuda {

namespace {

// Single-pair overlap. Cooperative-group reduction over the warp.

__global__ void prefix_overlap_kernel(
    const std::int32_t* __restrict__ tokens_a,
    const std::int32_t* __restrict__ tokens_b,
    std::int32_t n_a,
    std::int32_t n_b,
    std::int32_t* __restrict__ out_overlap)
{
    const std::int32_t n_min = (n_a < n_b) ? n_a : n_b;
    const std::int32_t tid = threadIdx.x;
    const std::int32_t blksz = blockDim.x;

    // Each thread finds the local first-mismatch position; warp reduce
    // gives the global LCP.

    __shared__ std::int32_t s_first_mismatch;
    if (tid == 0) s_first_mismatch = n_min;
    __syncthreads();

    for (std::int32_t i = tid; i < n_min; i += blksz) {
        if (tokens_a[i] != tokens_b[i]) {
            atomicMin(&s_first_mismatch, i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        *out_overlap = s_first_mismatch;
    }
}

// Batched: one CTA per (cached, successor) pair.

__global__ void prefix_overlap_batch_kernel(
    const std::int32_t* __restrict__ tokens_cached,
    std::int32_t n_cached,
    const std::int32_t* __restrict__ tokens_succ_flat,
    const std::int32_t* __restrict__ succ_offsets,
    std::int32_t n_succ,
    std::int32_t* __restrict__ out_overlap)
{
    const std::int32_t s = blockIdx.x;
    if (s >= n_succ) return;

    const std::int32_t lo = succ_offsets[s];
    const std::int32_t hi = succ_offsets[s + 1];
    const std::int32_t n_b = hi - lo;
    const std::int32_t n_min = (n_cached < n_b) ? n_cached : n_b;

    const std::int32_t tid = threadIdx.x;
    const std::int32_t blksz = blockDim.x;
    const std::int32_t* tb = tokens_succ_flat + lo;

    __shared__ std::int32_t s_first_mismatch;
    if (tid == 0) s_first_mismatch = n_min;
    __syncthreads();

    for (std::int32_t i = tid; i < n_min; i += blksz) {
        if (tokens_cached[i] != tb[i]) {
            atomicMin(&s_first_mismatch, i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        out_overlap[s] = s_first_mismatch;
    }
}

}  // anonymous namespace

void prefix_overlap_launch(
    const std::int32_t* tokens_a,
    const std::int32_t* tokens_b,
    std::int32_t n_a,
    std::int32_t n_b,
    std::int32_t* out_overlap,
    void* stream)
{
    if (n_a <= 0 || n_b <= 0) {
        cudaMemsetAsync(out_overlap, 0, sizeof(std::int32_t),
                        reinterpret_cast<cudaStream_t>(stream));
        return;
    }
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    prefix_overlap_kernel<<<1, 256, 0, s>>>(
        tokens_a, tokens_b, n_a, n_b, out_overlap);
}

void prefix_overlap_batch_launch(
    const std::int32_t* tokens_cached,
    std::int32_t n_cached,
    const std::int32_t* tokens_succ_flat,
    const std::int32_t* succ_offsets,
    std::int32_t n_succ,
    std::int32_t* out_overlap,
    void* stream)
{
    if (n_succ <= 0) return;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    prefix_overlap_batch_kernel<<<n_succ, 256, 0, s>>>(
        tokens_cached, n_cached,
        tokens_succ_flat, succ_offsets, n_succ,
        out_overlap);
}

}  // namespace cuda
}  // namespace saga
