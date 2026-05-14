// WA-LRU scoring on GPU.
//
// When the resident block pool is large (paper reports up to 12K resident
// blocks per worker on 80GB HBM), CPU-side scoring becomes the
// scheduler-loop bottleneck. This kernel computes the WA-LRU formula
//
//   P_evict = alpha * recency + beta * (1 - preuse) + gamma * size_norm
//   score   = -P_evict
//
// over the entire pool in a single grid launch, then uses
// cooperative_groups warp reduction to emit the argmin.

#include "saga_cuda.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float.h>

namespace cg = cooperative_groups;

namespace saga {
namespace cuda {

namespace {

constexpr int kThreads = 256;

// Block-reduce argmin via warp-level cooperative groups.

struct ScoreIdx {
    float score;
    std::int32_t idx;
};

__device__ ScoreIdx argmin_pair(const ScoreIdx& a, const ScoreIdx& b) {
    return (a.score < b.score) ? a : b;
}

__global__ void walru_score_kernel(
    const float* __restrict__ recency,
    const float* __restrict__ preuse,
    const float* __restrict__ size_norm,
    const std::uint8_t* __restrict__ pinned,
    std::int32_t n,
    float alpha, float beta, float gamma,
    std::int32_t* __restrict__ block_argmin,    // [grid.x]
    float* __restrict__ block_min_score)        // [grid.x]
{
    const std::int32_t tid = threadIdx.x;
    const std::int32_t gid = blockIdx.x * blockDim.x + tid;
    const std::int32_t stride = gridDim.x * blockDim.x;

    ScoreIdx local{FLT_MAX, -1};

    for (std::int32_t i = gid; i < n; i += stride) {
        if (pinned[i]) continue;
        const float r = recency[i];
        const float u = preuse[i];
        const float sz = size_norm[i];
        const float p_evict = alpha * r + beta * (1.f - u) + gamma * sz;
        const float score = -p_evict;
        if (score < local.score) {
            local.score = score;
            local.idx = i;
        }
    }

    // Block-level reduction in shared memory.
    __shared__ ScoreIdx shm[kThreads];
    shm[tid] = local;
    __syncthreads();

    // Tree reduction.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm[tid] = argmin_pair(shm[tid], shm[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min_score[blockIdx.x] = shm[0].score;
        block_argmin[blockIdx.x]    = shm[0].idx;
    }
}

__global__ void walru_finalize_kernel(
    const float* __restrict__ block_min_score,
    const std::int32_t* __restrict__ block_argmin,
    std::int32_t n_blocks,
    std::int32_t* __restrict__ out_argmin,
    float* __restrict__ out_min_score)
{
    if (threadIdx.x != 0) return;
    float best = FLT_MAX;
    std::int32_t best_idx = -1;
    for (std::int32_t i = 0; i < n_blocks; ++i) {
        const float s = block_min_score[i];
        if (s < best) {
            best = s;
            best_idx = block_argmin[i];
        }
    }
    *out_min_score = best;
    *out_argmin = best_idx;
}

}  // anonymous namespace

void walru_score_launch(
    const float* recency,
    const float* preuse,
    const float* size_norm,
    const std::uint8_t* pinned,
    std::int32_t n,
    float alpha, float beta, float gamma,
    std::int32_t* out_argmin,
    float* out_min_score,
    void* stream)
{
    if (n <= 0) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        std::int32_t neg = -1;
        float pos_inf = FLT_MAX;
        cudaMemcpyAsync(out_argmin, &neg, sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(out_min_score, &pos_inf, sizeof(float),
                        cudaMemcpyHostToDevice, s);
        return;
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    // Aim for ~64 KB of working set per SM (~6 blocks * 256 threads * 16B).
    const int n_blocks = (n + kThreads - 1) / kThreads;
    const int grid_blocks = (n_blocks > 1024) ? 1024 : n_blocks;

    // Reduction-temporary buffer. In production this is a per-worker
    // persistent allocation (lives in the cache-manager handle); here we
    // allocate per-launch which is fine for a reference implementation.
    std::int32_t* tmp_argmin = nullptr;
    float* tmp_min = nullptr;
    cudaMallocAsync(&tmp_argmin, sizeof(std::int32_t) * grid_blocks, s);
    cudaMallocAsync(&tmp_min, sizeof(float) * grid_blocks, s);

    walru_score_kernel<<<grid_blocks, kThreads, 0, s>>>(
        recency, preuse, size_norm, pinned, n,
        alpha, beta, gamma,
        tmp_argmin, tmp_min);

    walru_finalize_kernel<<<1, 1, 0, s>>>(
        tmp_min, tmp_argmin, grid_blocks,
        out_argmin, out_min_score);

    cudaFreeAsync(tmp_argmin, s);
    cudaFreeAsync(tmp_min, s);
}

}  // namespace cuda
}  // namespace saga
