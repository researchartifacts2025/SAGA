# Experimental Details

This document covers the production hyperparameters, the cluster geometry,
the per-tool latency calibration, and the policy-validation cost model
used by the CPU-side regression tests.

## Hardware (paper measurements)

| Item             | Value                                       |
|------------------|---------------------------------------------|
| Nodes            | 8                                           |
| GPUs per node    | 8 × NVIDIA A100-80GB (HBM2e, 2 TB/s)        |
| Total GPUs       | 64                                          |
| Inter-node       | 200 Gbps InfiniBand HDR with GPUDirect RDMA |
| CPUs per node    | 2 × AMD EPYC 7763 (128 cores total)         |
| RAM per node     | 1 TB DDR4-3200                              |
| Storage per node | 4 × 3.84 TB NVMe SSDs                       |

## Software

| Item            | Version           |
|-----------------|-------------------|
| OS              | Ubuntu 22.04      |
| CUDA            | 12.1.1 (driver 530.30.02) |
| Python          | 3.10.12           |
| PyTorch         | 2.1.2 + cu121     |
| vLLM            | 0.6.0 (V1 engine) |
| FlashAttention  | 2.5.6             |
| Ray             | 2.9.0             |
| Model           | Llama-3-70B-Instruct, TP=4 |

## Workloads

| Workload          | Size                       | Mean steps | Token shape          |
|-------------------|----------------------------|-----------:|----------------------|
| SWE-bench         | 500 verified tasks         |         37 | 2-4K prompt, 100-500 out |
| WebArena          | 812 browser tasks          |         18 | 4-8K prompt, 50-200 out  |
| BurstGPT-derived  | 10-tenant synthetic, 80 % offered load | varies (10/30/100) | tenant-class-dependent |

For the multi-tenant workload, tenants are partitioned as:

* 3 heavy   tenants: 100-step agents at ≈16 tasks/min/tenant
* 4 medium  tenants:  30-step agents at  ≈8 tasks/min/tenant
* 3 light   tenants:  10-step agents at  ≈4 tasks/min/tenant

## Per-Tool Latency Distributions

Calibration values per tool category (P50 / P95 / P99 in ms):

| Tool category    | P50 | P95   | P99    |
|------------------|----:|------:|-------:|
| Code execution   | 180 | 2 400 | 28 000 |
| File operations  |  45 |   320 |  1 200 |
| Web / API        | 850 | 4 500 | 45 000 |
| Database queries | 120 |   890 |  3 500 |

The TTL estimator fits a log-normal to the running history and uses the
configured percentile (default 95 %).

## SAGA Hyperparameters

| Parameter             | Default  | Source                       |
|-----------------------|---------:|------------------------------|
| α (recency weight)    | 0.3      | sensitivity analysis         |
| β (reuse weight)      | 0.5      | sensitivity analysis         |
| γ (size weight)       | 0.2      | size as tiebreaker           |
| θ (routing threshold) | 0.8      | 20 % headroom                |
| `threshold_low`       | 0.7      | soft-pressure onset          |
| `threshold_high`      | 0.9      | hard-eviction limit          |
| `T_idle` (steal)      | 100 ms   | amortize steal cost          |
| `R_max` (load ratio)  | 2.0      | imbalance tolerance          |
| `TTL_max`             | 300 s    | P99 tool latency cap         |
| `θ_conf` (AEG)        | 0.7      | precision-recall tradeoff    |
| coordinator epoch     | 100 ms   | end of paper §3              |
| AFS preempt threshold | 500 ms   | block-time bound             |
| Migration mean / P95  | 230 / 890 ms | overhead table           |

## Policy-Validation Cost Model

The wall-clock numbers in Tables 3-10 come from real Llama-3-70B inference
on 64 A100s (see [`src/saga/serving/benchmarks/`](../src/saga/serving/benchmarks/)).
The CPU-side policy-validation harness in [`saga.sim`](../src/saga/sim/) uses
a coarse cost model so the algorithm unit tests (98 tests, no GPUs) run in
seconds. Per-step duration is charged as

```
prefill_ms     = max(new_prompt_tokens, 1) / prefill_tokens_per_ms
decode_ms      = max(output_tokens, 1)    / decode_tokens_per_ms
miss_stall_ms  = cache_miss_stall_ms       if the admit was a miss else 0
duration       = prefill_ms + decode_ms + miss_stall_ms
```

with `prefill_tokens_per_ms = 850`, `decode_tokens_per_ms = 38` (calibrated
to Llama-3-70B on TP=4 A100-80GB), and `cache_miss_stall_ms = 280`
(per-miss batch backpressure: when a fresh prefill enters a running batch
its prefix-stage stalls other sessions' decodes).

On a cache hit, only the new observation tokens are prefilled and no stall
is charged. On a miss, the full current context is prefilled *and* the
batch stall is paid. Tool durations come from the per-workload ground-truth
`ToolPlan` produced by the workload generator.

### CPU-DRAM Offload (§5.4)

When `dram_tier_enabled = True`, evictions from HBM go to a per-worker DRAM
pool sized by `dram_capacity_tokens_per_worker` (default 4M tokens, ~320 GB
at ~80 KB/token for Llama-3-70B GQA with TP=4). PCIe transfer cost is

```
transfer_ms = bytes_per_token * n_tokens / sustained_bandwidth_bytes_per_ms
```

`sustained_bandwidth_bytes_per_ms = 25 * 10^6` (Gen4 ×16); halved to
12.5 GB/s when `dram_contention = True` (multi-tenant PCIe sharing,
paper §5.4).

## Calibration Band (CPU validation harness)

The policy-validation harness in `saga.sim` exercises the same
`WALRUPolicy`, `ToolTTLPolicy`, `SessionRouter`, `AFSScheduler`, and
`AgentExecutionGraph` objects that drive the live cluster, against a
coarse cost model so unit tests stay CPU-only. The harness is faithful to:

* relative cache-hit-rate ordering of policies (LRU < LRU+Prefix < WA-LRU),
* relative regeneration-cost ordering (more re-prefilling for cache-blind
  routers),
* qualitative effects of each ablation (session affinity is the dominant
  contributor, exactly as in the paper).

It does *not* attempt to match the absolute wall-clock TCTs reported in
Tables 3-10: those come from the real 64-A100 cluster runs orchestrated
by [`saga.serving.benchmarks.WallClockBenchmark`](../src/saga/serving/benchmarks/runner.py).
The CPU harness (a) serializes one session per worker step rather than
batching ~32 concurrent requests with PagedAttention, and (b) samples
tool durations from the log-normal fit instead of replaying a captured
trace. Its purpose is regression coverage of the policy code, not
performance prediction.

## Statistics

Tables show mean ± standard deviation over the configured seed set
(default 3 seeds: 42, 123, 456). Pairwise comparisons use two-sided
Welch's t-test for unequal variances. Significance is reported with the
usual stars: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001.
