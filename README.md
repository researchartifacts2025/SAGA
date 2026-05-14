<div align="center">

# SAGA

### Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![OpenMP](https://img.shields.io/badge/OpenMP-enabled-success.svg)](https://www.openmp.org/)
[![pybind11](https://img.shields.io/badge/pybind11-3.x-blue.svg)](https://github.com/pybind/pybind11)
[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen.svg)](#-testing)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/types-mypy-2A6DB2.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-research--artifact-blue.svg)](#)

**A program-level scheduler that treats agent workflows --- not individual
LLM calls --- as the first-class schedulable unit. Within 1.31× of Bélády's
optimal offline policy on production agent traces.**

[Quick Start](#-quick-start) · [Results](#-results) · [Design](#-design) ·
[HPC Acceleration](#-hpc-acceleration) ·
[Running the Experiments](#-running-the-experiments) · [Architecture](#-architecture)

</div>

---

## 📌 Why?

AI agents (SWE-bench coding agents, WebArena browser agents, AutoGen tool-use
loops) execute **tens to hundreds of chained LLM calls** per task with
gigabytes of intermediate KV-cache state between steps. GPU schedulers built
for one-shot inference discard that state on every tool-call boundary,
inflating end-to-end latency by **3-8×**. Tool calls (the 50 ms `read_file`
or the 30 s `pytest`) are the variable-duration idle gaps where the cache is
at risk.

SAGA shifts the schedulable unit from *request* to *program*. Six mechanisms:

| Mechanism | What it does | Where in the code |
|---|---|---|
| **Agent Execution Graphs (AEGs)** | Capture workflow structure (ReAct chains, tree-of-thought branches) so the scheduler can predict KV-cache reuse across tool-call boundaries. | [`saga.core.aeg`](src/saga/core/aeg.py) |
| **WA-LRU + Tool-Call-Aware TTL** | Workflow-aware eviction `P_evict = α·R + β·(1 − P_reuse) + γ·S` plus a per-tool-type log-normal TTL with memory-pressure scaling. | [`saga.cache.policies`](src/saga/cache/policies.py), [`saga.cache.ttl`](src/saga/cache/ttl.py) |
| **Session-Affinity Routing + Work Stealing** | Keep correlated requests on the same worker; randomized stealing for tail balance. | [`saga.scheduler.routing`](src/saga/scheduler/routing.py), [`saga.scheduler.stealing`](src/saga/scheduler/stealing.py) |
| **Agent Fair Share (AFS)** | Task-completion-time fairness via urgency-proportional allocation, with a Lyapunov-drift completion-time bound. | [`saga.fairness.afs`](src/saga/fairness/afs.py) |
| **Speculative Prefetch** | Pin and pre-extend the most-likely successor's cache during the tool-call idle period. | [`saga.sim.engine`](src/saga/sim/engine.py) |
| **CPU-DRAM Offload Tier** | Two-tier cache: HBM evictions move to host DRAM via PCIe rather than discard. | [`saga.cache.dram_tier`](src/saga/cache/dram_tier.py) |

Plus two structural pieces:

| Piece | What it does | Where |
|---|---|---|
| **Pattern Inference** | Infer AEGs from observed request streams when framework hints are unavailable (cold-start guard, θ_conf=0.7 confidence). | [`saga.workflow.pattern`](src/saga/workflow/pattern.py) |
| **Framework Hint Parser** | Convert LangChain / AutoGen / CrewAI callback metadata into AEGs. | [`saga.workflow.analyzer`](src/saga/workflow/analyzer.py) |

---

## 🚀 Quick Start

### Install

```bash
git clone <your-fork-url> saga
cd saga
pip install -e .
```

Python 3.10–3.12 supported. Pure-Python install works on any platform; no GPU required.

### Optional: Compile the C++ Acceleration Layer

```bash
pip install pybind11
python setup_native.py build_ext --inplace
```

With the `_native` extension built, WA-LRU and Bélády eviction kernels run
in OpenMP-parallel C++. Verify:

```bash
python -c "from saga import is_native_available, native_build_info; print(native_build_info())"
# saga_native v1 (OpenMP, threads=20)
```

If the build is skipped, SAGA transparently uses the pure-Python fallback —
no API split, identical behaviour.

### One-Minute Demo

```bash
python -m saga.entrypoints.simulate experiment=demo
```

Expected output (truncated):

```
        Simulation: saga on swe_bench
┌────────────────────────┬─────────────┐
│ Metric                 │       Value │
├────────────────────────┼─────────────┤
│ Tasks completed        │     20 / 20 │
│ TCT mean (s)           │  17.8 ± 5.4 │
│ Memory utilization     │      22.4 % │
│ Cache hit rate         │      96.2 % │
│ Regen ratio            │       0.067 │
│ Steals / migrations    │       0 / 0 │
└────────────────────────┴─────────────┘
```

### Compare Schedulers

```bash
python -m saga.entrypoints.simulate scheduler=vllm_apc
python -m saga.entrypoints.simulate scheduler=saga
saga presets   # list all 13
```

### Available Presets

```
vllm                   vLLM v0.6.0 (V1 engine), LRU + FCFS
vllm_apc               vLLM v0.15.1 with Automatic Prefix Caching
sglang                 SGLang v0.5.8 with RadixAttention
llumnix                vLLM + live KV-cache migration
trt_llm_scaffolding    TensorRT-LLM v1.1 + Scaffolding
vllm_kvflow            vLLM + KVFlow workflow-aware eviction
saga                   SAGA (this paper)
saga_no_walru          SAGA w/o workflow-aware eviction        (ablation)
saga_no_ttl            SAGA w/o tool-call-aware TTL            (ablation)
saga_no_prefetch       SAGA w/o speculative prefetch           (ablation)
saga_no_affinity       SAGA w/o session affinity               (ablation)
saga_no_stealing       SAGA w/o work stealing                  (ablation)
saga_no_afs            SAGA w/o AFS fairness                   (ablation)
```

---

## ⚡ HPC Acceleration

SAGA ships with an optional C++17 native module compiled via pybind11. It
implements the **hot WA-LRU and Bélády eviction kernels** with OpenMP
parallel reduction over the resident cache pool, plus a sharded **lock-free
session table** that backs the global coordinator's affinity map.

Measured speedups (Windows 11, MSVC 2019, AMD Ryzen, OpenMP T=20):

| Kernel | N=64 | N=256 | N=1024 | N=4096 | N=16384 |
|---|---:|---:|---:|---:|---:|
| WA-LRU `select_victim`  |   16× |   14× |   80× |  669× | **1070×** |
| Bélády oracle lookup    |   13× |   39× |   62× |   88× |   82× |
| `predict_reuse_batch`   |    3× |    3× |    6× |    7× |    5× |
| Session table           | Python dict (GIL)         | 64-shard `std::mutex` (concurrent writers) ||||

Reproduce with:

```bash
make bench-native      # or python -m saga.entrypoints.bench_native
```

The kernel signatures accept flat NumPy arrays (zero-copy via pybind11's
buffer protocol), so no per-entry marshalling overhead is paid on the hot
path. The session table is a 64-shard hash map (`std::mutex` per shard)
backing the global coordinator's affinity map.

### Build flags

| Flag | Default | Effect |
|---|---|---|
| `SAGA_ENABLE_OPENMP` | ON | Compile OpenMP parallel reduction. |
| `SAGA_NATIVE_TUNE`   | OFF | `-O3 -march=native` (or `/O2` on MSVC). |

```bash
# Tuned build via CMake
cmake -S . -B build -DSAGA_NATIVE_TUNE=ON
cmake --build build --config Release -j

# Or via setuptools shim (no CMake)
pip install pybind11
python setup_native.py build_ext --inplace
```

### Runtime detection

```python
from saga import is_native_available, native_build_info

if is_native_available():
    print(native_build_info())   # 'saga_native v1 (OpenMP, threads=20)'
else:
    print("Using pure-Python fallback")
```

---

## 🏗️ Architecture

```
                                   ┌─────────────────────────────┐
   agent request ─────────────▶    │   Agent Interface Layer     │
   (LangChain / AutoGen /          │   • framework hint parser   │
    CrewAI / raw HTTP)             │   • pattern inference (θ=0.7)│
                                   └────────────┬────────────────┘
                                                │ AEG
                                                ▼
                                   ┌─────────────────────────────┐
                                   │   Global Coordinator        │
                                   │   • SessionRouter           │ ◀──── AFS Engine
                                   │   • WorkStealer             │       (Lyapunov drift,
                                   │   • Queue strategy (BFS/    │        urgency scoring)
                                   │     DFS/Hybrid)             │
                                   │   • Lock-free SessionTable* │
                                   └────────────┬────────────────┘
                                                │ session
                                                ▼
       worker 0           worker 1          worker N-1
   ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
   │ CacheManager │  │ CacheManager │   │ CacheManager │
   │ + WA-LRU*    │  │ + WA-LRU*    │   │ + WA-LRU*    │
   │ + Tool TTL   │  │ + Tool TTL   │   │ + Tool TTL   │
   │ + Spec.      │  │ + Spec.      │   │ + Spec.      │
   │   prefetch   │  │   prefetch   │   │   prefetch   │
   │ + DRAM tier  │  │ + DRAM tier  │   │ + DRAM tier  │
   └──────────────┘  └──────────────┘   └──────────────┘
                                                │ overflow
                                                ▼
                                   ┌─────────────────────────────┐
                                   │   CPU-DRAM offload tier     │
                                   │   (per-worker, PCIe Gen4)   │
                                   └─────────────────────────────┘
   * = C++/OpenMP-accelerated hot path
```

The whole thing runs as a deterministic discrete-event simulator on a single
machine in seconds. The same Python objects (`CacheManager`,
`SessionRouter`, `AFSScheduler`, `TieredCacheManager`) drop into a real
vLLM-extension build without modification.

---

## 🧠 Design

### Workflow-Aware LRU (WA-LRU)

The eviction score for a cached session `s` is

```
P_evict(s) = α · R̂(s)
           + β · (1 − P_reuse(s))
           + γ · Ŝ(s)
```

with normalized recency `R̂`, predicted reuse `P_reuse`, and size `Ŝ`.
Defaults: `α=0.3, β=0.5, γ=0.2` (β > α > γ ordering, robust to ±33 %
perturbation per [Parameter Sensitivity](#parameter-sensitivity)).

`P_reuse(s)` walks the AEG from the session's current node:

```
P_reuse(s) = Σ_{u ∈ succ(v_s)}  P(v_s → u) · overlap(s, u)
overlap(s, u) = cached_tokens / (cached_tokens + Ê[obs_tokens(u)])
```

### Tool-Call-Aware TTL

```
ttl_base   = percentile_p(log-normal fit of latency_history[tool])
pressure   = max(0, (used − low) / (high − low))    # low=0.7, high=0.9
ttl        = min(ttl_base · (1 − 0.5 · pressure), 300 s)
```

Per-tool defaults (calibrated to production traces, P50/P95/P99 in ms):

| Tool         | P50  | P95   | P99    |
|--------------|------|-------|--------|
| Code exec    |  180 | 2 400 | 28 000 |
| File ops     |   45 |   320 |  1 200 |
| Web / API    |  850 | 4 500 | 45 000 |
| DB query     |  120 |   890 |  3 500 |

### Session-Affinity Routing

```
route(r) =  w*_s             if load(w*_s) < θ and cached(w*_s, s)
        =  argmin_w load(w) otherwise
```

`θ = 0.8` (20 % headroom). Strategies: `session_affinity` (default),
`prefix_affinity` (vLLM-style), `least_loaded` (vanilla load balancer).

### Work Stealing

Trigger conditions (both checked every 100 ms epoch):

* a worker's queue has been empty for `T_idle = 100 ms`, OR
* load ratio max/min > `R_max = 2.0×`

Migration latency drawn from log-normal with mean 230 ms / P95 890 ms. Three
thrashing safeguards (load-ratio guard, post-migration affinity stickiness,
asynchronous source) keep the steal rate bounded.

### Agent Fair Share (AFS)

```
urgency_i(t) = (W_i − S_i(t)) / (deadline_i − t)
a_i(t) = urgency_i(t) / Σ_j urgency_j(t) · C
```

Lyapunov-drift analysis gives a high-probability completion-time bound:
`Pr[TCT_i ≤ (1+ε) E[TCT_i]] ≥ 1 − δ` with
`ε = O(ρ · √(log(N/δ)/n))`.

### Speculative Prefetch

When inference on node `v` ends and a tool call begins, the engine

1.  identifies `u = argmax_{u'} P(v → u')`,
2.  pre-extends the cache to the size needed by `u` (admitting and counting
    the prefill cost into the otherwise-idle gap), and
3.  pins the entry so eviction cannot rob the prefetched prefix during the
    tool call.

The pin is cleared on tool-end, returning the entry to normal WA-LRU
candidacy.

### CPU-DRAM Offload Tier (§5.4)

A second eviction tier in host DRAM, sized independently:

* HBM hit → cheap path.
* DRAM hit → swap-in (PCIe Gen4 ×16, ~25 GB/s sustained, halved under
  contention) + HBM admit.
* Miss → full re-prefill on next step.

Activated via `cluster.overrides.dram_tier_enabled=true`.

### Pattern Inference (§3.4)

When no framework hint is available, the inference engine

1.  buckets observed sessions by agent type,
2.  builds a tool-to-tool transition count matrix `C[a, b]`,
3.  normalizes to a probability matrix `P[a, b]`,
4.  keeps transitions with `P[a, b] ≥ θ_conf = 0.7` (paper's confidence
    threshold).

A fresh agent type is served as request-level until `cold_start_tasks = 30`
sessions complete. Paper reports 87 % accuracy and 12-18 % TCT degradation
versus explicit hints.

---

## 📊 Results

### Headline numbers (paper, 64× A100-80GB cluster)

| System | SWE-bench TCT (s) | WebArena TCT (s) | Speedup vs SAGA |
|---|---:|---:|---:|
| vLLM v0.6.0                 | 612.3 ± 32.1 | 178.4 ± 14.2 | 3.01× |
| vLLM v0.15.1 + APC          | 352.1 ± 21.4 | 127.3 ± 10.1 | 1.73× |
| SGLang v0.5.8               | 387.2 ± 24.3 | 138.7 ± 11.3 | 1.90× |
| Llumnix v1.2                | 498.1 ± 28.7 | 156.2 ± 12.8 | 2.45× |
| TRT-LLM + Scaffolding       | 324.6 ± 19.8 | 118.9 ±  9.4 | 1.60× |
| vLLM + KVFlow               | 298.4 ± 18.2 | 108.2 ±  8.7 | 1.47× |
| **SAGA**                    | **203.4 ± 12.8** | **82.1 ± 6.8** | — |

Geometric-mean speedup vs vLLM+APC: **1.64×** (`p < 0.001`, paired Welch's t-test).

### Competitive Ratio vs Bélády's Optimal

| Policy                 | SWE-bench | WebArena | Mean |
|------------------------|----------:|---------:|-----:|
| Standard LRU           | 2.84      | 2.12     | 2.48 |
| LRU + Prefix (vLLM)    | 1.97      | 1.74     | 1.86 |
| **WA-LRU (ours)**      | **1.31**  | **1.28** | **1.30** |

### Multi-Tenant SLO Attainment

| System  | Heavy | Medium | Light | Overall |
|---------|------:|-------:|------:|--------:|
| vLLM    | 89.4  | 72.1   | 43.2  | 67.3 |
| SGLang  | 91.2  | 78.6   | 51.4  | 73.4 |
| Llumnix | 92.8  | 81.3   | 58.9  | 77.2 |
| **SAGA**| **99.1** | **99.4** | **98.7** | **99.2** |

### Ablation (SWE-bench, % slowdown vs full SAGA)

| Configuration                | TCT (s) | vs Full |
|------------------------------|--------:|--------:|
| Full SAGA                    | 203.4   | —       |
| w/o session affinity         | 398.2   | **+96 %** |
| w/o workflow-aware eviction  | 312.8   | +54 %   |
| w/o tool-call TTL            | 289.1   | +42 %   |
| w/o work stealing            | 267.3   | +31 %   |
| w/o speculative prefetch     | 241.6   | +19 %   |
| w/o AFS fairness             | 218.7   | +8 %    |

### Execution Strategy Tradeoff (32 GPUs)

| Strategy | TCT (s) | Throughput | Evict Rate |
|----------|--------:|-----------:|-----------:|
| Pure BFS         | 487.2 ± 28.4 | 12.4 t/m | 78 % |
| Pure DFS         | 623.1 ± 34.2 |  4.2 t/m |  3 % |
| **Hybrid (SAGA)**| **203.4 ± 12.8** | 8.7 t/m | 12 % |

### Tool-Latency Variance Sensitivity

| CV  | TCT (s) | TTL Accuracy | Evict Rate |
|----:|--------:|-------------:|-----------:|
| 0.5 | 195.1 ± 11.2 | 96 % |  9 % |
| 1.0 | 203.4 ± 12.8 | 93 % | 12 % |
| 1.5 | 218.6 ± 15.3 | 88 % | 18 % |
| 2.0 | 241.3 ± 18.7 | 82 % | 24 % |
| 3.0 | 298.4 ± 24.1 | 71 % | 35 % |

### Parameter Sensitivity

| Parameter             | Default | Tested Range | Max ΔTCT |
|-----------------------|--------:|:------------|---------:|
| α (recency weight)    | 0.3     | [0.2, 0.4]  | < 5 %    |
| β (reuse weight)      | 0.5     | [0.4, 0.6]  | < 8 %    |
| γ (size weight)       | 0.2     | [0.1, 0.3]  | < 3 %    |
| θ (routing)           | 0.8     | [0.6, 0.95] | < 5 %    |
| `threshold_low`       | 0.7     | [0.6, 0.8]  | < 4 %    |
| `threshold_high`      | 0.9     | [0.85, 0.95]| < 6 %    |
| `T_idle` (steal)      | 100 ms  | [50, 200] ms| < 7 %    |
| `R_max` (load ratio)  | 2.0     | [1.5, 3.0]  | < 4 %    |
| `TTL_max`             | 300 s   | [120, 600] s| < 3 %    |
| `θ_conf` (AEG)        | 0.7     | [0.5, 0.9]  | < 6 %    |

---

## 🔁 Running the Experiments

The simulator implements every algorithm in the paper and emits each
result table.

```bash
# End-to-end TCT comparison across all systems
make tables             # → runs/<timestamp>/e2e.md

# Component ablation
make ablation           # → runs/<timestamp>/ablation.md

# Multi-tenant fairness (SLO by tenant class)
make fairness           # → runs/<timestamp>/fairness.md

# Competitive ratio vs Bélády's optimal offline policy
make competitive        # → runs/<timestamp>/competitive.md

# Parameter sensitivity sweeps
make sensitivity        # → runs/<timestamp>/sensitivity.md

# BFS / DFS / Hybrid execution strategy tradeoff
python -m saga.entrypoints.benchmark experiment=bfsdfs

# Tool-latency variance sweep (CV ∈ {0.5, 1.0, 1.5, 2.0, 3.0})
python -m saga.entrypoints.benchmark experiment=tool_variance
```

Each command runs 3 seeds × N presets on the cluster size configured in
[`configs/cluster/a100_64gpu.yaml`](configs/cluster/a100_64gpu.yaml); set
`cluster=single_node` for a CI-sized run.

---

## 🧪 Testing

```bash
make test            # 57 unit + integration tests
make typecheck       # mypy
make lint            # ruff (linter + formatter)
make check           # all three
```

The simulator is fully deterministic given a seed: workload generation, tool
durations, work-stealing victim selection, AEG construction, and the C++
kernels' tie-breaking all draw from a single explicit RNG threaded through
the call stack.

The C++ and Python paths produce identical eviction decisions on the same
input. Tests in `tests/test_native.py` enforce this invariant by exercising
both paths and asserting equality of victim selection.

---

## 📁 Repository Layout

```
saga/
├── configs/                       Hydra configurations
│   ├── config.yaml                top-level
│   ├── workload/                  SWE-bench, WebArena, BurstGPT
│   ├── cluster/                   single-node, 32-GPU, 64-GPU
│   ├── scheduler/                 vLLM, vLLM+APC, ..., SAGA
│   └── experiment/                e2e, ablation, fairness, bfsdfs,
│                                  competitive, sensitivity, tool_variance
├── csrc/                          C++17 hot-path kernels (OpenMP)
│   └── saga_native.cpp            WA-LRU + Bélády + lock-free SessionTable
├── src/saga/
│   ├── core/                      AEG + domain types
│   ├── cache/                     policies, TTL, manager, DRAM tier
│   ├── scheduler/                 router, work-stealer, BFS/DFS/Hybrid,
│   │                              coordinator
│   ├── fairness/                  AFS
│   ├── workflow/                  hint parser, pattern inference
│   ├── workload/                  generators (SWE-bench, WebArena, BurstGPT)
│   ├── sim/                       discrete-event engine
│   ├── analysis/                  metrics, stats, tables
│   ├── entrypoints/               simulate / benchmark / evaluate
│   ├── native.py                  C++ extension wrapper + fallback
│   ├── presets.py                 13 named scheduler bundles
│   └── cli.py                     typer CLI
├── tests/                         57 unit + integration tests
├── docs/                          DATA / EXPERIMENTAL_DETAILS /
│                                  TROUBLESHOOTING
├── CMakeLists.txt                 canonical native build
├── setup_native.py                pybind11-only build shim
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## 📜 Algorithms in Code

Every formula in the paper appears in code with the same names:

* `P_evict` → [`WALRUPolicy.score`](src/saga/cache/policies.py)
* `P_reuse` → [`AgentExecutionGraph.predict_reuse`](src/saga/core/aeg.py)
* TTL computation → [`ToolTTLPolicy.compute_ttl_ms`](src/saga/cache/ttl.py)
* routing rule → [`SessionRouter.route`](src/saga/scheduler/routing.py)
* work-stealing trigger → [`WorkStealer.step`](src/saga/scheduler/stealing.py)
* urgency → [`TenantUrgency.urgency`](src/saga/fairness/afs.py)
* AFS allocation → [`AFSScheduler.allocation`](src/saga/fairness/afs.py)
* Bélády oracle → [`BeladyOracle`](src/saga/cache/policies.py)
* pattern inference → [`PatternInferenceEngine.infer_aeg`](src/saga/workflow/pattern.py)
* PCIe swap-time model → [`SwapTimeModel.transfer_ms`](src/saga/cache/dram_tier.py)

---

## 🔬 Honest Calibration Note

This artifact is a *discrete-event simulator*, not a real serving system on
real GPUs. The simulator faithfully captures the algorithmic mechanisms:

* SAGA's session affinity yields measurably higher cache hit rate than vLLM's
  least-loaded routing on the same trace.
* WA-LRU evicts terminal-node sessions in preference to actively-resuming
  ones (verified in `tests/test_cache_policies.py`).
* AFS produces a restoring drift that pulls underserved tenants up in the
  priority order (`tests/test_afs.py`).
* C++ and Python eviction kernels yield identical decisions
  (`tests/test_native.py`).

The 1.64× geomean speedup in the paper emerges at production scale (64 GPUs,
hundreds of concurrent sessions per worker). The simulator's
one-session-per-worker-step model surfaces the mechanisms but compresses the
absolute speedup ratio. See
[`docs/EXPERIMENTAL_DETAILS.md`](docs/EXPERIMENTAL_DETAILS.md) for the
calibration discussion.

---

## 🤝 Acknowledgements

We thank the anonymous reviewers for their constructive feedback. SAGA
builds on PagedAttention (vLLM), RadixAttention (SGLang), Llumnix's live
migration, KVFlow's workflow-aware eviction, and the work-stealing
scheduling work of Blumofe and Leiserson.
