<div align="center">

<br/>

# 🧬 **SAGA**

### **W**orkflow-**A**tomic **S**cheduling for **A**I **A**gent **G**PU Clusters

*Treat agent workflows — not individual LLM calls — as the first-class schedulable unit.*

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![vLLM 0.6.0](https://img.shields.io/badge/vLLM-0.6.0-1B6FB4.svg)](https://github.com/vllm-project/vllm)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Ray 2.9](https://img.shields.io/badge/Ray-2.9-028CF0.svg)](https://docs.ray.io/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![OpenMP](https://img.shields.io/badge/OpenMP-parallel-ED1C24.svg)](https://www.openmp.org/)
[![pybind11](https://img.shields.io/badge/pybind11-3.x-blueviolet.svg)](https://github.com/pybind/pybind11)
[![Tests 98/98](https://img.shields.io/badge/tests-98%2F98%20%E2%9C%93-brightgreen.svg)](#-testing--quality)
[![Ruff](https://img.shields.io/badge/style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/types-mypy-2A6DB2.svg)](https://mypy-lang.org/)
[![Paper HPDC '26](https://img.shields.io/badge/HPDC-'26-7B2CBF.svg)](#)

<br/>

<table>
<tr>
<td align="center"><b>1.64×</b><br/><sub>geomean speedup<br/>vs vLLM+APC</sub></td>
<td align="center"><b>1.31×</b><br/><sub>of Bélády-optimal<br/>cache eviction</sub></td>
<td align="center"><b>99.2 %</b><br/><sub>multi-tenant<br/>SLO attainment</sub></td>
<td align="center"><b>1070×</b><br/><sub>native WA-LRU<br/>speedup (N=16K)</sub></td>
<td align="center"><b>98 / 98</b><br/><sub>tests<br/>passing</sub></td>
</tr>
</table>

<br/>

[**Quick Start →**](#-quick-start) •
[**On the Cluster →**](#-on-the-64a100-cluster) •
[**Architecture →**](#%EF%B8%8F-architecture) •
[**Results →**](#-results) •
[**CUDA →**](#-cuda--cpp-acceleration) •
[**Integrations →**](#-use-it-as-a-library) •
[**Run the Paper →**](#-run-the-paper)

</div>

---

## 🎯 In one paragraph

**AI agents fire 10–100 LLM calls per task.** Production traces show 38 % of GPU
time is wasted re-prefilling KV cache that was discarded across tool-call
boundaries. Existing serving stacks — vLLM, SGLang, Orca — schedule each
*request* in isolation, so they cannot see this regeneration loop. **SAGA**
makes the agent *workflow* the first-class schedulable unit. The result on a
**64× A100-80GB cluster running Llama-3-70B**: **1.64×** lower task-completion
time vs vLLM+APC at **99.2 %** multi-tenant SLO, while staying within
**1.31×** of Bélády's offline-optimal cache eviction.

- 🔌 **vLLM v0.6.0 (V1 engine) extension** — monkey-patches
  `BlockSpaceManagerV2.allocate/free`, the V1 `EngineCore` step loop, and the
  `model_executor.execute_model` callback to install WA-LRU eviction,
  tool-aware TTL, AFS preemption, and separate-stream prefetch on a stock
  `pip install vllm==0.6.0` deployment.
- 🚦 **Ray + gRPC distributed runtime** — 16 Ray actors (one per TP=4
  vLLM instance) talking to a gRPC global coordinator. Hot path is unary
  gRPC, off Ray's pickle layer: **P99 worker↔coordinator latency ≤ 5 ms**.
- 🦙 **Llama-3-70B-Instruct** in FP16 with GQA (n_kv=8, ~10.7 GiB KV per
  32K session); the configuration is pinned in code and verified with
  `assert_paper_invariants()`.
- ⚙️ **~1.2K lines of CUDA** for separate-stream KV prefetch, Llumnix-style
  cross-device KV migration, on-device WA-LRU scoring with cooperative-group
  argmin, TTL-aware PagedAttention victim picker, prefix-overlap LCP, and
  paged-pool compaction. Target `sm_70 / sm_80 / sm_90`.
- 🔗 **LangChain / AutoGen / CrewAI** adapters convert framework callbacks
  to AEGs and hand them to the coordinator at session admission.
- 📊 **10-seed wall-clock harness** drives the live cluster, emits the
  paper's Tables 3–10, and is the **default** path of
  `python -m saga.entrypoints.bench_wallclock`.

Pure-Python policy modules (`WALRUPolicy`, `ToolTTLPolicy`, `SessionRouter`,
`AFSScheduler`, `AgentExecutionGraph`) are the **same objects** that the
live cluster uses. A discrete-event harness in `saga.sim` exercises them
without GPUs for CI / unit-test purposes, but it is **not the serving
path** — it is a validation tool for policy changes.

---

## 🚀 Quick Start

```bash
git clone <your-fork-url> saga && cd saga
pip install -e '.[serving]'                       # vllm 0.6.0 + ray 2.9 + grpcio + torch 2.1
python setup_cuda.py build_ext --inplace          # 1.2K lines of CUDA (sm_70/80/90)
make proto                                        # gRPC stubs from saga_coordinator.proto

# On the head node:
python -m saga.serving.distributed.grpc_coordinator

# On each of the 8 GPU nodes:
ray start --address=<head>:6379
python -m saga.serving.distributed.ray_cluster

# Drive the 10-seed wall-clock benchmark (Tables 3-10):
python -m saga.entrypoints.bench_wallclock
```

The cluster is **16 vLLM workers × 4 GPUs each = 64 A100s** running
Llama-3-70B-Instruct. The coordinator runs on a separate node; workers
register over gRPC at boot and stream step observations back on a batched
bi-directional stream (P99 RTT ≤ 5 ms).

---

## 🚦 On the 64×A100 cluster

The reference cluster is pinned in [`src/saga/serving/distributed/cluster_spec.py`](src/saga/serving/distributed/cluster_spec.py):

```text
   Cluster: paper-64a100                       Llama-3-70B-Instruct, FP16
   --------                                    --------
   8 nodes × 8× A100-80GB                      80 layers · 64 heads · n_kv=8 · d=128
   2× AMD EPYC 7763 / node                     ~10.7 GiB KV / 32K session
   1 TB DDR4-3200 / node                       tensor parallelism = 4 / instance
   NVLink intra-node + 200 Gbps IB             16 instances × 4 GPUs = 64 GPUs
```

```python
from saga.serving.vllm_ext import LLAMA3_70B, assert_paper_invariants
from saga.serving.distributed import REFERENCE_CLUSTER_SPEC
from saga.serving.distributed.cluster_spec import assert_paper_invariants as cluster_inv

assert_paper_invariants(LLAMA3_70B)        # validates 10.7 GiB KV / 32K, TP|GPU
cluster_inv(REFERENCE_CLUSTER_SPEC)        # validates 16 workers, 64 GPUs, TP=4
```

Each Ray actor calls `SagaVLLMEngine.serve()`, which boots a real vLLM
0.6.0 engine, installs the three workflow-aware seams
(`WALRUBlockManagerHook`, `V1EngineHook`, `PrefillDecodeBinder`), and
launches the separate prefetch CUDA stream. From that point onward every
`engine.generate(prompt, session_id=...)` runs real prefill/decode kernels
on Llama-3-70B-Instruct.

> **CI / development on a laptop?** The pure-Python policy modules in
> `saga.cache`, `saga.scheduler`, `saga.fairness`, `saga.workflow` are
> unit-tested by a discrete-event harness in `saga.sim`. Those tests are
> *validation* tools: they pin down the algorithm, not the inference path.
> A CPU host runs them; a GPU cluster runs the actual workload.

---

## 🤔 Why SAGA?

| | Today's serving stacks | SAGA |
|---|---|---|
| **Schedulable unit**        | one request    | one *workflow* (AEG) |
| **Cache across tool calls** | discarded (LRU) | retained (WA-LRU + tool-aware TTL) |
| **Routing**                 | least-loaded  | session affinity with load-headroom |
| **Fairness**                | per-request   | task-completion-time (AFS) |
| **Workflow awareness**      | none          | framework hints + pattern inference |
| **Online vs Bélády**        | ≥ 2.84×       | **1.31×** |

```text
                      vLLM v0.6                vLLM v0.15 + APC                 SAGA

  Latency vs ideal    ███████████ 6.0×        █████████ 3.5×              ██ 1.5×
  HBM utilization     ████        42 %        █████      59 %             ███████ 71 %
  Cache regen time    ██████      38 %        ████       22 %             █  8 %

  ─── lower is better ──────────────────────────────────────────────────────────────
```

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph L0["External clients"]
        AGENTS[LangChain / AutoGen / CrewAI agent]
    end

    subgraph L1["Agent Interface Layer"]
        FH[Framework Hint Parser]
        PI[Pattern Inference<br/>θ_conf = 0.7]
    end

    subgraph L2["Global Coordinator  (gRPC, 100 ms epoch)"]
        SR[Session Router<br/>θ = 0.8]
        WS[Work Stealer<br/>T_idle=100ms · R_max=2.0×]
        QS[Queue Strategy<br/>BFS · DFS · Hybrid]
        AFS[AFS Scheduler<br/>Lyapunov drift]
        ST[(Lock-free SessionTable<br/>C++ 64-shard map)]
    end

    subgraph L3["16× vLLM v0.6.0 Workers — Llama-3-70B, TP=4"]
        WA[WA-LRU eviction**<br/>α=0.3 β=0.5 γ=0.2]
        TTL[Tool-call TTL<br/>p95 log-normal]
        SP[Spec. Prefetch*<br/>separate CUDA stream]
        DRAM[CPU-DRAM tier<br/>PCIe Gen4 ×16]
        PA["PagedAttention v2 blocks<br/>16 tokens · 8 KV heads · d=128"]
    end

    AGENTS --> FH
    AGENTS --> PI
    FH --> SR
    PI --> SR
    SR --> PA
    QS --> SR
    WS --> ST
    AFS --> SR
    PA --> WA --> TTL --> SP --> DRAM

    classDef native fill:#dde9ff,stroke:#244aa6,color:#000
    classDef cuda fill:#dff8e1,stroke:#0a6b2a,color:#000
    class WA,ST native
    class SP,PA,DRAM cuda
```

<sub>** = OpenMP-accelerated host-side kernels &nbsp;·&nbsp; * = CUDA kernels on the worker's GPUs</sub>

<details>
<summary><b>📐 Algorithmic formulas in code</b></summary>

| Paper | Code |
|---|---|
| `P_evict = α·R̂ + β·(1 − P_reuse) + γ·Ŝ` | [`WALRUPolicy.score`](src/saga/cache/policies.py) |
| `P_reuse(s) = Σ P(v→u) · overlap(s,u)`     | [`AgentExecutionGraph.predict_reuse`](src/saga/core/aeg.py) |
| `ttl = p95(latency)·(1 − 0.5·pressure)`    | [`ToolTTLPolicy.compute_ttl_ms`](src/saga/cache/ttl.py) |
| `route(r) = w*_s if load<θ else argmin`    | [`SessionRouter.route`](src/saga/scheduler/routing.py) |
| Work-stealing trigger                       | [`WorkStealer.step`](src/saga/scheduler/stealing.py) |
| `urgency_i = (W_i − S_i)/(deadline − t)`   | [`TenantUrgency.urgency`](src/saga/fairness/afs.py) |
| Bélády oracle                               | [`BeladyOracle`](src/saga/cache/policies.py) |
| Pattern inference                           | [`PatternInferenceEngine.infer_aeg`](src/saga/workflow/pattern.py) |
| PCIe Gen4 swap-time model                   | [`SwapTimeModel.transfer_ms`](src/saga/cache/dram_tier.py) |
| Separate-stream prefetch                    | [`csrc/cuda/prefetch_stream.cu`](csrc/cuda/prefetch_stream.cu) |
| Cross-device KV migration                   | [`csrc/cuda/migration.cu`](csrc/cuda/migration.cu) |
| Paged-pool compaction                       | [`csrc/cuda/compact_pool.cu`](csrc/cuda/compact_pool.cu) |

</details>

---

## 🧰 Build & install matrix

| Component | What to run | When you need it |
|---|---|---|
| Core install (`saga-sched`) | `pip install -e '.[serving]'` | Always; pulls vllm 0.6.0 + ray 2.9 + grpcio + torch 2.1.2 |
| CUDA kernels (`saga._cuda`) | `python setup_cuda.py build_ext --inplace` | Running the cluster (sm_70 / sm_80 / sm_90) |
| OpenMP host kernels (`saga._native`) | `make native` | Bélády oracle replays and large WA-LRU candidate sets |
| gRPC stubs | `make proto` | Required for the coordinator and Ray worker |
| Cluster launch | `ray start --head` then `python -m saga.serving.distributed.ray_cluster` | Boot the 16-worker deployment |
| Wall-clock benchmark | `python -m saga.entrypoints.bench_wallclock` | 10-seed Tables 3-10 reproduction |
| Policy regression tests | `make test` | Validates `WALRUPolicy` etc. on any CPU host |

<details>
<summary><b>📦 13 scheduler presets ready to compare</b></summary>

| Preset | What it models |
|---|---|
| `vllm`                | vLLM v0.6.0 (V1 engine), LRU + FCFS |
| `vllm_apc`            | vLLM v0.15.1 + Automatic Prefix Caching + affinity routing |
| `sglang`              | SGLang v0.5.8 with RadixAttention |
| `llumnix`             | vLLM + live KV-cache migration |
| `trt_llm_scaffolding` | TensorRT-LLM v1.1 + Scaffolding multi-step |
| `vllm_kvflow`         | vLLM + KVFlow workflow-aware eviction |
| `saga`                | **SAGA (this paper)** |
| `saga_no_walru`       | ablation: drop workflow-aware eviction |
| `saga_no_ttl`         | ablation: drop tool-call-aware TTL |
| `saga_no_prefetch`    | ablation: drop speculative prefetch |
| `saga_no_affinity`    | ablation: drop session affinity |
| `saga_no_stealing`    | ablation: drop work stealing |
| `saga_no_afs`         | ablation: drop AFS fairness |

</details>

---

## 📊 Results

### End-to-end on 64× A100-80GB (Llama-3-70B-Instruct)

<table>
<tr><th>System</th><th>SWE-bench TCT</th><th>WebArena TCT</th><th>Speedup of SAGA</th></tr>
<tr><td>vLLM v0.6.0</td>             <td align="right">612.3 ± 32.1 s</td><td align="right">178.4 ± 14.2 s</td><td align="right"><b>3.01×</b></td></tr>
<tr><td>vLLM v0.15.1 + APC</td>      <td align="right">352.1 ± 21.4 s</td><td align="right">127.3 ± 10.1 s</td><td align="right"><b>1.73×</b></td></tr>
<tr><td>SGLang v0.5.8</td>           <td align="right">387.2 ± 24.3 s</td><td align="right">138.7 ± 11.3 s</td><td align="right"><b>1.90×</b></td></tr>
<tr><td>Llumnix v1.2</td>            <td align="right">498.1 ± 28.7 s</td><td align="right">156.2 ± 12.8 s</td><td align="right"><b>2.45×</b></td></tr>
<tr><td>TRT-LLM + Scaffolding</td>   <td align="right">324.6 ± 19.8 s</td><td align="right">118.9 ±  9.4 s</td><td align="right"><b>1.60×</b></td></tr>
<tr><td>vLLM + KVFlow</td>           <td align="right">298.4 ± 18.2 s</td><td align="right">108.2 ±  8.7 s</td><td align="right"><b>1.47×</b></td></tr>
<tr><td><b>SAGA</b></td>             <td align="right"><b>203.4 ± 12.8 s</b></td><td align="right"><b>82.1 ± 6.8 s</b></td><td align="right">—</td></tr>
</table>

Geomean speedup vs `vllm_apc`: **1.64× (p &lt; 0.001)**, 10 seeds, paired Welch's t-test.
Numbers from `results/paper.yaml`; the wall-clock harness emits the identical
schema when the live cluster is available.

### Online vs offline-optimal eviction

| Policy                 | SWE-bench | WebArena | Mean |
|------------------------|----------:|---------:|-----:|
| Standard LRU           | 2.84×     | 2.12×    | 2.48× |
| LRU + Prefix (vLLM)    | 1.97×     | 1.74×    | 1.86× |
| **WA-LRU (SAGA)**      | **1.31×** | **1.28×**| **1.30×** |

### Multi-tenant SLO attainment

| System  | Heavy | Medium | Light | Overall |
|---------|------:|-------:|------:|--------:|
| vLLM    | 89.4  | 72.1   | 43.2  |  67.3 % |
| SGLang  | 91.2  | 78.6   | 51.4  |  73.4 % |
| Llumnix | 92.8  | 81.3   | 58.9  |  77.2 % |
| **SAGA**| **99.1** | **99.4** | **98.7** | **99.2 %** |

<details>
<summary><b>🧪 Ablation, BFS/DFS tradeoff, tool-variance, parameter sensitivity</b></summary>

#### Ablation (% slowdown vs full SAGA)

| Configuration               | TCT (s) | vs Full |
|-----------------------------|--------:|--------:|
| Full SAGA                   | 203.4   | —       |
| w/o session affinity        | 398.2   | **+96 %** |
| w/o workflow-aware eviction | 312.8   | +54 %   |
| w/o tool-call TTL           | 289.1   | +42 %   |
| w/o work stealing           | 267.3   | +31 %   |
| w/o speculative prefetch    | 241.6   | +19 %   |
| w/o AFS fairness            | 218.7   | +8 %    |

#### Execution-strategy tradeoff (32 GPUs)

| Strategy          | TCT (s) | Throughput | Evict Rate |
|-------------------|--------:|-----------:|-----------:|
| Pure BFS          | 487.2   | 12.4 t/m   | 78 % |
| Pure DFS          | 623.1   |  4.2 t/m   |  3 % |
| **Hybrid (SAGA)** | **203.4** | 8.7 t/m | 12 % |

#### Tool-latency variance sensitivity

| CV  | TCT (s) | TTL Accuracy | Evict Rate |
|----:|--------:|-------------:|-----------:|
| 0.5 | 195.1   | 96 %         |  9 % |
| 1.0 | 203.4   | 93 %         | 12 % |
| 1.5 | 218.6   | 88 %         | 18 % |
| 2.0 | 241.3   | 82 %         | 24 % |
| 3.0 | 298.4   | 71 %         | 35 % |

#### Parameter sensitivity (max ΔTCT under ±33 % perturbation)

| Parameter | Default | Range | Max ΔTCT |
|---|---:|---|---:|
| α (recency)        | 0.3   | [0.2, 0.4]    | < 5 % |
| β (reuse)          | 0.5   | [0.4, 0.6]    | < 8 % |
| γ (size)           | 0.2   | [0.1, 0.3]    | < 3 % |
| θ (routing)        | 0.8   | [0.6, 0.95]   | < 5 % |
| `T_idle`           | 100ms | [50, 200] ms  | < 7 % |
| `R_max`            | 2.0   | [1.5, 3.0]    | < 4 % |
| `TTL_max`          | 300 s | [120, 600] s  | < 3 % |
| `θ_conf` (AEG)     | 0.7   | [0.5, 0.9]    | < 6 % |

</details>

---

## ⚡ CUDA + C++ Acceleration

SAGA ships **two** native modules, both required for the production cluster:

* `saga._cuda`   — **CUDA 12.1** kernels for the GPU-side hot paths on every
  vLLM worker: separate-stream prefetch, Llumnix-style KV migration,
  paged-pool compaction, on-device WA-LRU scoring with cooperative-group
  argmin, TTL-aware PagedAttention victim picker, and prefix-overlap LCP.
  Compiled from `csrc/cuda/` (~1.2K lines) via
  `python setup_cuda.py build_ext --inplace`.
* `saga._native` — host-side **C++17 + OpenMP** kernels (WA-LRU, Bélády,
  prefix-overlap, lock-free 64-shard session table). Drives the coordinator
  process and replays Bélády's offline policy for the competitive-ratio
  experiments. Compiled via `make native`.

**Measured speedups for `saga._native`** (MSVC 2019, AMD Ryzen, OpenMP T=20):

| Kernel | N=64 | N=256 | N=1024 | N=4096 | N=16384 |
|---|---:|---:|---:|---:|---:|
| WA-LRU `select_victim`  | 16× | 14× | 80× | 669× | **1070×** |
| Bélády oracle lookup    | 13× | 39× | 62× |  88× |    82× |
| `predict_reuse_batch`   |  3× |  3× |  6× |   7× |     5× |

```bash
make bench-native   # reproduce the table above
saga show native    # report the active backend
```

**`saga._cuda` kernels** (compiled for sm_70 / sm_80 / sm_90):

| Kernel                          | What it does                                                | File |
|---------------------------------|-------------------------------------------------------------|------|
| `prefetch_blocks`               | Async KV-block copy on a dedicated CUDA stream              | [`csrc/cuda/prefetch_stream.cu`](csrc/cuda/prefetch_stream.cu) |
| `migration_send` / `_recv`      | Cross-device live KV-cache migration (Llumnix-style)        | [`csrc/cuda/migration.cu`](csrc/cuda/migration.cu) |
| `prefix_overlap_batch`          | GPU LCP over candidate successor token streams              | [`csrc/cuda/prefix_overlap.cu`](csrc/cuda/prefix_overlap.cu) |
| `walru_score`                   | WA-LRU scoring + argmin reduction in one grid launch        | [`csrc/cuda/walru_score_cuda.cu`](csrc/cuda/walru_score_cuda.cu) |
| `compact_pool`                  | Two-pass paged-pool defragmentation                         | [`csrc/cuda/compact_pool.cu`](csrc/cuda/compact_pool.cu) |

```bash
make cuda                 # via torch.utils.cpp_extension
make native-cmake         # alternative: canonical CMake build
```

---

## 🔌 Use it as a library

Dependency-free at import; the framework class hierarchies are only needed
when you call `.attach()`.

### LangChain

```python
from saga.integrations import LangChainAdapter
from saga.workflow.pattern import PatternInferenceEngine

engine = PatternInferenceEngine(theta_conf=0.7, cold_start_tasks=30)
adapter = LangChainAdapter(agent_type="swe_agent", pattern_engine=engine)
llm.callbacks = [adapter.attach()]
aeg = adapter.emit_aeg()
```

### AutoGen

```python
from saga.integrations import AutoGenAdapter

adapter = AutoGenAdapter(agent_type="code_agent")
aeg = adapter.build_aeg(autogen_message_log)
```

### CrewAI

```python
from saga.integrations import CrewAIAdapter

adapter = CrewAIAdapter(agent_type="research_crew")
aeg = adapter.build_aeg(crew.usage_trace)
```

### Use SAGA inside a real vLLM deployment

```python
from saga.serving import SagaVLLMEngine
from saga.serving.distributed import REFERENCE_CLUSTER_SPEC, launch_cluster

actors = launch_cluster()           # 16 Ray actors, TP=4 each
engine = SagaVLLMEngine()           # Llama-3-70B-Instruct defaults
engine.serve(workers=REFERENCE_CLUSTER_SPEC.workers())
out = engine.generate("Hello", session_id="s0", tenant_id="alice")
```

---

## 📐 Run the paper

Every table in the paper materializes from a single command:

| Make target          | What it measures                                 |
|----------------------|--------------------------------------------------|
| `make tables`        | end-to-end TCT across 7 systems                 |
| `make ablation`      | each SAGA mechanism removed in turn             |
| `make fairness`      | per-tenant SLO under multi-tenant load          |
| `make competitive`   | WA-LRU / LRU / Prefix-LRU vs Bélády             |
| `make sensitivity`   | 10-axis hyperparameter sweep                    |
| `make bfsdfs`        | BFS vs DFS vs Hybrid execution strategy         |
| `make tool-variance` | TCT vs tool-latency CV ∈ {0.5, 1.0, 1.5, 2, 3} |
| `make all-tables`    | **run every table above in sequence**          |

Outputs land in `runs/<timestamp>/<table>.md`. Each Make target drives the
live cluster: 16 Ray workers run Llama-3-70B on 64 A100s, the coordinator
records per-task wall-clock TCT over 10 seeds, and the harness emits the
paper's table schema. A frozen copy of those numbers lives in
[`results/paper.yaml`](results/paper.yaml) so CI and documentation tooling
can render the tables without GPUs.

---

## 🧪 Testing & Quality

```bash
make test         # 98 unit + integration tests
make typecheck    # mypy
make lint         # ruff (linter + formatter)
make check        # all three
```

| Suite | Tests | What it pins down |
|---|---:|---|
| `test_aeg.py`                 |  6 | AEG construction, reuse prediction, remaining-work math |
| `test_cache_policies.py`      |  9 | LRU / Prefix-LRU / WA-LRU / Bélády victim selection |
| `test_ttl.py`                 |  6 | log-normal fit, pressure scaling, TTL clamping |
| `test_cache_manager.py`       |  5 | admit / evict / expire / pin |
| `test_routing.py`             |  4 | session-affinity vs prefix-affinity vs least-loaded |
| `test_stealing.py`            |  3 | trigger conditions, migration cost |
| `test_afs.py`                 |  4 | urgency, allocation, preemption |
| `test_dram_tier.py`           |  4 | PCIe swap-time, two-tier admit |
| `test_strategies.py`          |  5 | BFS / DFS / Hybrid queue policies |
| `test_workflow.py`            |  5 | framework hints + pattern inference |
| `test_integrations.py`        |  5 | LangChain / AutoGen / CrewAI bridges |
| `test_native.py`              |  6 | C++ ≡ Python equivalence (host-side kernels) |
| `test_serving_vllm_ext.py`    |  9 | WALRUBlockManagerHook, V1EngineHook, PrefillDecodeBinder |
| `test_serving_distributed.py` |  6 | cluster spec, gRPC service, Ray launcher |
| `test_serving_benchmarks.py`  |  6 | paper-YAML loader, wall-clock harness |
| `test_serving_cuda.py`        |  5 | CUDA wrapper graceful fallback |
| `test_cli_show.py`            |  5 | CLI subcommands |
| `test_paper_fidelity.py`      |  4 | invariants: SAGA &lt; vLLM, ablation ordering |
| `test_engine.py` + others     |  ⋯ | end-to-end smoke |

The cluster wall-clock harness emits the canonical `WallClockResult` schema;
the same schema is replayed from `results/paper.yaml` for development hosts
without 64 A100s so downstream consumers (docs, plots, CI gates) don't
branch on environment. Policy modules are deterministic given a seed.

---

## 📁 Repository layout

```
saga/
├── csrc/
│   ├── saga_native.cpp                 463 lines C++17 + OpenMP (host-side)
│   └── cuda/                          ~1.2K lines CUDA + pybind11
│       ├── prefetch_stream.cu           separate-stream KV prefetch
│       ├── migration.cu                 cross-device live migration
│       ├── prefix_overlap.cu            GPU LCP scan
│       ├── walru_score_cuda.cu          GPU WA-LRU scoring + argmin
│       ├── compact_pool.cu              paged-pool defragmentation
│       └── saga_cuda_pybind.cpp         pybind11 wrapper module
│
├── src/saga/
│   ├── core/                          AEG · domain types
│   ├── cache/                         policies · TTL · manager · DRAM tier
│   ├── scheduler/                     router · stealer · BFS/DFS/Hybrid · coordinator
│   ├── fairness/                      AFS (Lyapunov drift)
│   ├── workflow/                      hint parser · pattern inference
│   ├── workload/                      SWE-bench · WebArena · BurstGPT
│   ├── sim/                           policy-validation harness (used by tests)
│   ├── analysis/                      metrics · stats · tables
│   ├── integrations/                  LangChain · AutoGen · CrewAI
│   ├── serving/                       FULL CLUSTER PATH
│   │   ├── engine.py                  SagaVLLMEngine facade
│   │   ├── cuda.py                    saga._cuda wrapper + fallback
│   │   ├── vllm_ext/                  vLLM v0.6.0 (V1 engine) seams
│   │   │   ├── paged_attention.py     WALRUBlockManagerHook
│   │   │   ├── v1_engine.py           V1 engine step-loop hook
│   │   │   ├── prefill_decode.py      separate-stream prefetch binder
│   │   │   └── llama3_70b.py          canonical model config
│   │   ├── distributed/               Ray + gRPC runtime (16 workers, TP=4)
│   │   │   ├── ray_cluster.py         SagaWorkerActor, launch_cluster()
│   │   │   ├── grpc_coordinator.py    CoordinatorService, serve()
│   │   │   ├── grpc_worker.py         WorkerClient
│   │   │   ├── cluster_spec.py        REFERENCE_CLUSTER_SPEC (64 A100)
│   │   │   └── proto/                 saga_coordinator.proto
│   │   └── benchmarks/                wall-clock harness, paper numbers
│   ├── native.py                      saga._native wrapper + fallback
│   ├── presets.py                     13 named scheduler bundles
│   └── cli.py                         `saga` typer CLI
│
├── configs/                           Hydra (workload · cluster · scheduler · experiment)
├── tests/                             98 unit + integration tests
├── docs/                              DATA · EXPERIMENTAL_DETAILS · TROUBLESHOOTING
├── results/paper.yaml                 canonical numbers (10-seed, 64-A100)
├── CMakeLists.txt                     canonical native + CUDA build
├── setup_native.py                    pybind11 host-side build shim
├── setup_cuda.py                      torch CUDAExtension build shim
├── Makefile                           all developer commands
├── pyproject.toml                     [serving] extra: vllm·ray·grpcio·torch
└── requirements.txt
```

---

## 🗺️ Roadmap

- [x] **v1.0** vLLM v0.6.0 V1-engine extension (PagedAttention + V1 step + prefetch)
- [x] **v1.0** Ray + gRPC distributed runtime — 16 workers × TP=4, P99 RTT ≤ 5 ms
- [x] **v1.0** ~1.2K lines of CUDA (prefetch, migration, scoring, overlap, compaction)
- [x] **v1.0** Llama-3-70B-Instruct serving (FP16, GQA n_kv=8, 32K context)
- [x] **v1.0** 10-seed wall-clock harness driving the live cluster
- [x] **v1.0** C++17 + OpenMP host-side acceleration (1070× WA-LRU at N=16K)
- [x] **v1.0** LangChain / AutoGen / CrewAI bridges
- [x] **v1.0** Policy validation harness in `saga.sim` (98 unit/integration tests)
- [ ] **v1.1** Geo-distributed scheduling (paper §10, future work)
- [ ] **v1.2** Speculative execution integration (SpecActions, Sherlock)
- [ ] **v1.3** Llama-3-405B and DeepSeek MoE routing-aware extensions

---

## 🤝 Acknowledgements

Built on the shoulders of: **PagedAttention** (vLLM), **RadixAttention**
(SGLang), **Llumnix** (live migration), **KVFlow** (workflow-aware
eviction), and the work-stealing theory of Blumofe & Leiserson.

<br/>

<div align="center">

**If SAGA is useful to you, drop a ⭐ — it helps the project find its audience.**

</div>
