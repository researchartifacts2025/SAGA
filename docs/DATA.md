# Workloads

SAGA's evaluation is driven by three workload generators built into the
package. None of them require external data: each samples task definitions
from a calibrated distribution at construction time. The numbers below come
straight from the relevant production-trace tables.

## SWE-bench

Source code: [`saga.workload.swe_bench`](../src/saga/workload/swe_bench.py).

| Field              | Value         |
|--------------------|---------------|
| Step count         | gamma(2, 18.5), capped at 150 (mean 37) |
| Prompt tokens      | uniform[2 000, 4 000]           |
| Output tokens      | uniform[100, 500]               |
| Observation tokens | gamma(2, 175)                   |
| Tool mix           | 45 % code / 40 % file / 10 % web / 5 % DB |

Yield a workload of 500 verified tasks with

```bash
python -m saga.entrypoints.simulate workload=swe_bench workload.n_tasks=500
```

## WebArena

Source: [`saga.workload.web_arena`](../src/saga/workload/web_arena.py).

| Field              | Value                            |
|--------------------|----------------------------------|
| Step count         | gamma(2, 9), capped at 80 (mean 18) |
| Prompt tokens      | uniform[4 000, 8 000]            |
| Output tokens      | uniform[50, 200]                 |
| Observation tokens | gamma(2, 400)                    |
| Tool mix           | 75 % web / 15 % file / 5 % code / 5 % DB |

## BurstGPT-derived multi-tenant

Source: [`saga.workload.burst_gpt`](../src/saga/workload/burst_gpt.py).

Ten tenants:

| Tenant class | Count | Mean steps | Arrival rate (tasks/min) | Weight |
|--------------|------:|-----------:|-------------------------:|-------:|
| heavy        | 3     | 100        | 16                       | 3.0    |
| medium       | 4     |  30        |  8                       | 2.0    |
| light        | 3     |  10        |  4                       | 1.0    |

Per-tenant arrival is a Poisson process with the given rate; the aggregate
offered load is calibrated to drive cluster utilization to roughly 80 % of
peak throughput.

## Custom workloads

Subclass [`WorkloadGenerator`](../src/saga/workload/base.py) and register it
in [`saga.workload.__init__`](../src/saga/workload/__init__.py)'s
`build_workload` factory. Each `sample()` returns an `AgentTaskTemplate`
(`task`, `aeg`, `tool_plan`) which the engine consumes directly.
