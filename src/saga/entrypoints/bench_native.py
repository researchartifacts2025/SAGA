"""Microbenchmark: native C++ vs pure-Python eviction kernels.

Measures wall-clock time for WA-LRU ``select_victim``, Bélády ``select_victim``
and batched ``predict_reuse`` across cache-pool sizes that span small (8) up
to large (16 384). Reports speedup at each size and the OpenMP thread count.

Usage:
    python -m saga.entrypoints.bench_native
    python -m saga.entrypoints.bench_native --sizes 64 256 1024 4096
    python -m saga.entrypoints.bench_native --repeats 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterable

import numpy as np
from rich.console import Console
from rich.table import Table

from saga import is_native_available, native_build_info
from saga.cache.policies import WALRUPolicy
from saga.core.aeg import build_linear_aeg
from saga.core.types import KVCacheEntry, ToolType
from saga.native import (
    belady_select_victim_flat,
    predict_reuse_batch,
    walru_select_victim_flat,
)


console = Console()


def _make_entries(n: int) -> list[KVCacheEntry]:
    return [
        KVCacheEntry(
            session_id=f"s{i}",
            worker_id=0,
            n_tokens=200 + (i % 7) * 50,
            last_access_time=float(i),
            creation_time=0.0,
        )
        for i in range(n)
    ]


def _bench_walru(n: int, repeats: int) -> tuple[float, float]:
    n_tokens = np.array([200 + (i % 7) * 50 for i in range(n)], dtype=np.int64)
    last_access = np.array([float(i) for i in range(n)], dtype=np.float64)
    reuse = np.full(n, 0.5, dtype=np.float64)
    pinned = np.zeros(n, dtype=np.uint8)
    now = float(n) + 1.0

    def _native_run() -> None:
        walru_select_victim_flat(
            n_tokens,
            last_access,
            reuse,
            pinned,
            now=now,
            tau_max=float(n),
            size_max=400.0,
            alpha=0.3,
            beta=0.5,
            gamma=0.2,
        )

    def _python_run() -> None:
        entries = _make_entries(n)
        policy = WALRUPolicy(use_native=False)
        from saga.cache.policies import PolicyContext

        ctx = PolicyContext()
        ctx.with_max(entries, now=now)
        policy.select_victim(entries, now=now, ctx=ctx)

    t_native = _time(_native_run, repeats=repeats)
    t_python = _time(_python_run, repeats=repeats)
    return t_native, t_python


def _bench_belady(n: int, repeats: int) -> tuple[float, float]:
    # Build CSR-encoded future-access list.
    future_offsets = np.empty(n + 1, dtype=np.int64)
    future_times = np.empty(n * 8, dtype=np.float64)
    for i in range(n):
        future_offsets[i] = i * 8
        for j in range(8):
            future_times[i * 8 + j] = float(i * 3 + j)
    future_offsets[n] = n * 8
    pinned = np.zeros(n, dtype=np.uint8)
    now = float(n) + 1.0

    def _native_run() -> None:
        belady_select_victim_flat(pinned, future_times, future_offsets, now=now)

    def _python_run() -> None:
        best_idx = -1
        best_next = -float("inf")
        found_inf = False
        for i in range(n):
            if pinned[i]:
                continue
            lo = int(future_offsets[i])
            hi = int(future_offsets[i + 1])
            nxt = None
            for j in range(lo, hi):
                if future_times[j] > now:
                    nxt = float(future_times[j])
                    break
            if nxt is None:
                if not found_inf:
                    found_inf = True
                    best_idx = i
                continue
            if found_inf:
                continue
            if nxt > best_next:
                best_next = nxt
                best_idx = i
        _ = best_idx

    t_native = _time(_native_run, repeats=repeats)
    t_python = _time(_python_run, repeats=repeats)
    return t_native, t_python


def _bench_predict_reuse(n: int, repeats: int) -> tuple[float, float]:
    aeg = build_linear_aeg(
        graph_id="bench",
        n_steps=37,
        tool_types=[ToolType.CODE_EXECUTION] * 37,
        prompt_tokens_est=3_000,
        output_tokens_est=250,
        observation_tokens_est=300,
    )
    cached = [3_000 + (i * 17) % 7_000 for i in range(n)]
    succ_probs: list[float] = []
    succ_obs: list[int] = []
    offsets: list[int] = []
    pos = 0
    for _ in range(n):
        offsets.append(pos)
        edges = aeg.successors(0)
        for edge in edges:
            succ_probs.append(edge.probability)
            succ_obs.append(aeg.node(edge.dst).observation_tokens_est)
            pos += 1
    offsets.append(pos)

    def _native_run() -> None:
        predict_reuse_batch(cached, succ_probs, succ_obs, offsets)

    def _python_run() -> None:
        out: list[float] = []
        total = len(succ_probs)
        for i in range(n):
            c = cached[i]
            if c <= 0:
                out.append(0.0)
                continue
            lo = offsets[i]
            hi = offsets[i + 1] if i + 1 < len(offsets) else total
            s = 0.0
            for j in range(lo, hi):
                p = succ_probs[j]
                obs = max(1, succ_obs[j])
                s += p * (c / (c + obs))
            out.append(max(0.0, min(1.0, s)))

    t_native = _time(_native_run, repeats=repeats)
    t_python = _time(_python_run, repeats=repeats)
    return t_native, t_python


def _time(fn, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _row(name: str, sizes: Iterable[int], repeats: int, bench) -> list[list[str]]:
    rows: list[list[str]] = []
    for n in sizes:
        t_native, t_python = bench(n, repeats=repeats)
        speedup = t_python / max(t_native, 1e-12)
        rows.append(
            [
                name,
                str(n),
                f"{t_python * 1e6:8.1f} µs",
                f"{t_native * 1e6:8.1f} µs",
                f"{speedup:6.2f}x",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[64, 256, 1024, 4096, 16_384])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    console.print(f"[bold]{native_build_info()}[/bold]")
    if not is_native_available():
        console.print(
            "[yellow]Native extension not built; both columns will be identical.[/yellow]"
        )

    table = Table(title="SAGA Native Acceleration Microbenchmark", show_lines=False)
    table.add_column("Kernel", style="cyan")
    table.add_column("N entries", justify="right")
    table.add_column("Pure Python", justify="right")
    table.add_column("Native (C++)", justify="right")
    table.add_column("Speedup", justify="right", style="green")

    for row in _row("WA-LRU select", args.sizes, args.repeats, _bench_walru):
        table.add_row(*row)
    for row in _row("Belady lookup", args.sizes, args.repeats, _bench_belady):
        table.add_row(*row)
    for row in _row("predict_reuse", args.sizes, args.repeats, _bench_predict_reuse):
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
