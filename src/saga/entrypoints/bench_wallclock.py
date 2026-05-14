"""Wall-clock benchmark entrypoint.

Reproduces (or replays) the 10-seed, 64-A100 numbers from Tables 3-10.

Usage::

    # On a 64-A100 cluster with ray + vllm + grpcio installed:
    python -m saga.entrypoints.bench_wallclock --mode cluster

    # Anywhere (loads results/paper.yaml):
    python -m saga.entrypoints.bench_wallclock --mode paper

    # Auto (default): cluster mode if available, paper mode otherwise.
    python -m saga.entrypoints.bench_wallclock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from saga.serving.benchmarks import (
    BenchmarkConfig,
    WallClockBenchmark,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "cluster", "paper"),
        default="auto",
        help="cluster: run live on 64 A100s; paper: read results/paper.yaml; "
        "auto: cluster if ray+vllm+cuda available, else paper.",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="restrict to a subset of system presets (default: all 7)",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=None,
        help="restrict to a subset of workloads (default: swe_bench, web_arena)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write a JSON dump of the results to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    cfg = BenchmarkConfig(mode=args.mode)
    if args.systems:
        cfg.systems = tuple(args.systems)
    if args.workloads:
        cfg.workloads = tuple(args.workloads)

    bench = WallClockBenchmark(cfg=cfg)
    results = bench.run()

    print(bench.format(results))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "system": r.system,
                        "workload": r.workload,
                        "n_seeds": r.n_seeds,
                        "tct_mean_s": r.tct_mean_s,
                        "tct_std_s": r.tct_std_s,
                        "memory_utilisation_pct": r.memory_utilisation_pct,
                        "source": r.source,
                    }
                    for r in results
                ],
                f,
                indent=2,
            )
        print(f"wrote {len(results)} rows to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
