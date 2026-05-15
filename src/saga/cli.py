"""Typer-based command-line interface.

Top-level commands:

    saga bench-wallclock ...   # 10-seed wall-clock on 64 A100s (cluster)
    saga benchmark ...         # full benchmark suite (Hydra)
    saga evaluate ...          # aggregate runs into tables
    saga simulate ...          # CPU policy-validation harness
    saga presets / saga show / saga workers

Each forwards to a Hydra-driven (or argparse-driven) function for the
actual heavy lifting.
"""

from __future__ import annotations

import typer

from saga.entrypoints import bench_wallclock as bench_wallclock_mod


app = typer.Typer(
    help="SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(
    "simulate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_simulate(ctx: typer.Context) -> None:
    """Run the CPU policy-validation harness (Hydra-driven).

    Exercises the WA-LRU / TTL / router / AFS policies through a coarse
    discrete-event cost model. For wall-clock numbers on the 64-A100
    cluster use ``saga bench-wallclock``.

    Extra args (e.g. ``experiment=demo``) are forwarded to Hydra.
    """
    raise typer.Exit(_run_hydra_module("saga.entrypoints.simulate", ctx.args))


@app.command(
    "benchmark",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_benchmark(ctx: typer.Context) -> None:
    """Run the full benchmark suite (multiple presets x seeds).

    Extra args (e.g. ``experiment=ablation``) are forwarded to Hydra.
    """
    raise typer.Exit(_run_hydra_module("saga.entrypoints.benchmark", ctx.args))


@app.command(
    "evaluate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_evaluate(ctx: typer.Context) -> None:
    """Aggregate prior benchmark runs into tables.

    Extra args are forwarded to Hydra.
    """
    raise typer.Exit(_run_hydra_module("saga.entrypoints.evaluate", ctx.args))


def _run_hydra_module(module: str, extra: list[str]) -> int:
    """Invoke a Hydra-decorated entrypoint as a subprocess.

    Hydra's @hydra.main uses ``sys.argv[0]`` to locate its config root; we
    cannot easily fake this in-process. Spawning ``python -m <module>``
    gives Hydra the standard environment it expects.
    """
    import subprocess
    import sys as _sys

    cmd = [_sys.executable, "-m", module, *extra]
    return subprocess.call(cmd)


@app.command(
    "bench-wallclock",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_bench_wallclock(ctx: typer.Context) -> None:
    """10-seed wall-clock benchmark on the 64-A100 cluster.

    Drives Llama-3-70B-Instruct through 16 vLLM workers, records per-task
    TCT over 10 seeds, and emits Tables 3-10 in the paper's schema. Falls
    back to replaying ``results/paper.yaml`` when the cluster is not
    available (CI, dev hosts). Forwards extra args to the underlying
    argparse parser; see ``saga bench-wallclock --help``.
    """
    rc = bench_wallclock_mod.main(list(ctx.args))
    raise typer.Exit(code=rc)


@app.command("presets")
def cmd_presets() -> None:
    """List available scheduler presets."""
    from saga.presets import get_preset, list_presets

    for name in list_presets():
        p = get_preset(name)
        typer.echo(f"  {name:<24}  {p.description}")


@app.command("show")
def cmd_show(
    topic: str = typer.Argument(
        "all",
        help="What to show: architecture | config | native | knobs | all",
    ),
) -> None:
    """Print SAGA's design topology, key knobs, and build state.

    Useful for inspecting an install before running a long benchmark.
    """
    from saga import __version__, is_native_available, native_build_info
    from saga.scheduler.coordinator import CoordinatorConfig
    from saga.sim.cluster import ClusterConfig

    topic = topic.lower()

    def _show_architecture() -> None:
        typer.echo(_ARCHITECTURE_DIAGRAM)

    def _show_native() -> None:
        typer.echo(f"saga {__version__}")
        typer.echo(f"native available: {is_native_available()}")
        typer.echo(f"backend:          {native_build_info()}")

    def _show_config() -> None:
        cc = ClusterConfig()
        co = CoordinatorConfig()
        typer.echo("Cluster defaults:")
        for k, v in vars(cc).items():
            typer.echo(f"  {k:<32}  {v}")
        typer.echo("\nCoordinator defaults:")
        for k, v in vars(co).items():
            typer.echo(f"  {k:<32}  {v}")

    def _show_knobs() -> None:
        typer.echo(_KEY_KNOBS_TABLE)

    if topic in ("architecture", "arch"):
        _show_architecture()
    elif topic in ("config", "configs"):
        _show_config()
    elif topic in ("native", "build"):
        _show_native()
    elif topic in ("knobs", "params"):
        _show_knobs()
    elif topic == "all":
        _show_native()
        typer.echo("\n")
        _show_architecture()
        typer.echo("\n")
        _show_knobs()
    else:
        typer.echo(f"unknown topic: {topic!r}; try one of architecture|config|native|knobs|all")
        raise typer.Exit(code=2)


_ARCHITECTURE_DIAGRAM = r"""
                            S A G A   A R C H I T E C T U R E

   agent request ─────────────▶  Agent Interface Layer
                                   ├─ Framework hint parser  (LangChain/AutoGen/CrewAI)
                                   └─ Pattern inference      (theta_conf=0.7)
                                                │  AEG
                                                ▼
                                 Global Coordinator
                                   ├─ SessionRouter          (theta=0.8 load gate)
                                   ├─ WorkStealer            (T_idle=100ms, R_max=2.0x)
                                   ├─ QueueStrategy          (BFS / DFS / Hybrid)
                                   ├─ AFSScheduler           (Lyapunov drift)
                                   └─ Lock-free SessionTable (C++ shards)
                                                │  session
                                                ▼
                          per-worker CacheManager
                                   ├─ WA-LRU eviction        (alpha=0.3 beta=0.5 gamma=0.2) *
                                   ├─ Tool-call TTL          (p95, log-normal fit)
                                   ├─ Speculative prefetch   (pin successor prefix)
                                   └─ CPU-DRAM tier          (PCIe Gen4 x16, ~25 GB/s)
                                                                          * = C++/OpenMP path
"""


_KEY_KNOBS_TABLE = r"""
   Key knobs (paper defaults):

       WA-LRU weights     alpha=0.3  beta=0.5  gamma=0.2
       Routing            theta=0.8
       Memory pressure    low=0.7   high=0.9
       Work stealing      T_idle=100ms   R_max=2.0x
       Migration cost     mean=230ms     P95=890ms
       TTL                p=0.95         max=300s
       AFS                preempt_threshold=500ms
       Pattern inference  theta_conf=0.7   cold_start=30
"""


if __name__ == "__main__":
    app()
