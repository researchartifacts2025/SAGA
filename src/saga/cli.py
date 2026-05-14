"""Typer-based command-line interface.

Three top-level commands wrap the entry-point modules:

    saga simulate ...
    saga benchmark ...
    saga evaluate ...

Each forwards to a Hydra-driven function for the actual heavy lifting.
"""

from __future__ import annotations

import typer

from saga.entrypoints import benchmark as benchmark_mod
from saga.entrypoints import evaluate as evaluate_mod
from saga.entrypoints import simulate as simulate_mod


app = typer.Typer(
    help="SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("simulate")
def cmd_simulate() -> None:
    """Run a single simulator pass (Hydra-driven)."""
    simulate_mod.main()


@app.command("benchmark")
def cmd_benchmark() -> None:
    """Run the full benchmark suite (multiple presets x seeds)."""
    benchmark_mod.main()


@app.command("evaluate")
def cmd_evaluate() -> None:
    """Aggregate prior benchmark runs into tables."""
    evaluate_mod.main()


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
