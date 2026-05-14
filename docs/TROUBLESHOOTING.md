# Troubleshooting

## Installation

### `ERROR: ... requires a different Python: 3.X.Y not in '>=3.10,<3.14'`

SAGA targets Python 3.10–3.12. The simplest fix is a fresh environment:

```bash
conda create -n saga python=3.11 -y
conda activate saga
pip install -e .
```

### NumPy / SciPy build errors on first install

If `pip` insists on building NumPy/SciPy from source, install the binary
wheels first:

```bash
pip install --only-binary=:all: numpy scipy pandas
pip install -e .
```

## Runtime

### Simulation completes but reports zero tasks

The horizon was too short. Increase it:

```bash
python -m saga.entrypoints.simulate horizon_ms=3600000
```

### Hydra error: `MissingMandatoryValue while resolving interpolation`

Pass the missing key, e.g.

```bash
python -m saga.entrypoints.simulate experiment=demo workload=swe_bench
```

The default `config.yaml` requires a workload and a scheduler preset.

### Path errors on Windows

Hydra's run-dir defaults to `runs/${now:%Y-%m-%d}/${now:%H-%M-%S}` which is
Windows-safe. If a custom `hydra.run.dir` contains a colon, quote it.

## Tests

### `pytest` fails with `Unknown config option: timeout`

The `timeout` option is a pytest-timeout plugin setting; install
`pytest-timeout` or remove the `timeout =` line from `pyproject.toml` if
you do not need it. The warning is benign.

### `test_engine.py::test_engine_deterministic_under_seed` fails

The engine is seeded but the cache manager's `regenerated_tokens` counter
can vary if RNG draws happen in a different order. The test pins both the
workload seed and engine seed and constructs fresh templates each call;
if it fails, file an issue with the diff in counters.

## Performance

### A benchmark sweep takes too long

Drop to `cluster=single_node` and shrink the task count:

```bash
python -m saga.entrypoints.benchmark experiment=e2e_main \
    +experiment.workloads.0.n_tasks=30 \
    cluster=single_node
```

A full sweep on a laptop with default sizes runs in 5-10 minutes; with
`single_node` and 30 tasks per workload it is under a minute.
