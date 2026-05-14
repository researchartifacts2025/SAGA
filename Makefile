# SAGA developer commands.
# Targets are documented; run `make` (no arg) to see help.

.PHONY: help install install-dev demo simulate benchmark evaluate \
        tables ablation fairness sensitivity competitive \
        test test-fast lint format typecheck check clean

.DEFAULT_GOAL := help

PYTHON ?= python
PIP    ?= pip

help: ## Show this help message
	@printf "\nSAGA development commands\n"
	@printf "==========================\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@printf "\n"

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development extras (dev + docs)
	$(PIP) install -e ".[all]"

native: ## Build the optional C++ acceleration extension via pybind11
	$(PIP) install "pybind11>=2.11"
	$(PYTHON) setup_native.py build_ext --inplace
	$(PYTHON) -c "from saga import native_build_info; print(native_build_info())"

native-cmake: ## Build the native extension via CMake (with -march=native)
	cmake -S . -B build -DSAGA_NATIVE_TUNE=ON
	cmake --build build --config Release -j

native-clean: ## Remove compiled native artifacts
	rm -rf build/ src/saga/_native*.so src/saga/_native*.pyd src/saga/_native*.dylib

demo: ## Run a 60-second quick demo simulation
	$(PYTHON) -m saga.entrypoints.simulate experiment=demo

simulate: ## Run a single simulation with default config
	$(PYTHON) -m saga.entrypoints.simulate

benchmark: ## Run the full end-to-end benchmark suite
	$(PYTHON) -m saga.entrypoints.benchmark

evaluate: ## Aggregate prior runs into result tables
	$(PYTHON) -m saga.entrypoints.evaluate

tables: ## Compute the end-to-end results comparison
	$(PYTHON) -m saga.entrypoints.benchmark experiment=e2e_main

ablation: ## Run the ablation sweep
	$(PYTHON) -m saga.entrypoints.benchmark experiment=ablation

fairness: ## Run the multi-tenant fairness study
	$(PYTHON) -m saga.entrypoints.benchmark experiment=fairness

sensitivity: ## Run parameter sensitivity analysis
	$(PYTHON) -m saga.entrypoints.benchmark experiment=sensitivity

competitive: ## Compute competitive ratios vs Belady oracle
	$(PYTHON) -m saga.entrypoints.benchmark experiment=competitive

bfsdfs: ## BFS / DFS / Hybrid execution-strategy tradeoff
	$(PYTHON) -m saga.entrypoints.benchmark experiment=bfsdfs

tool-variance: ## Tool-latency variance (CV) sweep
	$(PYTHON) -m saga.entrypoints.benchmark experiment=tool_variance

all-tables: tables ablation fairness competitive sensitivity bfsdfs tool-variance ## Run every result table

bench-native: ## Microbenchmark: native C++ vs pure Python on the hot eviction paths
	$(PYTHON) -m saga.entrypoints.bench_native

show: ## Print architecture, key knobs, and native build state
	$(PYTHON) -m saga.cli show all

test: ## Run all tests
	pytest tests/ -v

test-fast: ## Run fast unit tests only
	pytest tests/ -v -m "not slow and not integration"

lint: ## Run ruff linter
	ruff check src tests
	ruff format --check src tests

format: ## Auto-format the codebase
	ruff check --fix src tests
	ruff format src tests

typecheck: ## Run mypy
	mypy src/saga --ignore-missing-imports

check: lint typecheck test-fast ## Run lint + typecheck + fast tests

clean: ## Remove caches and build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .hypothesis/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
