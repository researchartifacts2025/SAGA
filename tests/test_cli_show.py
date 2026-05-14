"""Tests for the ``saga show`` CLI subcommand."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from saga.cli import app


runner = CliRunner()


@pytest.mark.unit
def test_show_all_prints_architecture_and_knobs() -> None:
    result = runner.invoke(app, ["show", "all"])
    assert result.exit_code == 0
    assert "A R C H I T E C T U R E" in result.stdout
    assert "alpha=0.3" in result.stdout
    assert "T_idle=100ms" in result.stdout


@pytest.mark.unit
def test_show_config_prints_defaults() -> None:
    result = runner.invoke(app, ["show", "config"])
    assert result.exit_code == 0
    assert "Cluster defaults" in result.stdout
    assert "Coordinator defaults" in result.stdout
    assert "walru_alpha" in result.stdout


@pytest.mark.unit
def test_show_native_prints_build_state() -> None:
    result = runner.invoke(app, ["show", "native"])
    assert result.exit_code == 0
    assert "native available" in result.stdout
    assert "backend:" in result.stdout


@pytest.mark.unit
def test_show_unknown_topic_exits_nonzero() -> None:
    result = runner.invoke(app, ["show", "nonsense"])
    assert result.exit_code != 0


@pytest.mark.unit
def test_presets_command_lists_thirteen() -> None:
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0
    assert "saga" in result.stdout
    assert "vllm_apc" in result.stdout
    assert "saga_no_walru" in result.stdout
