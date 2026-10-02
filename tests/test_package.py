from __future__ import annotations

import subprocess
import sys

import pytest

import scoreparity
from scoreparity.cli import EXIT_ERROR, main


def test_version_is_pep440() -> None:
    from packaging.version import Version

    assert Version(scoreparity.__version__)


def test_cli_without_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_ERROR
    assert "usage" in capsys.readouterr().err


def test_cli_version_flag() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "scoreparity.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == f"scoreparity {scoreparity.__version__}"
