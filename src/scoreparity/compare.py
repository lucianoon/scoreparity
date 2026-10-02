"""Public entry point: compare two score tables under a parity configuration."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from scoreparity.config import ParityConfig
from scoreparity.errors import InputError
from scoreparity.outputs import get_output_kind
from scoreparity.report import Report


def _read_csv(path: Path) -> pd.DataFrame:
    # pandas' default float parser is not round-trip exact: about a quarter of float64 values
    # come back 1 ulp off. A parity tool must never invent differences, so use "round_trip".
    return pd.read_csv(path, float_precision="round_trip")


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": _read_csv,
    ".parquet": _read_parquet,
    ".pq": _read_parquet,
}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet score table."""
    p = Path(path)
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise InputError(f"{p}: unsupported format {p.suffix!r} (use .csv or .parquet)")
    if not p.exists():
        raise InputError(f"{p}: file not found")
    try:
        return reader(p)
    except ImportError as exc:
        raise InputError(
            f"{p}: reading Parquet needs pyarrow (pip install scoreparity[parquet])"
        ) from exc


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    config: ParityConfig | None = None,
    context: pd.DataFrame | None = None,
    inputs: dict[str, Any] | None = None,
) -> Report:
    """Compare reference and candidate scores.

    Raises `InputError` only when the tables cannot be compared at all. A candidate that is not
    equivalent produces a report with `report.passed == False`.
    """
    cfg = config or ParityConfig()
    gate_results, summary = get_output_kind(cfg.output.type).compare(
        reference, candidate, cfg, context
    )
    return Report(gates=gate_results, summary=summary, config=cfg.to_dict(), inputs=inputs or {})


def compare_files(
    reference: str | Path,
    candidate: str | Path,
    config: ParityConfig | None = None,
    context: str | Path | None = None,
) -> Report:
    """Like `compare`, reading tables from disk and recording their sha256 in the report."""
    paths = {"reference": reference, "candidate": candidate}
    if context is not None:
        paths["context"] = context
    inputs = {
        name: {"path": str(p), "sha256": _sha256(p)}
        for name, p in paths.items()
        if Path(p).exists()
    }
    return compare(
        read_table(reference),
        read_table(candidate),
        config,
        context=read_table(context) if context is not None else None,
        inputs=inputs,
    )
