"""Join reference and candidate scores on a record id, refusing ambiguous inputs.

Problems that make the comparison meaningless (missing columns, duplicate ids, non-numeric
scores) raise `InputError`. Problems that are *behaviour* of the candidate (ids it dropped,
NaN where the reference had a number) are measured and handed to the gates instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scoreparity.config import Columns
from scoreparity.errors import InputError

REF = "__ref"
CAND = "__cand"
DIFF = "__diff"
LABEL = "__label"


@dataclass(frozen=True)
class Alignment:
    """Matched rows plus the bookkeeping the coverage and non-finite gates need."""

    frame: pd.DataFrame  # one row per matched id: REF, CAND, DIFF, [LABEL], segment columns
    n_reference: int
    n_candidate: int
    n_missing: int  # ids in the reference but not in the candidate
    n_extra: int  # ids in the candidate but not in the reference
    n_nonfinite_mismatch: int  # rows where exactly one side is NaN/inf
    n_nonfinite_both: int  # rows where both sides are NaN/inf (agreement, excluded from stats)
    missing_examples: tuple[str, ...]
    extra_examples: tuple[str, ...]

    @property
    def n_matched(self) -> int:
        return len(self.frame)

    @property
    def coverage(self) -> float:
        return (self.n_reference - self.n_missing) / self.n_reference


def _require_columns(df: pd.DataFrame, columns: list[str], side: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputError(f"{side}: missing columns {missing}; available: {list(df.columns)}")


def _numeric(series: pd.Series, side: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise InputError(f"{side}: score column {series.name!r} is not numeric ({series.dtype})")
    return series.astype(np.float64)


def _check_unique(ids: pd.Series, side: str) -> None:
    dup = ids[ids.duplicated(keep=False)]
    if len(dup):
        examples = list(dict.fromkeys(dup.astype(str)))[:5]
        raise InputError(f"{side}: {dup.nunique()} duplicated ids, e.g. {examples}")
    if ids.isna().any():
        raise InputError(f"{side}: id column has missing values")


def align(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Columns,
    segments: tuple[str, ...] = (),
    context: pd.DataFrame | None = None,
) -> Alignment:
    """Align the two score tables.

    Labels and segment columns may live in the reference table or in a separate `context`
    table keyed by the same id (useful when the score files only contain id and score).
    """
    cid = columns.id
    _require_columns(reference, [cid, columns.reference], "reference")
    _require_columns(candidate, [cid, columns.candidate], "candidate")
    if len(reference) == 0:
        raise InputError("reference: no rows to compare")
    _check_unique(reference[cid], "reference")
    _check_unique(candidate[cid], "candidate")
    if reference[cid].dtype != candidate[cid].dtype:
        raise InputError(
            f"id column {cid!r} has different types: reference {reference[cid].dtype}, "
            f"candidate {candidate[cid].dtype}; convert them before comparing"
        )

    ref = pd.DataFrame(
        {cid: reference[cid], REF: _numeric(reference[columns.reference], "reference")}
    )
    cand = pd.DataFrame(
        {cid: candidate[cid], CAND: _numeric(candidate[columns.candidate], "candidate")}
    )

    extra_cols = [c for c in (columns.label, *segments) if c is not None]
    if extra_cols:
        source = reference if context is None else context
        side = "reference" if context is None else "context"
        _require_columns(source, [cid, *extra_cols], side)
        if context is not None:
            _check_unique(context[cid], "context")
        ref = ref.merge(source[[cid, *extra_cols]], on=cid, how="left", validate="one_to_one")
        if columns.label is not None:
            ref = ref.rename(columns={columns.label: LABEL})

    merged = ref.merge(cand, on=cid, how="outer", indicator=True, validate="one_to_one")
    missing = merged.loc[merged["_merge"] == "left_only", cid]
    extra = merged.loc[merged["_merge"] == "right_only", cid]
    matched = merged.loc[merged["_merge"] == "both"].drop(columns="_merge")

    ref_ok = np.isfinite(matched[REF].to_numpy())
    cand_ok = np.isfinite(matched[CAND].to_numpy())
    both_ok = ref_ok & cand_ok
    frame = matched.loc[both_ok].copy()
    frame[DIFF] = frame[CAND] - frame[REF]

    if columns.label is not None:
        labels = frame[LABEL]
        if labels.isna().any():
            raise InputError(f"label column {columns.label!r} has missing values in matched rows")
        values = set(pd.unique(labels))
        if not values <= {0, 1}:  # True/False compare equal to 1/0
            found = sorted(map(str, values))[:5]
            raise InputError(f"label column {columns.label!r} must be binary 0/1, got {found}")
        frame[LABEL] = labels.astype(bool)
    for seg in segments:
        frame[seg] = frame[seg].astype("string").fillna("<missing>")

    return Alignment(
        frame=frame.reset_index(drop=True),
        n_reference=len(reference),
        n_candidate=len(candidate),
        n_missing=len(missing),
        n_extra=len(extra),
        n_nonfinite_mismatch=int((ref_ok != cand_ok).sum()),
        n_nonfinite_both=int((~ref_ok & ~cand_ok).sum()),
        missing_examples=tuple(missing.astype(str).head(5)),
        extra_examples=tuple(extra.astype(str).head(5)),
    )
