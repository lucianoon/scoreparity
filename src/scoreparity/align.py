"""Join reference and candidate outputs on a record id, refusing ambiguous inputs.

Problems that make the comparison meaningless (missing columns, duplicate ids, wrong types)
raise `InputError`. Problems that are *behaviour* of the candidate (ids it dropped, a missing or
non-finite output where the reference had one) are measured and handed to the gates instead.

Three output kinds share the same join:
- `align`: one numeric score per row;
- `align_labels`: one class label per row;
- `align_probabilities`: one probability per class per row (columns sharing a prefix).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scoreparity.config import Columns, Normalize
from scoreparity.errors import InputError

REF = "__ref"
CAND = "__cand"
DIFF = "__diff"
LABEL = "__label"  # binary ground truth (score outputs, AUC)
TRUTH = "__truth"  # class ground truth (categorical outputs)
REF_P = "__ref_p::"  # + class name, probabilities outputs
CAND_P = "__cand_p::"


@dataclass(frozen=True)
class Alignment:
    """Matched rows plus the bookkeeping the coverage and non-finite gates need."""

    frame: pd.DataFrame  # one row per matched id with a usable output on both sides
    n_reference: int
    n_candidate: int
    n_missing: int  # ids in the reference but not in the candidate
    n_extra: int  # ids in the candidate but not in the reference
    n_nonfinite_mismatch: int  # rows where exactly one side has no usable output
    n_nonfinite_both: int  # rows where neither side has one (agreement, excluded from stats)
    missing_examples: tuple[str, ...]
    extra_examples: tuple[str, ...]
    classes: tuple[str, ...] = field(default=())  # categorical outputs only

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


def _check_ids(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    cid: str,
    ref_cols: list[str],
    cand_cols: list[str],
) -> None:
    _require_columns(reference, [cid, *ref_cols], "reference")
    _require_columns(candidate, [cid, *cand_cols], "candidate")
    if len(reference) == 0:
        raise InputError("reference: no rows to compare")
    _check_unique(reference[cid], "reference")
    _check_unique(candidate[cid], "candidate")
    if reference[cid].dtype != candidate[cid].dtype:
        raise InputError(
            f"id column {cid!r} has different types: reference {reference[cid].dtype}, "
            f"candidate {candidate[cid].dtype}; convert them before comparing"
        )


def _join(
    ref: pd.DataFrame,
    cand: pd.DataFrame,
    cid: str,
    reference: pd.DataFrame,
    context: pd.DataFrame | None,
    extra_cols: list[str],
    renames: dict[str, str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Attach labels/segments, outer-join on the id; return (matched, missing ids, extra ids)."""
    if extra_cols:
        source = reference if context is None else context
        side = "reference" if context is None else "context"
        _require_columns(source, [cid, *extra_cols], side)
        if context is not None:
            _check_unique(context[cid], "context")
        ref = ref.merge(source[[cid, *extra_cols]], on=cid, how="left", validate="one_to_one")
        ref = ref.rename(columns=renames)
    merged = ref.merge(cand, on=cid, how="outer", indicator=True, validate="one_to_one")
    missing = merged.loc[merged["_merge"] == "left_only", cid]
    extra = merged.loc[merged["_merge"] == "right_only", cid]
    matched = merged.loc[merged["_merge"] == "both"].drop(columns="_merge")
    return matched, missing, extra


def _finish(
    matched: pd.DataFrame,
    ref_ok: np.ndarray,
    cand_ok: np.ndarray,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    missing: pd.Series,
    extra: pd.Series,
    segments: tuple[str, ...],
    classes: tuple[str, ...] = (),
) -> Alignment:
    frame = matched.loc[ref_ok & cand_ok].copy()
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
        classes=classes,
    )


def _class_truth(frame: pd.DataFrame, columns: Columns) -> None:
    if columns.truth is None:
        return
    truth = frame[TRUTH]
    if truth.isna().any():
        raise InputError(f"truth column {columns.truth!r} has missing values in matched rows")
    frame[TRUTH] = truth.astype(str).str.strip()


# --------------------------------------------------------------------------- scores


def align(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Columns,
    segments: tuple[str, ...] = (),
    context: pd.DataFrame | None = None,
) -> Alignment:
    """Align two score tables (one numeric score per row).

    Labels and segment columns may live in the reference table or in a separate `context`
    table keyed by the same id (useful when the score files only contain id and score).
    """
    cid = columns.id
    _check_ids(reference, candidate, cid, [columns.reference], [columns.candidate])
    ref = pd.DataFrame(
        {cid: reference[cid], REF: _numeric(reference[columns.reference], "reference")}
    )
    cand = pd.DataFrame(
        {cid: candidate[cid], CAND: _numeric(candidate[columns.candidate], "candidate")}
    )
    extra_cols = [c for c in (columns.label, *segments) if c is not None]
    renames = {columns.label: LABEL} if columns.label is not None else {}
    matched, missing, extra = _join(ref, cand, cid, reference, context, extra_cols, renames)

    ref_ok = np.isfinite(matched[REF].to_numpy())
    cand_ok = np.isfinite(matched[CAND].to_numpy())
    alignment = _finish(matched, ref_ok, cand_ok, reference, candidate, missing, extra, segments)
    frame = alignment.frame
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
    return alignment


# --------------------------------------------------------------------------- class labels


def _labels(series: pd.Series, norm: Normalize | None = None) -> tuple[pd.Series, np.ndarray]:
    """Normalise labels to stripped strings; return (labels, usable mask).

    With `norm`, labels are also lowercased and mapped as configured; with `norm.allowed`, any
    label outside it, including a missing or empty one, becomes `norm.invalid` and is usable.
    """
    text = series.astype("string").str.strip()
    usable = (text.notna() & (text != "")).to_numpy(dtype=bool)
    text = text.fillna("")
    if norm is None:
        return text, usable
    keys = text.str.lower() if norm.lowercase else text
    mapping = {norm.key(k): norm.key(v) for k, v in norm.map.items()}
    if mapping:
        keys = keys.replace(mapping)
    if not norm.allowed:
        return keys, usable
    canonical = {norm.key(a): a for a in norm.allowed}
    classes = keys.map(canonical).astype("string")  # NA where not allowed (or empty)
    return classes.fillna(norm.invalid), np.ones(len(classes), dtype=bool)


def _normalized_truth(frame: pd.DataFrame, columns: Columns, norm: Normalize | None) -> None:
    _class_truth(frame, columns)
    if columns.truth is None or norm is None:
        return
    truth, _ = _labels(frame[TRUTH], norm)
    if norm.allowed:
        bad = frame.loc[(truth == norm.invalid).to_numpy(), TRUTH]
        if len(bad):
            examples = sorted(set(bad.astype(str)))[:5]
            raise InputError(
                f"truth column {columns.truth!r} has {len(bad)} rows outside "
                f"output.normalize.allowed, e.g. {examples}; ground truth must be a valid class"
            )
    frame[TRUTH] = truth.astype(str).to_numpy()


def align_labels(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Columns,
    segments: tuple[str, ...] = (),
    context: pd.DataFrame | None = None,
    normalize: Normalize | None = None,
) -> Alignment:
    """Align two tables holding one predicted class per row.

    Labels are compared as stripped strings, so 1 and "1" are the same class. A missing or empty
    label on exactly one side counts as a non-finite mismatch (behaviour, not an input error),
    unless `normalize.allowed` is set: then it is the invalid class, like any other answer
    outside the allowed classes.
    """
    cid = columns.id
    _check_ids(reference, candidate, cid, [columns.reference], [columns.candidate])
    ref = pd.DataFrame({cid: reference[cid], REF: reference[columns.reference]})
    cand = pd.DataFrame({cid: candidate[cid], CAND: candidate[columns.candidate]})
    extra_cols = [c for c in (columns.truth, *segments) if c is not None]
    renames = {columns.truth: TRUTH} if columns.truth is not None else {}
    matched, missing, extra = _join(ref, cand, cid, reference, context, extra_cols, renames)

    ref_labels, ref_ok = _labels(matched[REF], normalize)
    cand_labels, cand_ok = _labels(matched[CAND], normalize)
    matched = matched.assign(**{REF: ref_labels.to_numpy(), CAND: cand_labels.to_numpy()})
    alignment = _finish(matched, ref_ok, cand_ok, reference, candidate, missing, extra, segments)
    frame = alignment.frame
    frame[REF] = frame[REF].astype(str)
    frame[CAND] = frame[CAND].astype(str)
    _normalized_truth(frame, columns, normalize)
    observed = set(frame[REF]) | set(frame[CAND])
    if columns.truth is not None:
        observed |= set(frame[TRUTH])
    if normalize is not None:
        observed |= set(normalize.allowed)  # declared classes are reported even when unseen
    return Alignment(**{**alignment.__dict__, "classes": tuple(sorted(observed))})


# --------------------------------------------------------------------------- probabilities


def _prob_columns(df: pd.DataFrame, prefix: str, side: str) -> dict[str, str]:
    cols = {c[len(prefix) :]: c for c in df.columns if isinstance(c, str) and c.startswith(prefix)}
    if len(cols) < 2:
        raise InputError(
            f"{side}: expected at least 2 probability columns with prefix {prefix!r}, "
            f"found {sorted(cols.values())}"
        )
    for cls, col in cols.items():
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise InputError(f"{side}: probability column {col!r} is not numeric")
        if cls == "":
            raise InputError(f"{side}: column {col!r} has an empty class name")
    return cols


def align_probabilities(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Columns,
    segments: tuple[str, ...] = (),
    context: pd.DataFrame | None = None,
    prefix: str = "p_",
    sum_tolerance: float = 1e-3,
) -> Alignment:
    """Align two tables holding one probability column per class (e.g. p_billing, p_cancel).

    Both tables must have the same classes. A row with any non-finite probability on exactly one
    side is a non-finite mismatch. Rows must sum to 1 within `sum_tolerance`, otherwise the
    columns are probably not probabilities and the comparison is refused.
    """
    cid = columns.id
    ref_cols = _prob_columns(reference, prefix, "reference")
    cand_cols = _prob_columns(candidate, prefix, "candidate")
    if set(ref_cols) != set(cand_cols):
        raise InputError(
            f"reference and candidate have different classes: "
            f"only in reference {sorted(set(ref_cols) - set(cand_cols))}, "
            f"only in candidate {sorted(set(cand_cols) - set(ref_cols))}"
        )
    classes = tuple(sorted(ref_cols))
    _check_ids(reference, candidate, cid, [], [])
    ref = pd.DataFrame(
        {
            cid: reference[cid],
            **{REF_P + c: reference[ref_cols[c]].astype(np.float64) for c in classes},
        }
    )
    cand = pd.DataFrame(
        {
            cid: candidate[cid],
            **{CAND_P + c: candidate[cand_cols[c]].astype(np.float64) for c in classes},
        }
    )
    extra_cols = [c for c in (columns.truth, *segments) if c is not None]
    renames = {columns.truth: TRUTH} if columns.truth is not None else {}
    matched, missing, extra = _join(ref, cand, cid, reference, context, extra_cols, renames)

    ref_p = matched[[REF_P + c for c in classes]].to_numpy()
    cand_p = matched[[CAND_P + c for c in classes]].to_numpy()
    ref_ok = np.isfinite(ref_p).all(axis=1)
    cand_ok = np.isfinite(cand_p).all(axis=1)
    for side, probs, ok in (("reference", ref_p, ref_ok), ("candidate", cand_p, cand_ok)):
        bad = ok & (np.abs(probs.sum(axis=1) - 1.0) > sum_tolerance)
        out_of_range = ok & ((probs < 0) | (probs > 1)).any(axis=1)
        if bad.any() or out_of_range.any():
            raise InputError(
                f"{side}: {int(bad.sum())} rows do not sum to 1 (tolerance {sum_tolerance}) and "
                f"{int(out_of_range.sum())} rows have values outside [0, 1]; "
                f"are the {prefix!r} columns probabilities?"
            )
    alignment = _finish(
        matched, ref_ok, cand_ok, reference, candidate, missing, extra, segments, classes
    )
    frame = alignment.frame
    names = np.array(classes, dtype=object)
    frame[REF] = names[frame[[REF_P + c for c in classes]].to_numpy().argmax(axis=1)].astype(str)
    frame[CAND] = names[frame[[CAND_P + c for c in classes]].to_numpy().argmax(axis=1)].astype(str)
    _class_truth(frame, columns)
    if columns.truth is not None:
        unknown = sorted(set(frame[TRUTH]) - set(classes))
        if unknown:
            raise InputError(f"truth has classes without probability columns: {unknown[:5]}")
    return alignment
