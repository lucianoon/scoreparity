"""Structured outputs: one JSON object per row (for example, fields extracted by an LLM).

Each declared field is compared as an output of its own type (`score`, `label` or `labels`)
with its own gates, and the verdict is the intersection of every field and the top-level gates:

- coverage: ids dropped by the candidate (checked once, for whole rows);
- nonfinite: rows where exactly one version produced a valid document;
- schema_valid_rate: one-sided upper bound of the candidate's share of invalid documents (not
  JSON, not an object, or a required field missing or null).

Field gates compare the rows where *both* documents are valid: an invalid document is judged
once, by the top-level gates, instead of failing every field. Within valid documents, an
optional field that is missing is a missing output for that field (for a `label` field with
`normalize.allowed`, the invalid class). Score fields accept JSON numbers only: text such as
"12.5" or a boolean is not a number and counts as missing, so a format change is visible.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from scoreparity import gates
from scoreparity.align import Alignment, _check_ids
from scoreparity.config import FIELD_COLUMN, ParityConfig, field_config
from scoreparity.errors import InputError
from scoreparity.gates import GateResult, _unverifiable
from scoreparity.gates_categorical import rows_needed
from scoreparity.stats_categorical import clopper_pearson


def _document(cell: object) -> dict[str, Any] | None:
    """The JSON object in a cell, or None when the cell is not one."""
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, (bytes, bytearray)):
        cell = cell.decode("utf-8", errors="replace")
    if not isinstance(cell, str):
        return None
    try:
        value = json.loads(cell)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _score_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _label_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()  # JSON true/false
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _labels_value(value: object) -> object:
    if value is None or isinstance(value, (str, list)):
        return value
    return _label_value(value)


_CONVERT: dict[str, Any] = {"score": _score_value, "label": _label_value, "labels": _labels_value}


class StructuredOutput:
    name = "structured"

    def compare(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        cfg: ParityConfig,
        context: pd.DataFrame | None,
    ) -> tuple[list[GateResult], dict[str, Any]]:
        from scoreparity.outputs import get_output_kind

        cid, columns = cfg.columns.id, cfg.columns
        _check_ids(reference, candidate, cid, [columns.reference], [columns.candidate])
        fields = cfg.output.fields
        required = [name for name, spec in fields.items() if spec.required]

        def parse(table: pd.DataFrame, column: str) -> tuple[list[Any], np.ndarray]:
            docs = [_document(v) for v in table[column]]
            valid = np.array(
                [d is not None and all(d.get(f) is not None for f in required) for d in docs],
                dtype=bool,
            )
            return docs, valid

        ref_docs, ref_valid = parse(reference, columns.reference)
        cand_docs, cand_valid = parse(candidate, columns.candidate)

        ref_ids = pd.Index(reference[cid])
        cand_ids = pd.Index(candidate[cid])
        in_cand = ref_ids.isin(cand_ids)
        missing = ref_ids[~in_cand]
        extra = cand_ids[~cand_ids.isin(ref_ids)]
        # Validity of matched rows, in reference order.
        cand_pos = pd.Series(np.arange(len(cand_ids)), index=cand_ids)
        matched_cand = cand_pos.reindex(ref_ids[in_cand]).to_numpy()
        r_ok, c_ok = ref_valid[in_cand], cand_valid[matched_cand]
        top = Alignment(
            frame=pd.DataFrame({cid: ref_ids[in_cand][r_ok & c_ok]}),
            n_reference=len(reference),
            n_candidate=len(candidate),
            n_missing=len(missing),
            n_extra=len(extra),
            n_nonfinite_mismatch=int((r_ok != c_ok).sum()),
            n_nonfinite_both=int((~r_ok & ~c_ok).sum()),
            missing_examples=tuple(missing.astype(str)[:5]),
            extra_examples=tuple(extra.astype(str)[:5]),
        )
        results = gates.common_gates(top, cfg, missing_text="a valid document")
        n_matched = int(in_cand.sum())
        if cfg.gates.schema_valid_rate:
            gate = cfg.gates.schema_valid_rate
            if n_matched == 0:
                results.append(_unverifiable("schema_valid_rate", gate.max, "no matched rows"))
            else:
                k = int((~c_ok).sum())
                lower, upper = clopper_pearson(k, n_matched, cfg.alpha)
                results.append(
                    GateResult(
                        "schema_valid_rate",
                        upper <= gate.max,
                        k / n_matched,
                        gate.max,
                        "share of candidate rows that are not a valid document (not JSON, not an "
                        f"object, or a required field missing); its {100 * (1 - cfg.alpha):g}% "
                        "upper bound must stay below the threshold",
                        {
                            "ci": [lower, upper],
                            "candidate_invalid": k,
                            "reference_invalid": int((~r_ok).sum()),
                            "rows": n_matched,
                            "rows_needed": rows_needed(gate.max, cfg.alpha),
                        },
                    )
                )

        summary: dict[str, Any] = {
            "n_reference": top.n_reference,
            "n_candidate": top.n_candidate,
            "n_matched_finite": top.n_matched,
            "n_missing": top.n_missing,
            "n_extra": top.n_extra,
            "n_nonfinite_mismatch": top.n_nonfinite_mismatch,
            "n_nonfinite_both": top.n_nonfinite_both,
            "schema_valid": {
                "reference": float(ref_valid.mean()) if len(ref_valid) else 0.0,
                "candidate": float(cand_valid.mean()) if len(cand_valid) else 0.0,
            },
            "fields": {},
        }
        invalid_ids = pd.Index(reference[cid][~ref_valid]).union(
            pd.Index(candidate[cid][~cand_valid])
        )
        ref_keep = ~reference[cid].isin(invalid_ids).to_numpy()
        cand_keep = ~candidate[cid].isin(invalid_ids).to_numpy()
        for name, spec in fields.items():
            convert = _CONVERT[spec.type]
            # Kept rows are valid documents (never None): rows of invalid ids were dropped.
            ref_values = [
                convert(d.get(name)) for d, ok in zip(ref_docs, ref_keep, strict=True) if ok
            ]
            cand_values = [
                convert(d.get(name)) for d, ok in zip(cand_docs, cand_keep, strict=True) if ok
            ]
            # Segments and truth come from the reference table, or from `context` when given
            # (as for every other output type).
            keep = [cid] if context is not None else [cid, *cfg.segments]
            if context is None and spec.truth:
                keep.append(spec.truth)
            missing_cols = [c for c in keep if c not in reference.columns]
            if missing_cols:
                raise InputError(f"reference: missing columns {missing_cols} (field {name!r})")
            ref_table = reference.loc[ref_keep, list(dict.fromkeys(keep))].copy()
            ref_table[FIELD_COLUMN] = pd.Series(
                ref_values,
                index=ref_table.index,
                dtype=np.float64 if spec.type == "score" else object,
            )
            cand_table = pd.DataFrame({cid: candidate[cid].to_numpy()[cand_keep]})
            cand_table[FIELD_COLUMN] = pd.Series(
                cand_values, dtype=np.float64 if spec.type == "score" else object
            )
            sub_cfg = field_config(cfg, name)
            try:
                sub_results, sub_summary = get_output_kind(spec.type).compare(
                    ref_table, cand_table, sub_cfg, context
                )
            except InputError as exc:
                raise InputError(f"field {name!r}: {exc}") from None
            results += [dataclasses.replace(r, field=name) for r in sub_results]
            summary["fields"][name] = {"type": spec.type, **sub_summary}
        return results, summary
