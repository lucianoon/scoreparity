"""Probability vectors (one probability column per class).

- Per-class probability gates (max_abs_diff, quantile_abs_diff, mean_diff_equivalence) reuse the
  continuous implementation on each class column; a gate passes only if it passes for every
  class (intersection-union, no multiplicity correction needed).
- tv_distance compares whole vectors: TV = 1/2 * sum_k |p_ref_k - p_cand_k|, between 0 and 1.
- Class gates (label_agreement, transitions, ...) apply to the argmax class of each row.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scoreparity import gates, gates_categorical
from scoreparity.align import CAND, CAND_P, DIFF, REF, REF_P, Alignment, align_probabilities
from scoreparity.config import Gates, Output, ParityConfig
from scoreparity.gates import GateResult, _decade_histogram, _unverifiable

_PER_CLASS = ("max_abs_diff", "quantile_abs_diff", "mean_diff_equivalence")


def _class_alignment(alignment: Alignment, cls: str) -> Alignment:
    frame = alignment.frame
    keep = [c for c in frame.columns if not c.startswith((REF_P, CAND_P)) and c not in (REF, CAND)]
    per = frame[keep].copy()
    per[REF] = frame[REF_P + cls].to_numpy(np.float64)
    per[CAND] = frame[CAND_P + cls].to_numpy(np.float64)
    per[DIFF] = per[CAND] - per[REF]
    return dataclasses.replace(alignment, frame=per, classes=())


def _per_class_gates(alignment: Alignment, cfg: ParityConfig) -> list[GateResult]:
    enabled = {name: getattr(cfg.gates, name) for name in _PER_CLASS if getattr(cfg.gates, name)}
    if not enabled:
        return []
    score_cfg = dataclasses.replace(cfg, output=Output(), gates=Gates(None, None, **enabled))
    by_gate: dict[str, list[tuple[str, GateResult]]] = {name: [] for name in enabled}
    for cls in alignment.classes:
        for res in gates.evaluate(_class_alignment(alignment, cls), score_cfg):
            by_gate[res.name].append((cls, res))
    results = []
    for name, items in by_gate.items():
        failing = [cls for cls, r in items if not r.passed]
        values = [r.value for _, r in items if r.value is not None]
        worst = max(values, key=abs) if values else None
        segs = [f"{cls}:{s}" for cls, r in items for s in r.details.get("failing_segments", [])]
        results.append(
            GateResult(
                name,
                not failing,
                worst,
                items[0][1].threshold if items else None,
                items[0][1].description + ", for every class" if items else name,
                {
                    "failing_classes": failing,
                    "failing_segments": segs,
                    "per_class": {
                        cls: {"value": r.value, "passed": r.passed, "ci": r.details.get("ci")}
                        for cls, r in items
                    },
                },
            )
        )
    return results


def _tv(alignment: Alignment) -> np.ndarray:
    frame, classes = alignment.frame, alignment.classes
    ref = frame[[REF_P + c for c in classes]].to_numpy(np.float64)
    cand = frame[[CAND_P + c for c in classes]].to_numpy(np.float64)
    return np.asarray(0.5 * np.abs(cand - ref).sum(axis=1))


class ProbabilitiesOutput:
    name = "probabilities"

    def compare(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        cfg: ParityConfig,
        context: pd.DataFrame | None,
    ) -> tuple[list[GateResult], dict[str, Any]]:
        alignment = align_probabilities(
            reference,
            candidate,
            cfg.columns,
            cfg.segments,
            context,
            cfg.output.prob_prefix,
            cfg.output.prob_sum_tolerance,
        )
        results = gates.common_gates(alignment, cfg)
        results += _per_class_gates(alignment, cfg)
        if cfg.gates.tv_distance:
            gate = cfg.gates.tv_distance
            if alignment.n_matched == 0:
                results.append(_unverifiable("tv_distance", gate.max, "no finite matched rows"))
            else:
                tv = _tv(alignment)
                value = float(np.quantile(tv, gate.q, method="higher"))
                results.append(
                    GateResult(
                        "tv_distance",
                        value <= gate.max,
                        value,
                        gate.max,
                        f"q{gate.q:g} of the total variation distance between probability vectors",
                        {"q": gate.q, "max_observed": float(tv.max())},
                    )
                )
        results += gates_categorical.evaluate(alignment, cfg)
        return results, self.summary(alignment)

    @staticmethod
    def summary(alignment: Alignment) -> dict[str, Any]:
        out = gates_categorical.summary(alignment)
        if alignment.n_matched:
            frame, classes = alignment.frame, alignment.classes
            ref = frame[[REF_P + c for c in classes]].to_numpy(np.float64)
            cand = frame[[CAND_P + c for c in classes]].to_numpy(np.float64)
            row_max = np.abs(cand - ref).max(axis=1)
            tv = _tv(alignment)
            out["abs_diff_quantiles"] = {
                q: float(np.quantile(row_max, float(q), method="higher"))
                for q in ("0.5", "0.9", "0.99", "0.999", "1.0")
            }
            out["tv_quantiles"] = {
                q: float(np.quantile(tv, float(q), method="higher"))
                for q in ("0.5", "0.9", "0.99", "1.0")
            }
            out["identical_share"] = float((row_max == 0).mean())
            out["abs_diff_histogram"] = _decade_histogram(row_max)
        return out
