"""Gates for multi-label outputs (a set of classes per row).

- label_agreement: the same *set* of classes (shared with single labels);
- transitions: per class, the share of the rows that had it and lost it ("c -> (removed)") and
  the share of the rows that lacked it and gained it ("(added) -> c"), each with its one-sided
  Clopper-Pearson upper bound; classes need min_class_size rows on the side being measured;
- class_prevalence: per class, the paired difference of the share of rows that have it (Tango);
- invalid_rate: rows whose candidate set contains the invalid class;
- quality_difference: accuracy is the exact-set match against the truth (Tango); macro-F1 is the
  mean of the per-class F1 scores, with a bootstrap over the distinct row types.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scoreparity import gates_categorical
from scoreparity.align import CAND, CAND_SET, REF, REF_SET, TRUTH, TRUTH_SET, Alignment
from scoreparity.config import Gates, ParityConfig
from scoreparity.gates import GateResult, _unverifiable
from scoreparity.gates_categorical import rows_needed
from scoreparity.stats_categorical import (
    clopper_pearson,
    macro_f1_sets,
    macro_f1_sets_difference,
    paired_proportion_difference,
)

REMOVED = "(removed)"
ADDED = "(added)"


def indicators(sets: pd.Series, classes: tuple[str, ...]) -> np.ndarray:
    """Rows x classes boolean matrix: does the row's set contain the class?"""
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(sets), len(classes)), dtype=bool)
    for row, labels in enumerate(sets):
        for label in labels:
            out[row, index[label]] = True
    return out


def evaluate(alignment: Alignment, cfg: ParityConfig) -> list[GateResult]:
    g = cfg.gates
    frame, classes = alignment.frame, alignment.classes
    n = len(frame)
    level = f"{100 * (1 - cfg.alpha):g}%"
    two_sided = f"{100 * (1 - 2 * cfg.alpha):g}%"

    # Set agreement and exact-match accuracy only compare the joined sets: reuse the
    # single-label implementation on them.
    quality = g.quality_difference
    accuracy = quality if quality is not None and quality.metric == "accuracy" else None
    shared = Gates(None, None, label_agreement=g.label_agreement, quality_difference=accuracy)
    results = [
        dataclasses.replace(
            r,
            description=r.description.replace("the same class", "the same set of classes"),
        )
        for r in gates_categorical.evaluate(alignment, dataclasses.replace(cfg, gates=shared))
    ]
    ref = indicators(frame[REF_SET], classes) if n else np.zeros((0, len(classes)), bool)
    cand = indicators(frame[CAND_SET], classes) if n else np.zeros((0, len(classes)), bool)

    if g.transitions:
        gate_t = g.transitions
        if n == 0:
            results.append(_unverifiable("transitions", gate_t.max_rate, "no matched rows"))
        else:
            failing: list[str] = []
            underpowered: list[str] = []
            pairs: list[dict[str, Any]] = []
            worst = 0.0
            for j, cls in enumerate(classes):
                limit = gate_t.per_class.get(cls, gate_t.max_rate)
                had, got = ref[:, j], cand[:, j]
                for name, base, moved in (
                    (f"{cls} -> {REMOVED}", had, had & ~got),
                    (f"{ADDED} -> {cls}", ~had, ~had & got),
                ):
                    n_base = int(base.sum())
                    if n_base < cfg.min_class_size:
                        continue
                    count = int(moved.sum())
                    upper = clopper_pearson(count, n_base, cfg.alpha)[1]
                    worst = max(worst, count / n_base)
                    if upper > limit:
                        failing.append(name)
                        if count == 0:
                            underpowered.append(cls)
                    if count:
                        pairs.append(
                            {
                                "from": cls if name.endswith(REMOVED) else ADDED,
                                "to": REMOVED if name.endswith(REMOVED) else cls,
                                "count": count,
                                "rows_of_source": n_base,
                                "rate": count / n_base,
                                "upper": upper,
                            }
                        )
            results.append(
                GateResult(
                    "transitions",
                    not failing,
                    worst,
                    gate_t.max_rate,
                    f"per class, share of the rows that lost it or gained it; {level} upper "
                    "bound (min_class_size rows on the side measured)",
                    {
                        "failing_pairs": failing[:50],
                        "pairs": sorted(pairs, key=lambda p: -p["rate"])[:50],
                        "underpowered_classes": sorted(set(underpowered)),
                        "rows_needed_per_class": rows_needed(gate_t.max_rate, cfg.alpha),
                    },
                )
            )

    if g.class_prevalence:
        gate_p = g.class_prevalence
        if n == 0:
            results.append(_unverifiable("class_prevalence", gate_p.margin, "no matched rows"))
        else:
            per_class: dict[str, Any] = {}
            failing_classes: list[str] = []
            worst_abs = 0.0
            for j, cls in enumerate(classes):
                r, c = ref[:, j], cand[:, j]
                res = paired_proportion_difference(
                    int((r & c).sum()),
                    int((r & ~c).sum()),
                    int((~r & c).sum()),
                    int((~r & ~c).sum()),
                    cfg.alpha,
                )
                ok = -gate_p.margin < res.low and res.high < gate_p.margin
                per_class[cls] = {
                    "reference": float(r.mean()),
                    "candidate": float(c.mean()),
                    "difference": res.estimate,
                    "ci": [res.low, res.high],
                    "passed": ok,
                }
                worst_abs = max(worst_abs, abs(res.estimate))
                if not ok:
                    failing_classes.append(cls)
            results.append(
                GateResult(
                    "class_prevalence",
                    not failing_classes,
                    worst_abs,
                    gate_p.margin,
                    f"{two_sided} CI of the difference in the share of rows having each class "
                    "within ±margin (Tango)",
                    {"failing_classes": failing_classes, "per_class": per_class},
                )
            )

    if g.invalid_rate:
        gate_i = g.invalid_rate
        invalid = cfg.output.normalize.invalid if cfg.output.normalize else ""
        if n == 0:
            results.append(_unverifiable("invalid_rate", gate_i.max, "no matched rows"))
        else:
            k_cand = int(sum(invalid in s for s in frame[CAND_SET]))
            k_ref = int(sum(invalid in s for s in frame[REF_SET]))
            lower, upper = clopper_pearson(k_cand, n, cfg.alpha)
            results.append(
                GateResult(
                    "invalid_rate",
                    upper <= gate_i.max,
                    k_cand / n,
                    gate_i.max,
                    "share of candidate rows with a label outside the allowed classes; its "
                    f"{level} upper bound must stay below the threshold",
                    {
                        "ci": [lower, upper],
                        "candidate_invalid": k_cand,
                        "reference_invalid": k_ref,
                        "reference_rate": k_ref / n,
                        "rows": n,
                        "rows_needed": rows_needed(gate_i.max, cfg.alpha),
                    },
                )
            )

    if g.quality_difference and g.quality_difference.metric == "macro_f1":
        gate_q = g.quality_difference
        if n == 0:
            results.append(_unverifiable("quality_difference", gate_q.margin, "no matched rows"))
        else:
            truth = indicators(frame[TRUTH_SET], classes)
            types, first, counts = _row_types(truth, ref, cand)
            res = macro_f1_sets_difference(truth[first], ref[first], cand[first], counts, cfg.alpha)
            ok = -gate_q.margin < res.low and res.high < gate_q.margin
            results.append(
                GateResult(
                    "quality_difference",
                    ok,
                    res.estimate,
                    gate_q.margin,
                    f"{two_sided} CI of macro_f1(candidate) - macro_f1(reference) within ±margin "
                    "(per-class F1 over sets of labels)",
                    {
                        "metric": "macro_f1",
                        "ci": [res.low, res.high],
                        "reference": macro_f1_sets(truth, ref),
                        "candidate": macro_f1_sets(truth, cand),
                        "row_types": int(types),
                    },
                )
            )

    order = [
        "label_agreement",
        "transitions",
        "class_prevalence",
        "quality_difference",
        "invalid_rate",
    ]
    return sorted(results, key=lambda r: order.index(r.name))


def _row_types(*matrices: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Distinct rows of the stacked indicator matrices: (count, first row of each, sizes)."""
    stacked = np.concatenate(matrices, axis=1)
    packed = np.packbits(stacked, axis=1)
    _, first, counts = np.unique(packed, axis=0, return_index=True, return_counts=True)
    return len(first), first, counts.astype(np.int64)


def summary(alignment: Alignment, cfg: ParityConfig) -> dict[str, Any]:
    frame, classes = alignment.frame, alignment.classes
    out: dict[str, Any] = {
        "n_reference": alignment.n_reference,
        "n_candidate": alignment.n_candidate,
        "n_matched_finite": alignment.n_matched,
        "n_missing": alignment.n_missing,
        "n_extra": alignment.n_extra,
        "n_nonfinite_mismatch": alignment.n_nonfinite_mismatch,
        "n_nonfinite_both": alignment.n_nonfinite_both,
        "classes": list(classes),
    }
    if not len(frame):
        return out
    ref = indicators(frame[REF_SET], classes)
    cand = indicators(frame[CAND_SET], classes)
    out["agreement"] = float((frame[REF] == frame[CAND]).mean())  # the same set
    both = (ref & cand).sum(axis=1)
    either = (ref | cand).sum(axis=1)
    out["jaccard_mean"] = float(np.where(either > 0, both / np.maximum(either, 1), 1.0).mean())
    out["prevalence"] = {
        cls: {"reference": float(ref[:, j].mean()), "candidate": float(cand[:, j].mean())}
        for j, cls in enumerate(classes)
    }
    out["label_changes"] = {
        cls: {
            "removed": int((ref[:, j] & ~cand[:, j]).sum()),
            "added": int((~ref[:, j] & cand[:, j]).sum()),
        }
        for j, cls in enumerate(classes)
    }
    norm = cfg.output.normalize
    if norm is not None and norm.allowed:
        out["invalid_share"] = {
            "reference": float(np.mean([norm.invalid in s for s in frame[REF_SET]])),
            "candidate": float(np.mean([norm.invalid in s for s in frame[CAND_SET]])),
        }
    if TRUTH_SET in frame.columns:
        truth = frame[TRUTH].to_numpy()
        out["accuracy"] = {
            "reference": float((frame[REF].to_numpy() == truth).mean()),
            "candidate": float((frame[CAND].to_numpy() == truth).mean()),
        }
    return out
