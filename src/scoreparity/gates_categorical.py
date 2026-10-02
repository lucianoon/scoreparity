"""Gates for class outputs: labels, or the argmax of probability vectors.

Every gate decides on the *unfavourable* bound of an interval, so "not enough evidence" fails
instead of passing:
- label_agreement: lower bound of the agreement >= min (globally and in every gated segment);
- transitions: for each source class with >= min_class_size rows, the upper bound of the share
  of its rows that moved to each other class <= max_rate;
- class_prevalence: (1-2*alpha) interval of each class's paired share difference inside
  +/- margin (Tango);
- kappa: lower bound of Cohen's kappa >= min;
- quality_difference: interval of accuracy (Tango) or macro-F1 (bootstrap) difference inside
  +/- margin, against the ground truth.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from scoreparity.align import CAND, CAND_STAB, REF, REF_STAB, TRUTH, Alignment
from scoreparity.config import ParityConfig
from scoreparity.gates import GateResult, _unverifiable
from scoreparity.stats_categorical import (
    clopper_pearson,
    cohen_kappa,
    macro_f1,
    macro_f1_difference,
    paired_proportion_difference,
)


def _codes(frame: pd.DataFrame, classes: tuple[str, ...], column: str) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return frame[column].map(index).to_numpy(dtype=np.int64)


def confusion_matrix(alignment: Alignment) -> np.ndarray:
    """Counts with rows = reference class and columns = candidate class."""
    k = len(alignment.classes)
    ref = _codes(alignment.frame, alignment.classes, REF)
    cand = _codes(alignment.frame, alignment.classes, CAND)
    return np.bincount(ref * k + cand, minlength=k * k).reshape(k, k)


def rows_needed(max_rate: float, alpha: float) -> int:
    """Smallest n for which zero events already prove rate <= max_rate (Clopper-Pearson)."""
    if max_rate <= 0:
        return math.inf  # type: ignore[return-value]
    if max_rate >= 1:
        return 1
    return math.ceil(math.log(alpha) / math.log(1 - max_rate))


def evaluate(alignment: Alignment, cfg: ParityConfig) -> list[GateResult]:
    g = cfg.gates
    frame = alignment.frame
    classes = alignment.classes
    n = len(frame)
    results: list[GateResult] = []
    level = f"{100 * (1 - cfg.alpha):g}%"
    two_sided = f"{100 * (1 - 2 * cfg.alpha):g}%"

    if g.label_agreement:
        gate = g.label_agreement
        if n == 0:
            results.append(_unverifiable("label_agreement", gate.min, "no matched rows"))
        else:

            def agreement(part: pd.DataFrame) -> tuple[float, float, float, int]:
                k_dis = int((part[REF] != part[CAND]).sum())
                lo_dis, hi_dis = clopper_pearson(k_dis, len(part), cfg.alpha)
                return 1 - k_dis / len(part), 1 - hi_dis, 1 - lo_dis, k_dis

            point, lower, upper, k_dis = agreement(frame)
            segments: list[dict[str, Any]] = []
            failing: list[str] = []
            for col in cfg.segments:
                for value, part in frame.groupby(col, sort=True, observed=True):
                    s_point, s_lower, s_upper, s_dis = agreement(part)
                    gated = len(part) >= cfg.min_segment_size
                    name = f"{col}={value}"
                    segments.append(
                        {
                            "segment": name,
                            "n": len(part),
                            "gated": gated,
                            "agreement": s_point,
                            "ci": [s_lower, s_upper],
                            "disagreements": s_dis,
                            "passed": s_lower >= gate.min,
                        }
                    )
                    if gated and s_lower < gate.min:
                        failing.append(name)
            results.append(
                GateResult(
                    "label_agreement",
                    lower >= gate.min and not failing,
                    point,
                    gate.min,
                    f"share of rows with the same class; its {level} lower bound must reach the "
                    "threshold" + (", in every segment" if cfg.segments else ""),
                    {
                        "ci": [lower, upper],
                        "disagreements": k_dis,
                        "rows": n,
                        "rows_needed": rows_needed(1 - gate.min, cfg.alpha),
                        "failing_segments": failing,
                        "segments": segments,
                    },
                )
            )

    if g.transitions:
        gate_t = g.transitions
        if n == 0:
            results.append(_unverifiable("transitions", gate_t.max_rate, "no matched rows"))
        else:
            cm = confusion_matrix(alignment)
            pairs: list[dict[str, Any]] = []
            failing_pairs: list[str] = []
            underpowered: list[str] = []
            skipped_classes: list[str] = []
            checked_classes = 0
            worst = 0.0
            for i, src in enumerate(classes):
                n_i = int(cm[i].sum())
                if n_i < cfg.min_class_size:
                    if n_i:
                        skipped_classes.append(src)
                    continue
                checked_classes += 1
                limit = gate_t.per_class.get(src, gate_t.max_rate)
                for j, dst in enumerate(classes):
                    if i == j:
                        continue
                    count = int(cm[i, j])
                    rate = count / n_i
                    upper = clopper_pearson(count, n_i, cfg.alpha)[1]
                    worst = max(worst, rate)
                    if upper > limit:
                        label = f"{src} -> {dst}"
                        failing_pairs.append(label)
                        if count == 0:
                            underpowered.append(src)
                    if count:
                        pairs.append(
                            {
                                "from": src,
                                "to": dst,
                                "count": count,
                                "rows_of_source": n_i,
                                "rate": rate,
                                "upper": upper,
                            }
                        )
            need = rows_needed(gate_t.max_rate, cfg.alpha)
            results.append(
                GateResult(
                    "transitions",
                    checked_classes > 0 and not failing_pairs,
                    worst if checked_classes else None,
                    gate_t.max_rate,
                    (
                        f"share of a class's rows that moved to each other class; {level} upper "
                        "bound (classes with at least min_class_size rows"
                        + (", some with their own limit)" if gate_t.per_class else ")")
                        if checked_classes
                        else "could not be evaluated: no class reaches min_class_size"
                    ),
                    {
                        "failing_pairs": failing_pairs[:50],
                        "pairs": sorted(pairs, key=lambda p: -p["rate"])[:50],
                        "underpowered_classes": sorted(set(underpowered)),
                        "skipped_classes": skipped_classes,
                        "checked_classes": checked_classes,
                        "rows_needed_per_class": need,
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
            ref_v, cand_v = frame[REF].to_numpy(), frame[CAND].to_numpy()
            for cls in classes:
                r, c = ref_v == cls, cand_v == cls
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
                    f"{two_sided} CI of each class's share difference within ±margin (Tango)",
                    {"failing_classes": failing_classes, "per_class": per_class},
                )
            )

    if g.kappa:
        gate_k = g.kappa
        if n == 0:
            results.append(_unverifiable("kappa", gate_k.min, "no matched rows"))
        else:
            kap = cohen_kappa(confusion_matrix(alignment), cfg.alpha)
            results.append(
                GateResult(
                    "kappa",
                    kap.lower >= gate_k.min,
                    kap.kappa,
                    gate_k.min,
                    f"Cohen's kappa between the versions; its {level} lower bound must reach the "
                    "threshold",
                    {
                        "lower_bound": kap.lower,
                        "se": kap.se,
                        "observed_agreement": kap.observed_agreement,
                        "expected_agreement": kap.expected_agreement,
                    },
                )
            )

    if g.quality_difference:
        gate_q = g.quality_difference
        if n == 0:
            results.append(_unverifiable("quality_difference", gate_q.margin, "no matched rows"))
        else:
            truth = frame[TRUTH].to_numpy()
            ref_v, cand_v = frame[REF].to_numpy(), frame[CAND].to_numpy()
            if gate_q.metric == "accuracy":
                r_ok, c_ok = ref_v == truth, cand_v == truth
                res = paired_proportion_difference(
                    int((r_ok & c_ok).sum()),
                    int((r_ok & ~c_ok).sum()),
                    int((~r_ok & c_ok).sum()),
                    int((~r_ok & ~c_ok).sum()),
                    cfg.alpha,
                )
                metrics = {"reference": float(r_ok.mean()), "candidate": float(c_ok.mean())}
            else:
                codes = {c: i for i, c in enumerate(classes)}
                t_c = np.array([codes[x] for x in truth], dtype=np.int64)
                r_c = np.array([codes[x] for x in ref_v], dtype=np.int64)
                c_c = np.array([codes[x] for x in cand_v], dtype=np.int64)
                res = macro_f1_difference(t_c, r_c, c_c, len(classes), cfg.alpha)
                metrics = {
                    "reference": macro_f1(t_c, r_c, len(classes)),
                    "candidate": macro_f1(t_c, c_c, len(classes)),
                }
            ok = -gate_q.margin < res.low and res.high < gate_q.margin
            results.append(
                GateResult(
                    "quality_difference",
                    ok,
                    res.estimate,
                    gate_q.margin,
                    f"{two_sided} CI of {gate_q.metric}(candidate) - {gate_q.metric}(reference) "
                    "within ±margin",
                    {"metric": gate_q.metric, "ci": [res.low, res.high], **metrics},
                )
            )

    if g.invalid_rate:
        gate_i = g.invalid_rate
        invalid = cfg.output.normalize.invalid if cfg.output.normalize else ""
        if n == 0:
            results.append(_unverifiable("invalid_rate", gate_i.max, "no matched rows"))
        else:
            k_cand = int((frame[CAND] == invalid).sum())
            k_ref = int((frame[REF] == invalid).sum())
            lower, upper = clopper_pearson(k_cand, n, cfg.alpha)
            results.append(
                GateResult(
                    "invalid_rate",
                    upper <= gate_i.max,
                    k_cand / n,
                    gate_i.max,
                    f"share of candidate answers outside the allowed classes; its {level} upper "
                    "bound must stay below the threshold",
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

    if g.stability:
        gate_s = g.stability
        if n == 0 or REF_STAB not in frame.columns:
            results.append(_unverifiable("stability", gate_s.margin, "no matched rows"))
        else:
            r_unstable = frame[REF_STAB].to_numpy(np.float64) < 1
            c_unstable = frame[CAND_STAB].to_numpy(np.float64) < 1
            res = paired_proportion_difference(
                int((r_unstable & c_unstable).sum()),
                int((r_unstable & ~c_unstable).sum()),
                int((~r_unstable & c_unstable).sum()),
                int((~r_unstable & ~c_unstable).sum()),
                cfg.alpha,
            )
            results.append(
                GateResult(
                    "stability",
                    res.high < gate_s.margin,
                    res.estimate,
                    gate_s.margin,
                    "share of ids whose replicas disagree, candidate minus reference; its "
                    f"{level} upper bound must stay below the margin (Tango)",
                    {
                        "ci": [res.low, res.high],
                        "reference_unstable": float(r_unstable.mean()),
                        "candidate_unstable": float(c_unstable.mean()),
                    },
                )
            )

    return results


def stability_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """How often replicas of the same id disagree, per version, and where changes happen."""
    ref, cand = frame[REF_STAB].to_numpy(np.float64), frame[CAND_STAB].to_numpy(np.float64)
    changed = (frame[REF] != frame[CAND]).to_numpy()
    unstable = (ref < 1) | (cand < 1)
    return {
        "reference": {"mean": float(ref.mean()), "unstable_share": float((ref < 1).mean())},
        "candidate": {"mean": float(cand.mean()), "unstable_share": float((cand < 1).mean())},
        # Changes concentrated on ids that were already unstable point to sampling noise, not
        # to a different model behaviour.
        "changes_on_unstable_ids": float(unstable[changed].mean()) if changed.any() else None,
    }


def examples(alignment: Alignment, cfg: ParityConfig) -> list[dict[str, Any]]:
    """Up to `cfg.examples` rows whose class changed, by id only (never the input text).

    Rows are taken round-robin over the kinds of change (reference -> candidate pairs), the most
    frequent first, so a rare kind of change is not crowded out by a common one.
    """
    frame = alignment.frame
    changed = frame.loc[frame[REF] != frame[CAND], [cfg.columns.id, REF, CAND]]
    if changed.empty or cfg.examples <= 0:
        return []
    changed = changed.sort_values([REF, CAND, cfg.columns.id], kind="stable")
    groups = [g for _, g in changed.groupby([REF, CAND], sort=False)]
    groups.sort(key=lambda g: (-len(g), str(g[REF].iloc[0]), str(g[CAND].iloc[0])))
    out: list[dict[str, Any]] = []
    for i in range(max(len(g) for g in groups)):
        for g in groups:
            if i < len(g):
                row = g.iloc[i]
                out.append(
                    {"id": str(row[cfg.columns.id]), "reference": row[REF], "candidate": row[CAND]}
                )
                if len(out) == cfg.examples:
                    return out
    return out


def summary(alignment: Alignment, cfg: ParityConfig | None = None) -> dict[str, Any]:
    frame = alignment.frame
    out: dict[str, Any] = {
        "n_reference": alignment.n_reference,
        "n_candidate": alignment.n_candidate,
        "n_matched_finite": alignment.n_matched,
        "n_missing": alignment.n_missing,
        "n_extra": alignment.n_extra,
        "n_nonfinite_mismatch": alignment.n_nonfinite_mismatch,
        "n_nonfinite_both": alignment.n_nonfinite_both,
        "classes": list(alignment.classes),
    }
    if len(frame):
        cm = confusion_matrix(alignment)
        total = float(cm.sum())
        out["agreement"] = float(np.trace(cm)) / total
        out["confusion"] = {"classes": list(alignment.classes), "counts": cm.tolist()}
        out["prevalence"] = {
            cls: {
                "reference": float(cm[i].sum()) / total,
                "candidate": float(cm[:, i].sum()) / total,
            }
            for i, cls in enumerate(alignment.classes)
        }
        out["kappa"] = cohen_kappa(cm).kappa
        if REF_STAB in frame.columns:
            out["stability"] = stability_summary(frame)
        norm = cfg.output.normalize if cfg is not None else None
        if norm is not None and norm.allowed:
            out["invalid_share"] = {
                "reference": float((frame[REF] == norm.invalid).mean()),
                "candidate": float((frame[CAND] == norm.invalid).mean()),
            }
        if TRUTH in frame.columns:
            truth = frame[TRUTH].to_numpy()
            out["accuracy"] = {
                "reference": float((frame[REF].to_numpy() == truth).mean()),
                "candidate": float((frame[CAND].to_numpy() == truth).mean()),
            }
    return out
