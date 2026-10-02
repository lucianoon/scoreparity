"""Gates: each one turns an aligned comparison into a pass/fail verdict with evidence.

A gate never raises because the candidate misbehaves. If a gate cannot be evaluated (for
example, every row is non-finite), it fails and says why: an unverifiable gate must not pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scoreparity import stats
from scoreparity.align import CAND, DIFF, LABEL, REF, Alignment
from scoreparity.config import ParityConfig


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    field: str | None = None  # structured outputs: the field this gate checked

    @property
    def label(self) -> str:
        """Unique name in a report: `field.gate` for structured fields, else the gate name."""
        return f"{self.field}.{self.name}" if self.field else self.name

    def to_dict(self) -> dict[str, Any]:
        doc = {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "description": self.description,
            "details": self.details,
        }
        if self.field is not None:
            doc["field"] = self.field
        return doc


def _unverifiable(name: str, threshold: float | None, why: str) -> GateResult:
    return GateResult(name, False, None, threshold, f"could not be evaluated: {why}")


def _top_k_ids(values: np.ndarray, k: int) -> set[int]:
    # Stable ordering by (-score, row position) makes ties deterministic.
    order = np.lexsort((np.arange(len(values)), -values))
    return set(order[:k].tolist())


def common_gates(
    alignment: Alignment, cfg: ParityConfig, missing_text: str = "NaN/inf"
) -> list[GateResult]:
    """Gates every output kind has: dropped ids and outputs missing on exactly one side."""
    g = cfg.gates
    results: list[GateResult] = []
    if g.coverage:
        results.append(
            GateResult(
                "coverage",
                alignment.coverage >= g.coverage.min,
                alignment.coverage,
                g.coverage.min,
                "share of reference ids present in the candidate",
                {
                    "missing": alignment.n_missing,
                    "extra": alignment.n_extra,
                    "missing_examples": list(alignment.missing_examples),
                    "extra_examples": list(alignment.extra_examples),
                },
            )
        )

    if g.nonfinite:
        mismatches = alignment.n_nonfinite_mismatch
        results.append(
            GateResult(
                "nonfinite",
                mismatches <= g.nonfinite.max_mismatches,
                float(mismatches),
                float(g.nonfinite.max_mismatches),
                f"rows where exactly one version produced {missing_text}",
                {"both_nonfinite": alignment.n_nonfinite_both},
            )
        )
    return results


def evaluate(alignment: Alignment, cfg: ParityConfig) -> list[GateResult]:
    g = cfg.gates
    frame = alignment.frame
    n = len(frame)
    diff = frame[DIFF].to_numpy(np.float64)
    abs_diff = np.abs(diff)
    ref = frame[REF].to_numpy(np.float64)
    cand = frame[CAND].to_numpy(np.float64)
    results = common_gates(alignment, cfg)

    if g.max_abs_diff:
        t = g.max_abs_diff.max
        if n == 0:
            results.append(_unverifiable("max_abs_diff", t, "no finite matched rows"))
        else:
            worst = int(np.argmax(abs_diff))
            results.append(
                GateResult(
                    "max_abs_diff",
                    float(abs_diff[worst]) <= t,
                    float(abs_diff[worst]),
                    t,
                    "largest absolute score difference",
                    {
                        "worst_row": {
                            "reference": float(ref[worst]),
                            "candidate": float(cand[worst]),
                        }
                    },
                )
            )

    if g.quantile_abs_diff:
        t, q = g.quantile_abs_diff.max, g.quantile_abs_diff.q
        if n == 0:
            results.append(_unverifiable("quantile_abs_diff", t, "no finite matched rows"))
        else:
            value = float(np.quantile(abs_diff, q, method="higher"))
            results.append(
                GateResult(
                    "quantile_abs_diff",
                    value <= t,
                    value,
                    t,
                    f"q{q:g} of the absolute score difference",
                    {"q": q},
                )
            )

    if g.mean_diff_equivalence:
        gate = g.mean_diff_equivalence
        if n < 2:
            results.append(_unverifiable("mean_diff_equivalence", gate.margin, "fewer than 2 rows"))
        else:
            overall = stats.tost_paired(diff, gate.margin, cfg.alpha)
            segments: list[dict[str, Any]] = []
            failing: list[str] = []
            if gate.per_segment:
                for col in cfg.segments:
                    for seg_value, part in frame.groupby(col, sort=True, observed=True):
                        gated = len(part) >= cfg.min_segment_size
                        res = stats.tost_paired(
                            part[DIFF].to_numpy(np.float64), gate.margin, cfg.alpha
                        )
                        name = f"{col}={seg_value}"
                        segments.append(
                            {
                                "segment": name,
                                "n": len(part),
                                "gated": gated,
                                "mean_diff": res.estimate,
                                "ci": [res.ci_low, res.ci_high],
                                "equivalent": res.equivalent,
                            }
                        )
                        if gated and not res.equivalent:
                            failing.append(name)
            results.append(
                GateResult(
                    "mean_diff_equivalence",
                    overall.equivalent and not failing,
                    overall.estimate,
                    gate.margin,
                    f"TOST: {100 * (1 - 2 * cfg.alpha):g}% CI of the mean difference within ±margin"
                    + (", in every segment" if gate.per_segment and cfg.segments else ""),
                    {
                        "ci": [overall.ci_low, overall.ci_high],
                        "p_value": overall.p_value,
                        "failing_segments": failing,
                        "segments": segments,
                    },
                )
            )

    if g.decision_flips:
        gate_df = g.decision_flips
        per_threshold: dict[str, dict[str, float]] = {}
        for t in gate_df.thresholds:
            flips = (ref >= t) != (cand >= t)
            per_threshold[f"{t:g}"] = {
                "rate": float(flips.mean()) if n else 0.0,
                "count": int(flips.sum()),
            }
        worst_rate = max((v["rate"] for v in per_threshold.values()), default=0.0)
        if n == 0:
            results.append(
                _unverifiable("decision_flips", gate_df.max_rate, "no finite matched rows")
            )
        else:
            results.append(
                GateResult(
                    "decision_flips",
                    worst_rate <= gate_df.max_rate,
                    worst_rate,
                    gate_df.max_rate,
                    "share of rows on a different side of a decision threshold (worst threshold)",
                    {"thresholds": per_threshold},
                )
            )

    if g.top_k_overlap:
        gate_tk = g.top_k_overlap
        per_k: dict[str, dict[str, float]] = {}
        for k_pct in gate_tk.k_pct:
            k = max(1, round(n * k_pct / 100))
            overlap = len(_top_k_ids(ref, k) & _top_k_ids(cand, k)) / k if n else 0.0
            per_k[f"{k_pct:g}%"] = {"k": k, "overlap": overlap}
        worst_overlap = min((v["overlap"] for v in per_k.values()), default=0.0)
        if n == 0:
            results.append(
                _unverifiable("top_k_overlap", gate_tk.min_overlap, "no finite matched rows")
            )
        else:
            results.append(
                GateResult(
                    "top_k_overlap",
                    worst_overlap >= gate_tk.min_overlap,
                    worst_overlap,
                    gate_tk.min_overlap,
                    "overlap of the top-k% rows ranked by each version (worst k)",
                    {"k": per_k},
                )
            )

    if g.auc_difference:
        margin = g.auc_difference.margin
        try:
            auc = stats.auc_paired_delong(frame[LABEL].to_numpy(bool), ref, cand, margin, cfg.alpha)
        except ValueError as exc:
            results.append(_unverifiable("auc_difference", margin, str(exc)))
        else:
            results.append(
                GateResult(
                    "auc_difference",
                    auc.equivalent,
                    auc.delta,
                    margin,
                    f"paired DeLong {100 * (1 - 2 * cfg.alpha):g}% CI of "
                    "AUC(candidate) - AUC(reference) within ±margin",
                    {
                        "auc_reference": auc.auc_reference,
                        "auc_candidate": auc.auc_candidate,
                        "ci": [auc.ci_low, auc.ci_high],
                        "n_positive": auc.n_positive,
                        "n_negative": auc.n_negative,
                    },
                )
            )

    return results


def examples(alignment: Alignment, cfg: ParityConfig, values: np.ndarray) -> list[dict[str, Any]]:
    """The `cfg.examples` rows with the largest `values` (ties by row order), by id only."""
    frame = alignment.frame
    if cfg.examples <= 0 or len(frame) == 0:
        return []
    order = np.lexsort((np.arange(len(values)), -values))[: cfg.examples]
    ids = frame[cfg.columns.id].to_numpy()
    ref, cand = frame[REF].to_numpy(), frame[CAND].to_numpy()
    return [
        {
            "id": str(ids[i]),
            "reference": ref[i].item() if hasattr(ref[i], "item") else ref[i],
            "candidate": cand[i].item() if hasattr(cand[i], "item") else cand[i],
            "difference": float(values[i]),
        }
        for i in order
        if values[i] > 0
    ]


def _decade_histogram(abs_diff: np.ndarray) -> dict[str, Any]:
    """Counts of |diff| per power of ten: bucket e holds values in [10**e, 10**(e+1)).

    Score differences span many orders of magnitude (1e-16 float noise to 1e-1 real changes),
    so linear bins would put everything in the first bin. Exact zeros get their own bucket.
    """
    zero = int((abs_diff == 0).sum())
    positive = abs_diff[abs_diff > 0]
    exponents = np.floor(np.log10(positive)).astype(np.int64)
    values, counts = np.unique(exponents, return_counts=True)
    return {
        "zero": zero,
        "decades": [
            {"exponent": int(e), "count": int(c)} for e, c in zip(values, counts, strict=True)
        ],
    }


def summary(alignment: Alignment) -> dict[str, Any]:
    """Descriptive statistics that are useful whether or not a gate uses them."""
    frame = alignment.frame
    abs_diff = frame[DIFF].abs()
    out: dict[str, Any] = {
        "n_reference": alignment.n_reference,
        "n_candidate": alignment.n_candidate,
        "n_matched_finite": alignment.n_matched,
        "n_missing": alignment.n_missing,
        "n_extra": alignment.n_extra,
        "n_nonfinite_mismatch": alignment.n_nonfinite_mismatch,
        "n_nonfinite_both": alignment.n_nonfinite_both,
    }
    if len(frame):
        out["abs_diff_quantiles"] = {
            q: float(abs_diff.quantile(float(q), interpolation="higher"))
            for q in ("0.5", "0.9", "0.99", "0.999", "1.0")
        }
        out["mean_diff"] = float(frame[DIFF].mean())
        out["identical_share"] = float((frame[DIFF] == 0).mean())
        out["abs_diff_histogram"] = _decade_histogram(abs_diff.to_numpy(np.float64))
        # Rank correlation is undefined when either side is constant.
        defined = frame[REF].nunique() > 1 and frame[CAND].nunique() > 1
        out["spearman"] = (
            float(pd.Series(frame[REF]).corr(pd.Series(frame[CAND]), method="spearman"))
            if defined
            else None
        )
    if LABEL in frame.columns:
        # Reported whenever labels are available, even without the AUC gate.
        try:
            auc = stats.auc_paired_delong(
                frame[LABEL].to_numpy(bool),
                frame[REF].to_numpy(np.float64),
                frame[CAND].to_numpy(np.float64),
                margin=1.0,
            )
        except ValueError:
            out["auc"] = None
        else:
            out["auc"] = {
                "reference": auc.auc_reference,
                "candidate": auc.auc_candidate,
                "delta": auc.delta,
                "delta_ci90": [auc.ci_low, auc.ci_high],
            }
    return out
