"""Noise floor of a classifier: how much a model disagrees with itself.

LLM classifiers are not deterministic, even at temperature 0 (batching, routing and hardware
on the provider side). Running the *reference* model again on the same rows and comparing it
with itself measures that floor. A class gate stricter than the floor fails the reference
against its own rerun, so the suggested gates start from it.

Each class gate decides on the unfavourable bound of an interval, so the measured quantity is
that bound, not the point estimate: the worst bound over the replicates (globally and in every
gated segment or class) times `safety`, rounded to two significant digits in the lenient
direction. Because the bounds already contain the sampling uncertainty of a run of this size,
the default safety factor is smaller than for scores (1.75 instead of 3).

`transitions` gets one limit per source class: a small class, or one that is ambiguous by
nature, has a much noisier transition rate than the others and must not loosen their limits.

Calibration was chosen by simulation (five classes, 2,000 to 10,000 rows, 0.5% to 2% of answers
resampled per run, three replicates): at least 97% of fresh reruns of the same model pass the
suggested gates, and a candidate that moves 5% of one class to another fails every time. A
candidate with three times the noise is caught about half of the time; a lower `safety` (1.5)
catches it more often at the cost of more false alarms on reruns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scoreparity.align import CAND, REF, Alignment, align_labels
from scoreparity.config import ParityConfig
from scoreparity.errors import InputError
from scoreparity.gates_categorical import confusion_matrix
from scoreparity.stats_categorical import (
    clopper_pearson,
    cohen_kappa,
    paired_proportion_difference,
)

DEFAULT_SAFETY = 1.75
MIN_MARGIN = 1e-6  # smallest prevalence margin ever suggested (Tango margins must be > 0)


@dataclass(frozen=True)
class ClassReplicateNoise:
    name: str
    n: int
    disagreement: float  # observed share of rows with another class
    disagreement_upper: float  # worst one-sided upper bound, globally and per gated segment
    transition_upper: float  # worst upper bound over class pairs (classes >= min_class_size)
    transition_pair: str | None  # the pair that set it
    transition_upper_by_class: dict[str, float]  # worst upper bound per source class
    prevalence_extent: float  # worst max(|low|, |high|) of the Tango interval over classes
    kappa_lower: float
    invalid_upper: float | None  # upper bound of the replicate's share of invalid answers


def _disagreement_upper(alignment: Alignment, cfg: ParityConfig) -> float:
    frame = alignment.frame
    parts = [frame]
    for col in cfg.segments:
        parts += [p for _, p in frame.groupby(col, observed=True) if len(p) >= cfg.min_segment_size]
    worst = 0.0
    for part in parts:
        k = int((part[REF] != part[CAND]).sum())
        worst = max(worst, clopper_pearson(k, len(part), cfg.alpha)[1])
    return worst


def measure_replicate(
    name: str,
    reference: pd.DataFrame,
    replicate: pd.DataFrame,
    cfg: ParityConfig,
    context: pd.DataFrame | None,
) -> ClassReplicateNoise:
    a = align_labels(reference, replicate, cfg.columns, cfg.segments, context, cfg.output.normalize)
    if a.n_missing or a.n_extra or a.n_nonfinite_mismatch:
        raise InputError(
            f"replicate {name!r} does not cover the same rows as the reference "
            f"(missing {a.n_missing}, extra {a.n_extra}, missing labels "
            f"{a.n_nonfinite_mismatch}); a noise run must classify exactly the same records"
        )
    frame, classes = a.frame, a.classes
    n = len(frame)
    if n < 2:
        raise InputError(f"replicate {name!r}: need at least 2 matched rows")
    cm = confusion_matrix(a)

    worst_t, worst_pair = 0.0, None
    by_class: dict[str, float] = {}
    for i, src in enumerate(classes):
        n_i = int(cm[i].sum())
        if n_i < cfg.min_class_size:
            continue
        by_class[src] = 0.0
        for j, dst in enumerate(classes):
            if i != j:
                upper = clopper_pearson(int(cm[i, j]), n_i, cfg.alpha)[1]
                by_class[src] = max(by_class[src], upper)
                if upper > worst_t:
                    worst_t, worst_pair = upper, f"{src} -> {dst}"

    ref_v, cand_v = frame[REF].to_numpy(), frame[CAND].to_numpy()
    extent = 0.0
    for cls in classes:
        r, c = ref_v == cls, cand_v == cls
        res = paired_proportion_difference(
            int((r & c).sum()),
            int((r & ~c).sum()),
            int((~r & c).sum()),
            int((~r & ~c).sum()),
            cfg.alpha,
        )
        extent = max(extent, abs(res.low), abs(res.high))

    norm = cfg.output.normalize
    invalid_upper = None
    if norm is not None and norm.allowed:
        invalid_upper = max(
            clopper_pearson(int((v == norm.invalid).sum()), n, cfg.alpha)[1]
            for v in (ref_v, cand_v)
        )

    k_dis = int((ref_v != cand_v).sum())
    return ClassReplicateNoise(
        name=name,
        n=n,
        disagreement=k_dis / n,
        disagreement_upper=_disagreement_upper(a, cfg),
        transition_upper=worst_t,
        transition_pair=worst_pair,
        transition_upper_by_class=by_class,
        prevalence_extent=extent,
        kappa_lower=cohen_kappa(cm, cfg.alpha).lower,
        invalid_upper=invalid_upper,
    )


def suggest(
    measured: list[ClassReplicateNoise], cfg: ParityConfig, safety: float
) -> dict[str, Any]:
    from scoreparity.noise import _ceil_sig

    def loosen(worst: float) -> float:
        return min(1.0, _ceil_sig(safety * worst))

    gates: dict[str, Any] = {
        "label_agreement": {
            "min": _floor_sig(max(0.0, 1 - loosen(max(r.disagreement_upper for r in measured))))
        },
        "transitions": _transitions(measured, loosen),
        "class_prevalence": {
            "margin": max(MIN_MARGIN, loosen(max(r.prevalence_extent for r in measured)))
        },
        "kappa": {
            "min": _floor_sig(max(-1.0, 1 - loosen(max(1 - r.kappa_lower for r in measured))))
        },
    }
    invalid = [r.invalid_upper for r in measured if r.invalid_upper is not None]
    if invalid:
        gates["invalid_rate"] = {"max": loosen(max(invalid))}
    return gates


def _transitions(measured: list[ClassReplicateNoise], loosen: Any) -> dict[str, Any]:
    """One limit per source class: a small or naturally noisy class must not loosen the limit
    of the others. `max_rate`, the loosest, applies to classes the replicates did not gate."""
    classes = sorted({c for r in measured for c in r.transition_upper_by_class})
    limits = {
        c: loosen(max(r.transition_upper_by_class.get(c, 0.0) for r in measured)) for c in classes
    }
    max_rate = max(limits.values(), default=loosen(0.0))
    per_class = {c: v for c, v in limits.items() if v < max_rate}
    return {"max_rate": max_rate, **({"per_class": per_class} if per_class else {})}


def _floor_sig(value: float) -> float:
    """Clean up a `1 - x` value (0.9529999999 -> 0.953) without ever rounding it up."""
    rounded = float(f"{value:.6g}")
    return rounded if rounded <= value else float(np.nextafter(rounded, -np.inf))


def comment_line(r: ClassReplicateNoise) -> str:
    # The name is a user file name; json.dumps escapes newlines that could otherwise end the
    # comment and inject configuration into the generated YAML.
    pair = f" ({json.dumps(r.transition_pair)})" if r.transition_pair else ""
    invalid = f", invalid upper {r.invalid_upper:.3g}" if r.invalid_upper is not None else ""
    return (
        f"#   {json.dumps(r.name)}: disagreement {r.disagreement:.3g} "
        f"(upper {r.disagreement_upper:.3g}), worst transition upper "
        f"{r.transition_upper:.3g}{pair}, prevalence CI extent {r.prevalence_extent:.3g}, "
        f"kappa lower {r.kappa_lower:.4g}{invalid}"
    )
