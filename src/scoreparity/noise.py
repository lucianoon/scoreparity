"""Measure the noise floor of a scoring system and suggest tolerances from it.

Choosing a margin is the hardest part of an equivalence test. Instead of guessing, run the
*reference* model several times under conditions that should not matter (another batch size,
another machine, another thread count, another row order) and measure how much its scores move.
That movement is the noise floor: no real change can be detected below it, and a gate set
tighter than it will fail on harmless reruns.

The suggestions are the worst noise observed across replicates times a safety factor, rounded
*up* to two significant digits. More replicates give a better estimate of the worst case; three
or more are recommended.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scoreparity import stats
from scoreparity.align import CAND, DIFF, LABEL, REF, align
from scoreparity.config import ParityConfig, from_dict
from scoreparity.errors import InputError
from scoreparity.gates import _top_k_ids

# Smallest margin ever suggested. TOST needs a strictly positive margin; when the measured noise
# is exactly zero (fully deterministic pipeline) this keeps the gate meaningful and tight.
MIN_MARGIN = 1e-12


def _ceil_sig(value: float, digits: int = 2) -> float:
    """Round up to `digits` significant digits (never below the input)."""
    if value <= 0 or not math.isfinite(value):
        return max(value, 0.0)
    exp = math.floor(math.log10(value)) - digits + 1
    # Try the nearest 2-digit decimal first (exact for values that already are one, e.g. 3e-07),
    # then step up until the result is not below the input. Decimal construction avoids
    # artefacts such as 3.0000000000000004e-07.
    scaled = max(1, round(value / 10**exp))
    while float(f"{scaled}e{exp}") < value:
        scaled += 1
    return float(f"{scaled}e{exp}")


@dataclass(frozen=True)
class ReplicateNoise:
    name: str
    n: int
    max_abs_diff: float
    quantile_abs_diff: float
    mean_ci_extent: float  # max(|ci_low|, |ci_high|) over the global and gated segment TOSTs
    flip_rate: float  # worst threshold
    top_k_overlap: float  # worst k
    auc_ci_extent: float | None


@dataclass(frozen=True)
class NoiseProfile:
    replicates: list[ReplicateNoise]
    safety: float
    q: float
    thresholds: tuple[float, ...]
    k_pct: tuple[float, ...]
    suggested_gates: dict[str, Any] = field(default_factory=dict)

    def worst(self, attr: str) -> float:
        values = [getattr(r, attr) for r in self.replicates if getattr(r, attr) is not None]
        if attr == "top_k_overlap":
            return float(min(values))
        return float(max(values)) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety": self.safety,
            "replicates": [r.__dict__ for r in self.replicates],
            "suggested_gates": self.suggested_gates,
        }


def _measure(
    name: str,
    reference: pd.DataFrame,
    replicate: pd.DataFrame,
    cfg: ParityConfig,
    context: pd.DataFrame | None,
    q: float,
    thresholds: tuple[float, ...],
    k_pct: tuple[float, ...],
) -> ReplicateNoise:
    a = align(reference, replicate, cfg.columns, cfg.segments, context)
    if a.n_missing or a.n_extra or a.n_nonfinite_mismatch:
        raise InputError(
            f"replicate {name!r} does not cover the same rows as the reference "
            f"(missing {a.n_missing}, extra {a.n_extra}, NaN mismatches {a.n_nonfinite_mismatch}); "
            "a noise run must score exactly the same records"
        )
    frame = a.frame
    if len(frame) < 2:
        raise InputError(f"replicate {name!r}: need at least 2 finite matched rows")
    diff = frame[DIFF].to_numpy(np.float64)
    abs_diff = np.abs(diff)
    ref = frame[REF].to_numpy(np.float64)
    cand = frame[CAND].to_numpy(np.float64)

    def extent(values: np.ndarray) -> float:
        res = stats.tost_paired(values, margin=1.0, alpha=cfg.alpha)
        return max(abs(res.ci_low), abs(res.ci_high))

    mean_extent = extent(diff)
    for col in cfg.segments:
        for _, part in frame.groupby(col, observed=True):
            if len(part) >= cfg.min_segment_size:
                mean_extent = max(mean_extent, extent(part[DIFF].to_numpy(np.float64)))

    flips = max(float(((ref >= t) != (cand >= t)).mean()) for t in thresholds)
    overlaps = []
    for k in k_pct:
        kk = max(1, round(len(frame) * k / 100))
        overlaps.append(len(_top_k_ids(ref, kk) & _top_k_ids(cand, kk)) / kk)

    auc_extent: float | None = None
    if LABEL in frame.columns:
        try:
            auc = stats.auc_paired_delong(frame[LABEL].to_numpy(bool), ref, cand, 1.0, cfg.alpha)
            auc_extent = max(abs(auc.ci_low), abs(auc.ci_high))
        except ValueError:
            auc_extent = None

    return ReplicateNoise(
        name=name,
        n=len(frame),
        max_abs_diff=float(abs_diff.max()),
        quantile_abs_diff=float(np.quantile(abs_diff, q, method="higher")),
        mean_ci_extent=float(mean_extent),
        flip_rate=flips,
        top_k_overlap=float(min(overlaps)),
        auc_ci_extent=auc_extent,
    )


def measure_noise(
    reference: pd.DataFrame,
    replicates: dict[str, pd.DataFrame],
    config: ParityConfig | None = None,
    context: pd.DataFrame | None = None,
    safety: float = 3.0,
    q: float = 0.99,
    thresholds: tuple[float, ...] = (0.5,),
    k_pct: tuple[float, ...] = (10.0,),
) -> NoiseProfile:
    """Measure reference-vs-replicate noise and suggest gates covering it with `safety` headroom."""
    if not replicates:
        raise InputError("at least one replicate is needed to measure noise")
    if safety < 1:
        raise InputError("safety must be >= 1 (a factor below 1 makes reruns fail)")
    cfg = config or ParityConfig()
    measured = [
        _measure(name, reference, rep, cfg, context, q, thresholds, k_pct)
        for name, rep in replicates.items()
    ]
    profile = NoiseProfile(measured, safety, q, tuple(thresholds), tuple(k_pct))

    # Discrete gates (decision flips, top-k membership, AUC rank order) are calibrated against
    # what the max_abs_diff tolerance M already allows: with every score free to move by up to
    # M, only rows within M of a threshold can flip, only rows within 2M of the top-k cut can
    # swap, and only positive/negative pairs within 2M can reorder. A few replicates that showed
    # *zero* such events do not prove the events are impossible, and a gate stricter than this
    # bound fails harmless reruns.
    first = align(reference, next(iter(replicates.values())), cfg.columns, cfg.segments, context)
    ref_scores = np.sort(first.frame[REF].to_numpy(np.float64))
    n = len(ref_scores)
    tol = _ceil_sig(safety * profile.worst("max_abs_diff"))

    def within(lo: float, hi: float) -> int:
        return int(
            np.searchsorted(ref_scores, hi, "right") - np.searchsorted(ref_scores, lo, "left")
        )

    flip_bound = (
        max((within(t - tol, t + tol) / n for t in thresholds), default=0.0) if tol else 0.0
    )
    overlap_bound = 1.0
    for k in k_pct:
        kk = max(1, round(n * k / 100))
        cut = ref_scores[n - kk]
        swappable = within(cut - 2 * tol, cut + 2 * tol) if tol else 0
        overlap_bound = min(overlap_bound, 1 - min(swappable, kk) / kk)

    worst_overlap = profile.worst("top_k_overlap")
    gates: dict[str, Any] = {
        "max_abs_diff": {"max": tol},
        "quantile_abs_diff": {
            "q": q,
            "max": _ceil_sig(safety * profile.worst("quantile_abs_diff")),
        },
        "mean_diff_equivalence": {
            "margin": max(MIN_MARGIN, _ceil_sig(safety * profile.worst("mean_ci_extent"))),
            "per_segment": bool(cfg.segments),
        },
        "decision_flips": {
            "thresholds": list(thresholds),
            "max_rate": min(1.0, _ceil_sig(max(safety * profile.worst("flip_rate"), flip_bound))),
        },
        "top_k_overlap": {
            "k_pct": list(k_pct),
            "min_overlap": max(
                0.0, min(1 - _ceil_sig(safety * (1 - worst_overlap)), overlap_bound)
            ),
        },
    }
    auc_extents = [r.auc_ci_extent for r in measured if r.auc_ci_extent is not None]
    if auc_extents and LABEL in first.frame.columns:
        labels = first.frame[LABEL].to_numpy(bool)
        scores = first.frame[REF].to_numpy(np.float64)
        pos, neg = np.sort(scores[labels]), np.sort(scores[~labels])
        reorderable = (
            int(
                (
                    np.searchsorted(neg, pos + 2 * tol, "right")
                    - np.searchsorted(neg, pos - 2 * tol, "left")
                ).sum()
            )
            if tol
            else 0
        )
        auc_bound = reorderable / (len(pos) * len(neg))
        # Unlike flips and top-k, the AUC gate tests a confidence interval, which adds the
        # DeLong uncertainty of those rare reorderings on top of the bound; hence `safety` here.
        gates["auc_difference"] = {
            "margin": max(MIN_MARGIN, _ceil_sig(safety * max(max(auc_extents), auc_bound)))
        }
    return NoiseProfile(measured, safety, q, tuple(thresholds), tuple(k_pct), gates)


def suggested_config_yaml(profile: NoiseProfile, base: ParityConfig) -> str:
    """A ready-to-commit configuration that documents where every number came from."""
    cols = {k: v for k, v in base.columns.__dict__.items() if v is not None}
    lines = [
        "# scoreparity configuration generated by `scoreparity noise`.",
        f"# Tolerances = worst noise over {len(profile.replicates)} replicate run(s) "
        f"x safety factor {profile.safety:g}, rounded up.",
        "# Measured noise (reference vs each replicate):",
    ]
    for r in profile.replicates:
        # The name is a user file name; json.dumps escapes newlines that could otherwise end
        # the comment and inject configuration into the generated YAML.
        lines.append(
            f"#   {json.dumps(r.name)}: max |diff| {r.max_abs_diff:.3g}, "
            f"q{profile.q:g} {r.quantile_abs_diff:.3g}, "
            f"mean CI extent {r.mean_ci_extent:.3g}, flips {r.flip_rate:.3g}, "
            f"top-k overlap {r.top_k_overlap:.6g}"
        )
    lines += [
        "# Review before committing: if the score feeds a decision, a tolerance can also be set",
        "# from what the business can absorb (see docs/choosing-tolerances.md).",
        "version: 1",
        "columns:",
        # Column names are user data: JSON strings are valid YAML and escape anything special.
        *[f"  {k}: {json.dumps(v)}" for k, v in cols.items()],
        f"segments: [{', '.join(json.dumps(s) for s in base.segments)}]",
        f"min_segment_size: {base.min_segment_size}",
        f"alpha: {base.alpha:g}",
        "gates:",
    ]
    for name, spec in profile.suggested_gates.items():
        body = ", ".join(f"{k}: {_yaml_value(v)}" for k, v in spec.items())
        lines.append(f"  {name}: {{{body}}}")
    text = "\n".join(lines) + "\n"
    import yaml

    from_dict(yaml.safe_load(text))  # the generated file must always be a valid config
    return text


def _yaml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(value) if value != 0 else "0.0"
    return str(value)
