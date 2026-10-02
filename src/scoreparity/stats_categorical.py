"""Statistics for categorical outputs (class labels).

All intervals follow the same convention as the continuous gates: an equivalence decision uses
the (1 - 2*alpha) two-sided interval (equivalently, two one-sided tests at level alpha), and a
one-sided bound ("at least", "at most") uses level 1 - alpha.

- Clopper-Pearson (1934): exact binomial bounds for agreement and event rates. It stays valid
  with zero events, which is the common case for well-behaved migrations.
- Tango (1998): score interval for the difference of two *paired* proportions (the same
  rows scored by both versions). Chosen over Newcombe's hybrid method 10 by simulation:
  with the true difference exactly at the margin, Newcombe declared equivalence in up to 7.5%
  of runs (nominal 5%) when discordance is one-sided, while Tango stayed at or below 5.6% and
  its 90% interval covered the truth 89.4-90.3% of the time in every scenario tested.
- Cohen's kappa with the large-sample variance of Fleiss, Cohen & Everitt (1969).
- Macro-F1 difference by nonparametric bootstrap, computed on the K^3 distinct
  (truth, reference, candidate) row types: resampling rows is a multinomial draw over those
  types, so the cost does not grow with the number of rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """One-sided exact bounds (lower, upper) for a binomial proportion, each at level 1-alpha."""
    if n <= 0:
        raise ValueError("clopper_pearson needs n >= 1")
    if not 0 <= k <= n:
        raise ValueError("clopper_pearson needs 0 <= k <= n")
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))
    return lower, upper


def _tango_z(delta: float, n12: int, n21: int, n: int) -> float:
    """Tango (1998) score statistic for delta = p12 - p21 (paired difference)."""
    a = 2.0 * n
    b = -n12 - n21 + (2.0 * n - n12 + n21) * delta
    c = -n21 * delta * (1.0 - delta)
    disc = max(0.0, b * b - 4 * a * c)
    p21 = (math.sqrt(disc) - b) / (2 * a)  # restricted MLE of p21 under delta
    # Var(n12 - n21) = n * (p12 + p21 - delta^2) and p12 = p21 + delta under the null.
    var = n * (2 * p21 + delta * (1 - delta))
    if var <= 0:
        diff = n12 - n21 - n * delta
        return math.copysign(math.inf, diff) if diff else 0.0
    return (n12 - n21 - n * delta) / math.sqrt(var)


def paired_proportion_difference(
    both: int, ref_only: int, cand_only: int, neither: int, alpha: float = 0.05
) -> Interval:
    """p_candidate - p_reference with Tango's score interval (two-sided level 1 - 2*alpha).

    The interval is {delta : |Z(delta)| <= z}; Z is monotone decreasing in delta, so each
    bound is found by bisection.
    """
    n = both + ref_only + cand_only + neither
    if n <= 0:
        raise ValueError("paired_proportion_difference needs at least one row")
    z = float(stats.norm.ppf(1 - alpha))
    n12, n21 = cand_only, ref_only  # X = candidate, Y = reference
    estimate = (n12 - n21) / n

    def solve(target: float, lo: float, hi: float) -> float:
        # find delta in [lo, hi] with Z(delta) = target (Z decreasing in delta)
        for _ in range(200):
            mid = (lo + hi) / 2
            if _tango_z(mid, n12, n21, n) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    eps = 1e-12
    low = solve(z, -1 + eps, estimate) if _tango_z(-1 + eps, n12, n21, n) > z else -1.0
    high = solve(-z, estimate, 1 - eps) if _tango_z(1 - eps, n12, n21, n) < -z else 1.0
    return Interval(estimate, low, high)


@dataclass(frozen=True)
class Kappa:
    kappa: float
    se: float
    lower: float  # one-sided lower bound at level 1 - alpha
    observed_agreement: float
    expected_agreement: float


def cohen_kappa(confusion: NDArray[np.int64], alpha: float = 0.05) -> Kappa:
    """Cohen's kappa between two raters (rows: reference, columns: candidate)."""
    counts = np.asarray(confusion, dtype=np.float64)
    n = counts.sum()
    if n <= 0:
        raise ValueError("cohen_kappa needs at least one row")
    p = counts / n
    rows, cols = p.sum(axis=1), p.sum(axis=0)
    po = float(np.trace(p))
    pe = float((rows * cols).sum())
    if math.isclose(pe, 1.0):
        # Every row in a single class on both sides: kappa is undefined; agreement is the
        # only meaningful statement (perfect agreement -> 1, otherwise -> 0).
        k = 1.0 if math.isclose(po, 1.0) else 0.0
        return Kappa(k, 0.0, k, po, pe)
    kappa = (po - pe) / (1 - pe)
    diag = np.diag(p)
    term_a = float((diag * (1 - (rows + cols) * (1 - kappa)) ** 2).sum())
    off = p.copy()
    np.fill_diagonal(off, 0.0)
    weights = (cols[:, None] + rows[None, :]) ** 2  # (p_.i + p_j.)^2 for cell (i, j)
    term_b = (1 - kappa) ** 2 * float((off * weights).sum())
    term_c = (kappa - pe * (1 - kappa)) ** 2
    var = max(0.0, (term_a + term_b - term_c) / (n * (1 - pe) ** 2))
    se = math.sqrt(var)
    lower = kappa - float(stats.norm.ppf(1 - alpha)) * se
    return Kappa(kappa, se, lower, po, pe)


def _macro_f1(
    tp: NDArray[np.float64], fp: NDArray[np.float64], fn: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Macro-F1 over the last axis; classes with no support and no predictions are skipped."""
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = denom > 0
    return np.asarray(f1.sum(axis=-1) / np.maximum(present.sum(axis=-1), 1))


def macro_f1(truth: NDArray[np.int64], pred: NDArray[np.int64], n_classes: int) -> float:
    """Macro-F1 over the classes present in truth or predictions."""
    k = n_classes
    hit = truth == pred
    tp = np.bincount(truth[hit], minlength=k).astype(np.float64)
    fp = np.bincount(pred[~hit], minlength=k).astype(np.float64)
    fn = np.bincount(truth[~hit], minlength=k).astype(np.float64)
    return float(_macro_f1(tp, fp, fn))


def macro_f1_difference(
    truth: NDArray[np.int64],
    reference: NDArray[np.int64],
    candidate: NDArray[np.int64],
    n_classes: int,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> Interval:
    """Macro-F1(candidate) - Macro-F1(reference) with a percentile bootstrap interval at 1-2*alpha.

    Inputs are integer class codes in [0, n_classes). The bootstrap resamples rows, implemented
    as a multinomial draw over the distinct (truth, reference, candidate) row types.
    """
    k = n_classes
    codes = (truth * k + reference) * k + candidate
    types, counts = np.unique(codes, return_counts=True)
    t_cls, rest = np.divmod(types, k * k)
    r_cls, c_cls = np.divmod(rest, k)

    def f1_for(weights: NDArray[np.float64], pred: NDArray[np.int64]) -> NDArray[np.float64]:
        # weights: (B, T) counts per row type; returns macro-F1 per bootstrap replicate.
        onehot_t = np.eye(k)[t_cls]  # (T, K)
        onehot_p = np.eye(k)[pred]
        hit = (t_cls == pred).astype(np.float64)
        tp = weights @ (onehot_t * hit[:, None])
        fp = weights @ (onehot_p * (1 - hit)[:, None])
        fn = weights @ (onehot_t * (1 - hit)[:, None])
        return _macro_f1(tp, fp, fn)

    observed = counts[None, :].astype(np.float64)
    estimate = float(f1_for(observed, c_cls)[0] - f1_for(observed, r_cls)[0])
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(int(counts.sum()), counts / counts.sum(), size=n_boot).astype(
        np.float64
    )
    deltas = f1_for(draws, c_cls) - f1_for(draws, r_cls)
    low, high = np.quantile(deltas, [alpha, 1 - alpha])
    return Interval(estimate, float(low), float(high))
