"""Statistical procedures for equivalence of paired scores.

Equivalence, not difference
---------------------------
A classical test asks "is there evidence of a difference?". Failing to find one (p > 0.05)
is *not* evidence of equivalence: it happens whenever the sample is small or noisy. Equivalence
is tested with TOST (two one-sided tests, Schuirmann 1987): the null hypothesis is that the true
difference lies *outside* [-margin, +margin], and it is rejected only when the (1 - 2*alpha)
confidence interval lies entirely inside the margin.

Several segments
----------------
Requiring every segment to pass its own TOST at level alpha is an intersection-union test: the
overall probability of wrongly declaring equivalence stays at or below alpha without any
multiple-comparison correction (Berger 1982). It only makes the gate more conservative.

AUC
---
The paired difference of two ROC AUCs on the same rows uses DeLong et al. (1988) with the
O(n log n) midrank algorithm of Sun & Xu (2014), which scales to millions of rows, unlike a
bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EquivalenceResult:
    estimate: float
    ci_low: float
    ci_high: float
    margin: float
    p_value: float
    n: int

    @property
    def equivalent(self) -> bool:
        return -self.margin < self.ci_low and self.ci_high < self.margin


def tost_paired(diff: FloatArray, margin: float, alpha: float = 0.05) -> EquivalenceResult:
    """TOST for the mean of paired differences, using the t distribution.

    With zero variance the interval collapses to the observed mean (exact equality is the
    common case for deterministic pipelines and must not crash or be reported as a failure).
    """
    d = np.asarray(diff, dtype=np.float64)
    n = int(d.size)
    if n == 0:
        raise ValueError("tost_paired needs at least one difference")
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0 or not np.isfinite(sd):
        inside = abs(mean) < margin
        return EquivalenceResult(mean, mean, mean, margin, 0.0 if inside else 1.0, n)
    se = sd / np.sqrt(n)
    df = n - 1
    p_lower = float(stats.t.sf((mean + margin) / se, df))  # H0: mu <= -margin
    p_upper = float(stats.t.cdf((mean - margin) / se, df))  # H0: mu >= +margin
    half = float(stats.t.ppf(1 - alpha, df)) * se
    return EquivalenceResult(mean, mean - half, mean + half, margin, max(p_lower, p_upper), n)


def _midrank(x: FloatArray) -> FloatArray:
    """Midranks (1-based, ties get the average rank), as in Sun & Xu (2014)."""
    return np.asarray(stats.rankdata(x, method="average"), dtype=np.float64)


@dataclass(frozen=True)
class AucComparison:
    auc_reference: float
    auc_candidate: float
    delta: float
    se: float
    ci_low: float
    ci_high: float
    margin: float
    n_positive: int
    n_negative: int

    @property
    def equivalent(self) -> bool:
        return -self.margin < self.ci_low and self.ci_high < self.margin


def auc_paired_delong(
    label: NDArray[np.bool_],
    reference: FloatArray,
    candidate: FloatArray,
    margin: float,
    alpha: float = 0.05,
) -> AucComparison:
    """AUC(candidate) - AUC(reference) with a (1 - 2*alpha) DeLong confidence interval."""
    y = np.asarray(label, dtype=bool)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        raise ValueError("AUC comparison needs at least 2 positive and 2 negative labels")
    scores = np.vstack([np.asarray(reference, np.float64), np.asarray(candidate, np.float64)])
    pos, neg = scores[:, y], scores[:, ~y]

    aucs = np.empty(2)
    v01 = np.empty((2, m))  # structural components of the positives
    v10 = np.empty((2, n))  # structural components of the negatives
    for k in range(2):
        tx = _midrank(pos[k])
        ty = _midrank(neg[k])
        tz = _midrank(np.concatenate([pos[k], neg[k]]))
        aucs[k] = tz[:m].sum() / (m * n) - (m + 1) / (2 * n)
        v01[k] = (tz[:m] - tx) / n
        v10[k] = 1.0 - (tz[m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    delta = float(aucs[1] - aucs[0])
    z = float(stats.norm.ppf(1 - alpha))
    return AucComparison(
        auc_reference=float(aucs[0]),
        auc_candidate=float(aucs[1]),
        delta=delta,
        se=se,
        ci_low=delta - z * se,
        ci_high=delta + z * se,
        margin=margin,
        n_positive=m,
        n_negative=n,
    )
