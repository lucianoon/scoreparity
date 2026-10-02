"""Statistical behaviour of the procedures, checked by simulation and independent estimators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from scoreparity.stats import auc_paired_delong, tost_paired


def test_tost_accepts_noise_well_inside_margin() -> None:
    rng = np.random.default_rng(0)
    res = tost_paired(rng.normal(0, 1e-7, 5000), margin=1e-5)
    assert res.equivalent
    assert res.ci_low < 0 < res.ci_high


def test_tost_rejects_shift_beyond_margin() -> None:
    rng = np.random.default_rng(1)
    assert not tost_paired(rng.normal(3e-5, 1e-7, 5000), margin=1e-5).equivalent


def test_tost_zero_variance_is_exact_equality() -> None:
    assert tost_paired(np.zeros(50), margin=1e-9).equivalent
    assert not tost_paired(np.full(50, 2e-9), margin=1e-9).equivalent


def test_no_significant_difference_is_not_equivalence() -> None:
    """Small, noisy sample: a t-test finds 'no difference', TOST correctly withholds equivalence."""
    rng = np.random.default_rng(2)
    d = rng.normal(0.0, 0.05, 20)
    assert sps.ttest_1samp(d, 0.0).pvalue > 0.05
    assert not tost_paired(d, margin=1e-3).equivalent


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_tost_false_equivalence_rate_is_at_most_alpha(alpha: float) -> None:
    """At the null boundary (true mean == margin) TOST must wrongly pass <= alpha of the time."""
    rng = np.random.default_rng(3)
    reps, n, margin, sd = 4000, 60, 0.1, 0.2
    hits = sum(
        tost_paired(rng.normal(margin, sd, n), margin, alpha).equivalent for _ in range(reps)
    )
    rate = hits / reps
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert rate <= alpha + 3 * se


def test_tost_has_power_when_truly_equivalent() -> None:
    rng = np.random.default_rng(4)
    passes = sum(tost_paired(rng.normal(0, 0.2, 400), 0.1).equivalent for _ in range(500))
    assert passes / 500 > 0.95


# --------------------------------------------------------------------------- DeLong


def _labels_and_scores(
    rng: np.random.Generator, n: int, shift: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    y = rng.random(n) < 0.3
    latent = rng.normal(0, 1, n) + shift * y
    return y, latent


def test_delong_auc_matches_mann_whitney() -> None:
    rng = np.random.default_rng(5)
    y, s = _labels_and_scores(rng, 800)
    s = np.round(s, 1)  # ties on purpose
    res = auc_paired_delong(y, s, s + 0.0, margin=0.01)
    u = sps.mannwhitneyu(s[y], s[~y]).statistic
    assert res.auc_reference == pytest.approx(u / (y.sum() * (~y).sum()), abs=1e-12)


def test_delong_identical_scores_have_zero_difference_and_width() -> None:
    rng = np.random.default_rng(6)
    y, s = _labels_and_scores(rng, 500)
    res = auc_paired_delong(y, s, s.copy(), margin=1e-6)
    assert res.delta == 0.0
    assert res.se == 0.0
    assert res.equivalent


def test_delong_standard_error_agrees_with_paired_bootstrap() -> None:
    rng = np.random.default_rng(7)
    n = 600
    y, latent = _labels_and_scores(rng, n)
    ref = latent + rng.normal(0, 0.5, n)
    cand = latent + rng.normal(0, 0.5, n)
    res = auc_paired_delong(y, ref, cand, margin=0.05)

    deltas = []
    for _ in range(1500):
        i = rng.integers(0, n, n)
        if y[i].all() or not y[i].any():
            continue
        r = auc_paired_delong(y[i], ref[i], cand[i], margin=0.05)
        deltas.append(r.delta)
    assert res.se == pytest.approx(float(np.std(deltas, ddof=1)), rel=0.15)


def test_delong_interval_coverage_under_equal_auc() -> None:
    """Two noisy versions of the same model: the 90% CI must cover 0 about 90% of the time."""
    rng = np.random.default_rng(8)
    reps, n, covered = 600, 400, 0
    for _ in range(reps):
        y, latent = _labels_and_scores(rng, n)
        res = auc_paired_delong(
            y, latent + rng.normal(0, 0.7, n), latent + rng.normal(0, 0.7, n), margin=1.0
        )
        covered += res.ci_low <= 0.0 <= res.ci_high
    assert 0.86 <= covered / reps <= 0.94


def test_delong_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="positive and 2 negative"):
        auc_paired_delong(np.array([True, True, False]), np.ones(3), np.ones(3), margin=0.1)
