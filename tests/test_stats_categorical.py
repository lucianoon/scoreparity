"""Statistical behaviour of the categorical procedures, checked by simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scoreparity.stats_categorical import (
    clopper_pearson,
    cohen_kappa,
    macro_f1_difference,
    paired_proportion_difference,
)

# --------------------------------------------------------------------------- Clopper-Pearson


def test_clopper_pearson_zero_events_matches_closed_form() -> None:
    for n in (10, 100, 3000):
        _, upper = clopper_pearson(0, n, alpha=0.05)
        assert upper == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-9)


@pytest.mark.parametrize(("p", "n"), [(0.001, 2000), (0.02, 300), (0.2, 100), (0.5, 50)])
def test_clopper_pearson_bounds_cover_at_least_nominal(p: float, n: int) -> None:
    rng = np.random.default_rng(0)
    ks = rng.binomial(n, p, size=4000)
    lower_ok = np.mean([clopper_pearson(int(k), n, 0.05)[0] <= p for k in ks])
    upper_ok = np.mean([clopper_pearson(int(k), n, 0.05)[1] >= p for k in ks])
    assert lower_ok >= 0.95 - 0.01
    assert upper_ok >= 0.95 - 0.01


# --------------------------------------------------------------------------- paired proportions


def _paired_cells(
    rng: np.random.Generator, n: int, p_ref: float, flip: float
) -> tuple[int, int, int, int]:
    """Reference event with prob p_ref; candidate copies it but flips each row with prob flip."""
    ref = rng.random(n) < p_ref
    cand = np.where(rng.random(n) < flip, ~ref, ref)
    both = int((ref & cand).sum())
    return both, int((ref & ~cand).sum()), int((~ref & cand).sum()), int((~ref & ~cand).sum())


@pytest.mark.parametrize(
    ("n", "p_ref", "flip"), [(200, 0.05, 0.02), (1000, 0.3, 0.05), (5000, 0.01, 0.002)]
)
def test_tango_interval_coverage(n: int, p_ref: float, flip: float) -> None:
    """Two-sided 1 - 2*alpha = 90% interval must cover the true difference about 90% of the time."""
    rng = np.random.default_rng(1)
    true_diff = flip * (1 - 2 * p_ref)  # E[p_cand - p_ref] under the flipping model
    covered = 0
    reps = 3000
    for _ in range(reps):
        res = paired_proportion_difference(*_paired_cells(rng, n, p_ref, flip), alpha=0.05)
        covered += res.low <= true_diff <= res.high
    assert 0.88 <= covered / reps <= 0.92


def test_tango_identical_outcomes_contain_zero() -> None:
    res = paired_proportion_difference(30, 0, 0, 970, alpha=0.05)
    assert res.estimate == 0.0
    assert res.low <= 0.0 <= res.high
    assert res.high - res.low < 0.01


def test_tango_false_equivalence_rate_at_most_alpha() -> None:
    """True difference exactly at the margin: claiming |diff| < margin must happen <= alpha."""
    rng = np.random.default_rng(2)
    n, margin = 800, 0.02
    hits, reps = 0, 3000
    for _ in range(reps):
        ref = rng.random(n) < 0.2
        cand = ref | (rng.random(n) < margin / 0.8)  # adds events -> E[diff] = margin
        cells = (
            int((ref & cand).sum()),
            int((ref & ~cand).sum()),
            int((~ref & cand).sum()),
            int((~ref & ~cand).sum()),
        )
        res = paired_proportion_difference(*cells, alpha=0.05)
        hits += -margin < res.low and res.high < margin
    assert hits / reps <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)


# --------------------------------------------------------------------------- kappa


def _rated(rng: np.random.Generator, n: int, k: int, flip: float) -> tuple[np.ndarray, np.ndarray]:
    ref = rng.choice(k, size=n, p=np.linspace(1, 2, k) / np.linspace(1, 2, k).sum())
    cand = np.where(rng.random(n) < flip, rng.integers(0, k, n), ref)
    return ref, cand


def _confusion(ref: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(ref * k + cand, minlength=k * k).reshape(k, k)


def test_kappa_perfect_agreement_and_single_class() -> None:
    assert cohen_kappa(np.diag([50, 30, 20])).kappa == pytest.approx(1.0)
    single = cohen_kappa(np.array([[100]]))
    assert single.kappa == 1.0


def test_kappa_standard_error_agrees_with_bootstrap() -> None:
    rng = np.random.default_rng(3)
    k, n = 4, 600
    ref, cand = _rated(rng, n, k, flip=0.15)
    res = cohen_kappa(_confusion(ref, cand, k))
    boot = []
    for _ in range(1500):
        i = rng.integers(0, n, n)
        boot.append(cohen_kappa(_confusion(ref[i], cand[i], k)).kappa)
    assert res.se == pytest.approx(float(np.std(boot, ddof=1)), rel=0.15)


def test_kappa_lower_bound_coverage() -> None:
    rng = np.random.default_rng(4)
    k, n = 3, 500
    # Large-sample "true" kappa for this generating process, from one huge sample.
    big_ref, big_cand = _rated(rng, 2_000_000, k, flip=0.2)
    true_kappa = cohen_kappa(_confusion(big_ref, big_cand, k)).kappa
    covered = sum(
        cohen_kappa(_confusion(*_rated(rng, n, k, 0.2), k)).lower <= true_kappa for _ in range(1500)
    )
    assert covered / 1500 >= 0.93


# --------------------------------------------------------------------------- macro-F1


def test_macro_f1_identical_predictions() -> None:
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 4, 2000)
    pred = np.where(rng.random(2000) < 0.8, truth, rng.integers(0, 4, 2000))
    res = macro_f1_difference(truth, pred, pred.copy(), 4)
    assert res.estimate == 0.0
    assert res.low == 0.0
    assert res.high == 0.0


def f1_score(truth: np.ndarray, pred: np.ndarray, average: str = "macro") -> float:
    """Independent reference implementation of macro-F1 (classes present in truth or pred)."""
    scores = []
    for c in np.union1d(truth, pred):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def test_macro_f1_bootstrap_matches_row_resampling() -> None:
    """The type-count multinomial bootstrap must agree with naive row resampling."""
    rng = np.random.default_rng(6)
    n, k = 800, 3
    truth = rng.integers(0, k, n)
    ref = np.where(rng.random(n) < 0.85, truth, rng.integers(0, k, n))
    cand = np.where(rng.random(n) < 0.80, truth, rng.integers(0, k, n))
    res = macro_f1_difference(truth, ref, cand, k, n_boot=3000, seed=7)
    point = f1_score(truth, cand, average="macro") - f1_score(truth, ref, average="macro")
    assert res.estimate == pytest.approx(point, abs=1e-12)
    naive = []
    for _ in range(1500):
        i = rng.integers(0, n, n)
        naive.append(
            f1_score(truth[i], cand[i], average="macro")
            - f1_score(truth[i], ref[i], average="macro")
        )
    lo, hi = np.quantile(naive, [0.05, 0.95])
    assert res.low == pytest.approx(lo, abs=0.01)
    assert res.high == pytest.approx(hi, abs=0.01)


def test_macro_f1_scales_to_millions_of_rows() -> None:
    import time

    rng = np.random.default_rng(8)
    n, k = 2_000_000, 10
    truth = rng.integers(0, k, n)
    ref = np.where(rng.random(n) < 0.9, truth, rng.integers(0, k, n))
    cand = np.where(rng.random(n) < 0.9, truth, rng.integers(0, k, n))
    t0 = time.perf_counter()
    macro_f1_difference(truth, ref, cand, k)
    assert time.perf_counter() - t0 < 20
