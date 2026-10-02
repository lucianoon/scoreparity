"""End-to-end behaviour of `compare`: alignment, gates and report."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scoreparity import InputError, compare, from_dict
from scoreparity.config import ParityConfig
from tests.conftest import make_scores

FULL = from_dict(
    {
        "preset": "float-noise",
        "columns": {"label": "label"},
        "segments": ["plan", "region"],
        "gates": {"auc_difference": {"margin": 0.001}},
    }
)


def gate(report_dict: dict[str, Any], name: str) -> dict[str, Any]:
    found: dict[str, Any] = next(g for g in report_dict["gates"] if g["name"] == name)
    return found


def test_identical_scores_pass_every_preset(reference: pd.DataFrame) -> None:
    for preset in ("exact", "float-noise", "quantization"):
        report = compare(reference, reference.copy(), from_dict({"preset": preset}))
        assert report.passed, (preset, report.failed_gates)


def test_float_noise_passes_float_noise_preset(reference: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    cand = reference.assign(score=reference["score"] + rng.normal(0, 1e-8, len(reference)))
    report = compare(reference, cand, FULL)
    assert report.passed, report.failed_gates


def test_row_order_and_extra_columns_do_not_matter(reference: pd.DataFrame) -> None:
    cand = reference.sample(frac=1.0, random_state=1)[["score", "id"]]
    assert compare(reference, cand, FULL).passed


def test_opposite_segment_shifts_fail_although_global_mean_is_zero(reference: pd.DataFrame) -> None:
    shift = np.where(reference["plan"] == "pre", 5e-6, -5e-6)
    # Balance the global mean exactly, so only the per-segment check can catch it.
    shift = shift - shift.mean()
    cand = reference.assign(score=reference["score"] + shift)
    cfg = from_dict(
        {
            "segments": ["plan"],
            "gates": {"mean_diff_equivalence": {"margin": 1e-6, "per_segment": True}},
        }
    )
    doc = compare(reference, cand, cfg).to_dict()
    g = gate(doc, "mean_diff_equivalence")
    assert not g["passed"]
    details = g["details"]
    assert isinstance(details, dict)
    assert abs(details["ci"][0]) < 1e-6
    assert abs(details["ci"][1]) < 1e-6
    assert any(s.startswith("plan=") for s in details["failing_segments"])


def test_nan_in_candidate_fails_gate_not_input(reference: pd.DataFrame) -> None:
    cand = reference.copy()
    cand.loc[:4, "score"] = np.nan
    doc = compare(reference, cand, FULL).to_dict()
    assert doc["verdict"] == "FAIL"
    assert gate(doc, "nonfinite")["value"] == 5


def test_nan_on_both_sides_is_agreement(reference: pd.DataFrame) -> None:
    ref = reference.copy()
    ref.loc[:2, "score"] = np.nan
    doc = compare(ref, ref.copy(), FULL).to_dict()
    assert doc["verdict"] == "PASS"
    assert doc["summary"]["n_nonfinite_both"] == 3


def test_missing_and_extra_ids(reference: pd.DataFrame) -> None:
    cand = pd.concat(
        [reference.iloc[10:], pd.DataFrame({"id": ["new1"], "score": [0.5]})], ignore_index=True
    )
    doc = compare(reference, cand, from_dict({})).to_dict()
    cov = gate(doc, "coverage")
    assert not cov["passed"]
    assert cov["details"]["missing"] == 10
    assert cov["details"]["extra"] == 1


def test_decision_flips_and_top_k(reference: pd.DataFrame) -> None:
    near = (reference["score"] - 0.5).abs().nsmallest(20).index
    cand = reference.copy()
    cand.loc[near, "score"] = 1 - cand.loc[near, "score"]  # cross the threshold
    cfg = from_dict({"gates": {"decision_flips": {"thresholds": [0.5], "max_rate": 0.001}}})
    g = gate(compare(reference, cand, cfg).to_dict(), "decision_flips")
    assert not g["passed"]
    assert g["details"]["thresholds"]["0.5"]["count"] == 20

    reversed_cand = reference.assign(score=1 - reference["score"])
    cfg = from_dict({"gates": {"top_k_overlap": {"k_pct": [10], "min_overlap": 0.9}}})
    g = gate(compare(reference, reversed_cand, cfg).to_dict(), "top_k_overlap")
    assert g["value"] == 0.0
    assert not g["passed"]


def test_auc_gate_catches_degraded_candidate(reference: pd.DataFrame) -> None:
    rng = np.random.default_rng(3)
    cand = reference.assign(score=reference["score"] + rng.normal(0, 0.3, len(reference)))
    g = gate(compare(reference, cand, FULL).to_dict(), "auc_difference")
    assert not g["passed"]
    assert g["details"]["auc_candidate"] < g["details"]["auc_reference"]


def test_labels_and_segments_from_context_table(reference: pd.DataFrame) -> None:
    scores = reference[["id", "score"]]
    context = reference[["id", "label", "plan", "region"]]
    assert compare(scores, scores.copy(), FULL, context=context).passed


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r, c: (r, c.drop(columns="score")), "missing columns"),
        (lambda r, c: (r, pd.concat([c, c.iloc[:1]])), "duplicated ids"),
        (lambda r, c: (r, c.assign(score=c["score"].astype(str))), "not numeric"),
        (lambda r, c: (r, c.assign(id=range(len(c)))), "different types"),
        (lambda r, c: (r.iloc[:0], c), "no rows"),
        (lambda r, c: (r.assign(label=2), c), "binary"),
    ],
)
def test_inputs_that_cannot_be_compared(reference: pd.DataFrame, mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    ref, cand = mutate(reference, reference.copy())
    with pytest.raises(InputError, match=message):
        compare(ref, cand, FULL)


def test_report_is_strict_json_with_stable_keys(reference: pd.DataFrame) -> None:
    constant = reference.assign(score=0.5)  # spearman of a constant is NaN
    doc = json.loads(compare(constant, constant.copy(), FULL).to_json())
    assert set(doc) == {
        "schema_version",
        "verdict",
        "failed_gates",
        "gates",
        "summary",
        "config",
        "inputs",
        "environment",
    }
    assert doc["schema_version"] == 1
    assert doc["summary"]["spearman"] is None


def test_markdown_mentions_verdict_and_failures(reference: pd.DataFrame) -> None:
    md = compare(reference, reference.assign(score=1 - reference["score"]), FULL).to_markdown()
    assert "FAIL" in md
    assert "`max_abs_diff`" in md


@settings(max_examples=40, deadline=None)
@given(
    scores=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=3, max_size=200),
    seed=st.integers(0, 2**32 - 1),
)
def test_property_self_comparison_passes_exact_and_is_order_invariant(
    scores: list[float], seed: int
) -> None:
    ref = pd.DataFrame({"id": np.arange(len(scores)), "score": scores})
    cand = ref.sample(frac=1.0, random_state=seed % (2**32 - 1))
    report = compare(ref, cand, from_dict({"preset": "exact"}))
    assert report.passed
    assert report.summary["identical_share"] == 1.0


def test_default_config_only_checks_coverage_and_nonfinite(reference: pd.DataFrame) -> None:
    report = compare(reference, reference.assign(score=0.0), ParityConfig())
    assert report.passed  # nothing about scores is gated by default; presets/configs add gates
    assert [g.name for g in report.gates] == ["coverage", "nonfinite"]


def test_many_rows_are_fast_enough() -> None:
    import time

    ref = make_scores(n=300_000, seed=9)
    cand = ref.assign(score=ref["score"] + 1e-9)
    t0 = time.perf_counter()
    report = compare(ref, cand, FULL)
    assert report.passed
    assert time.perf_counter() - t0 < 20
