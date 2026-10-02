"""v0.2: class labels and probability vectors, end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

import scoreparity as sp
from scoreparity.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main

CLASSES = ["fatura", "sinal", "oferta", "atendimento", "cancelamento"]
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "report-v1.schema.json").read_text("utf-8")
)


def labels_frame(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = rng.choice(CLASSES, n, p=[0.35, 0.25, 0.2, 0.15, 0.05])
    pred = np.where(rng.random(n) < 0.85, truth, rng.choice(CLASSES, n))
    frame: pd.DataFrame = pd.DataFrame(
        {"id": np.arange(n), "label": pred, "truth": truth, "canal": rng.choice(["app", "loja"], n)}
    )
    return frame


def probs_frame(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (n, len(CLASSES))) + np.linspace(1, -0.5, len(CLASSES))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    truth = np.array(CLASSES)[[rng.choice(len(CLASSES), p=row) for row in p]]
    frame: pd.DataFrame = pd.DataFrame(
        {"id": np.arange(n), "truth": truth, "canal": rng.choice(["a", "b"], n)}
    )
    for i, c in enumerate(CLASSES):
        frame[f"p_{c}"] = p[:, i]
    return frame


def label_cfg(**gates: Any) -> sp.ParityConfig:
    return sp.from_dict(
        {
            "output": {"type": "label"},
            "columns": {"score": "label", "truth": "truth"},
            "segments": ["canal"],
            "gates": gates
            or {
                "label_agreement": {"min": 0.995},
                "transitions": {"max_rate": 0.01},
                "class_prevalence": {"margin": 0.005},
                "kappa": {"min": 0.98},
                "quality_difference": {"metric": "accuracy", "margin": 0.01},
            },
        }
    )


def gate(report: sp.Report, name: str) -> Any:
    return next(g for g in report.gates if g.name == name)


# --------------------------------------------------------------------------- labels


def test_identical_labels_pass_every_class_gate() -> None:
    ref = labels_frame()
    report = sp.compare(ref, ref.copy(), label_cfg())
    assert report.passed, report.failed_gates
    assert report.summary["agreement"] == 1.0


def test_a_systematic_transition_fails_the_right_gates() -> None:
    ref = labels_frame()
    cand = ref.copy()
    rng = np.random.default_rng(1)
    moved = (ref["label"] == "fatura") & (rng.random(len(ref)) < 0.03)
    cand.loc[moved, "label"] = "cancelamento"
    report = sp.compare(ref, cand, label_cfg())
    assert "fatura -> cancelamento" in gate(report, "transitions").details["failing_pairs"]
    assert "cancelamento" in gate(report, "class_prevalence").details["failing_classes"]
    assert {"label_agreement", "transitions", "class_prevalence"} <= set(report.failed_gates)


def test_kappa_catches_what_agreement_hides_with_a_dominant_class() -> None:
    """99% of rows are 'outros': scrambling the rare classes keeps agreement high."""
    rng = np.random.default_rng(2)
    n = 50_000
    ref = pd.DataFrame(
        {
            "id": np.arange(n),
            "label": np.where(rng.random(n) < 0.99, "outros", rng.choice(["a", "b", "c"], n)),
        }
    )
    cand = ref.copy()
    rare = ref["label"] != "outros"
    cand.loc[rare, "label"] = rng.choice(["a", "b", "c"], int(rare.sum()))
    cfg = sp.from_dict(
        {
            "output": {"type": "label"},
            "columns": {"score": "label"},
            "gates": {"label_agreement": {"min": 0.99}, "kappa": {"min": 0.9}},
        }
    )
    report = sp.compare(ref, cand, cfg)
    assert gate(report, "label_agreement").passed
    assert not gate(report, "kappa").passed


def test_numbers_and_strings_are_the_same_class() -> None:
    ref = pd.DataFrame({"id": range(500), "label": [1, 2] * 250})
    cand = pd.DataFrame({"id": range(500), "label": ["1", " 2 "] * 250})
    cfg = sp.from_dict(
        {
            "output": {"type": "label"},
            "columns": {"score": "label"},
            "gates": {"label_agreement": {"min": 0.99}},
        }
    )
    assert sp.compare(ref, cand, cfg).summary["agreement"] == 1.0


def test_missing_labels_are_behaviour_not_input_errors() -> None:
    ref = labels_frame(3000)
    cand = ref.copy()
    cand.loc[:4, "label"] = None
    cand.loc[5:6, "label"] = "   "
    both = ref.copy()
    both.loc[:2, "label"] = None
    report = sp.compare(ref, cand, label_cfg())
    assert gate(report, "nonfinite").value == 7
    assert not gate(report, "nonfinite").passed
    agree = sp.compare(both, both.copy(), label_cfg(label_agreement={"min": 0.99}))
    assert agree.summary["n_nonfinite_both"] == 3


def test_underpowered_classes_are_reported_and_fail() -> None:
    ref = labels_frame(3000)
    report = sp.compare(ref, ref.copy(), label_cfg(transitions={"max_rate": 0.001}))
    g = gate(report, "transitions")
    assert not g.passed
    assert g.details["rows_needed_per_class"] == 2995
    assert set(g.details["underpowered_classes"]) == set(CLASSES)
    assert "not enough rows" in report.to_markdown()


def test_min_class_size_excludes_small_classes_deliberately() -> None:
    ref = labels_frame(20_000)
    cfg = sp.from_dict(
        {
            **json.loads(json.dumps({"output": {"type": "label"}})),
            "columns": {"score": "label"},
            "min_class_size": 1500,
            "gates": {"transitions": {"max_rate": 0.01}},
        }
    )
    assert sp.compare(ref, ref.copy(), cfg).passed  # 'cancelamento' (~5%) is excluded


def test_transitions_fail_when_all_classes_are_too_small() -> None:
    ref = pd.DataFrame({"id": range(10), "label": ["a"] * 5 + ["b"] * 5})
    cand = ref.assign(label=["b"] * 5 + ["a"] * 5)
    config = sp.from_dict(
        {
            "output": {"type": "label"},
            "columns": {"score": "label"},
            "gates": {"transitions": {"max_rate": 0.01}},
        }
    )
    result = sp.compare(ref, cand, config)
    transition = gate(result, "transitions")
    assert not result.passed
    assert transition.value is None
    assert transition.details["checked_classes"] == 0
    assert transition.details["skipped_classes"] == ["a", "b"]


def test_segment_agreement_is_gated() -> None:
    ref = labels_frame()
    cand = ref.copy()
    loja = ref["canal"] == "loja"
    rng = np.random.default_rng(3)
    cand.loc[loja & (rng.random(len(ref)) < 0.02), "label"] = "oferta"
    g = gate(sp.compare(ref, cand, label_cfg(label_agreement={"min": 0.995})), "label_agreement")
    assert "canal=loja" in g.details["failing_segments"]


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_quality_difference(metric: str) -> None:
    ref = labels_frame()
    worse = ref.copy()
    rng = np.random.default_rng(4)
    worse.loc[rng.random(len(ref)) < 0.1, "label"] = "oferta"
    gates = {"quality_difference": {"metric": metric, "margin": 0.01}}
    assert gate(sp.compare(ref, ref.copy(), label_cfg(**gates)), "quality_difference").passed
    g = gate(sp.compare(ref, worse, label_cfg(**gates)), "quality_difference")
    assert not g.passed
    assert g.details["candidate"] < g.details["reference"]


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), rename=st.permutations(CLASSES))
def test_property_renaming_classes_consistently_keeps_the_verdict(
    seed: int, rename: list[str]
) -> None:
    ref = labels_frame(4000, seed)
    cand = ref.copy()
    rng = np.random.default_rng(seed)
    cand.loc[rng.random(len(ref)) < 0.01, "label"] = "sinal"
    mapping = dict(zip(CLASSES, rename, strict=True))
    cfg = label_cfg(label_agreement={"min": 0.98}, class_prevalence={"margin": 0.02})
    original = sp.compare(ref, cand, cfg)
    renamed = sp.compare(
        ref.assign(label=ref["label"].map(mapping), truth=ref["truth"].map(mapping)),
        cand.assign(label=cand["label"].map(mapping), truth=cand["truth"].map(mapping)).sample(
            frac=1.0, random_state=seed
        ),
        cfg,
    )
    assert original.verdict == renamed.verdict
    assert original.summary["agreement"] == renamed.summary["agreement"]


# --------------------------------------------------------------------------- probabilities


PROB_CFG = {
    "output": {"type": "probabilities"},
    "preset": "float-noise",
    "columns": {"truth": "truth"},
    "segments": ["canal"],
    "gates": {
        "tv_distance": {"max": 1e-5},
        "label_agreement": {"min": 0.99},
        "quality_difference": {"metric": "accuracy", "margin": 0.01},
    },
}


def test_probability_noise_passes_and_a_shift_fails() -> None:
    ref = probs_frame()
    rng = np.random.default_rng(5)
    cols = [f"p_{c}" for c in CLASSES]
    noisy = ref.copy()
    noisy[cols] = ref[cols] + rng.normal(0, 1e-8, (len(ref), len(cols)))
    noisy[cols] = noisy[cols].div(noisy[cols].sum(axis=1), axis=0)
    cfg = sp.from_dict(PROB_CFG)
    assert sp.compare(ref, noisy, cfg).passed

    shifted = ref.copy()
    shifted["p_cancelamento"] *= 1.5
    shifted[cols] = shifted[cols].div(shifted[cols].sum(axis=1), axis=0)
    report = sp.compare(ref, shifted, cfg)
    assert not report.passed
    assert "p_cancelamento" not in gate(report, "max_abs_diff").details["failing_classes"]
    assert "cancelamento" in gate(report, "max_abs_diff").details["failing_classes"]
    assert gate(report, "tv_distance").value > 1e-5


def test_preset_is_filtered_to_gates_that_apply_to_probabilities() -> None:
    names = [
        g.name for g in sp.compare(probs_frame(500), probs_frame(500), sp.from_dict(PROB_CFG)).gates
    ]
    assert "decision_flips" not in names
    assert "top_k_overlap" not in names
    assert "max_abs_diff" in names


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.drop(columns="p_oferta"), "different classes"),
        (lambda f: f.assign(p_oferta=f["p_oferta"] + 0.5), "do not sum to 1"),
        (
            lambda f: f.rename(columns={c: c.replace("p_", "q_") for c in f.columns}),
            "probability columns",
        ),
    ],
)
def test_probability_inputs_that_cannot_be_compared(mutate: Any, message: str) -> None:
    ref = probs_frame(500)
    with pytest.raises(sp.InputError, match=message):
        sp.compare(ref, mutate(ref.copy()), sp.from_dict(PROB_CFG))


# --------------------------------------------------------------------------- config, reports, CLI


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "output": {"type": "label"},
                "gates": {"decision_flips": {"thresholds": [0.5], "max_rate": 0}},
            },
            "do not apply",
        ),
        ({"output": {"type": "label"}, "preset": "exact"}, "presets"),
        (
            {"output": {"type": "label"}, "gates": {"quality_difference": {"margin": 0.01}}},
            "columns.truth",
        ),
        (
            {
                "output": {"type": "label"},
                "columns": {"truth": "t"},
                "gates": {"quality_difference": {"margin": 0.01, "metric": "auc"}},
            },
            "metric",
        ),
        ({"gates": {"label_agreement": {"min": 0.9}}}, "do not apply"),
    ],
)
def test_invalid_multiclass_configs(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(sp.ConfigError, match=message):
        sp.from_dict(raw)


@pytest.mark.parametrize("kind", ["label", "probabilities"])
def test_reports_validate_against_schema(kind: str) -> None:
    if kind == "label":
        ref, cfg = labels_frame(3000), label_cfg()
    else:
        ref, cfg = probs_frame(1000), sp.from_dict(PROB_CFG)
    doc = json.loads(sp.compare(ref, ref.copy(), cfg).to_json())
    errors = list(Draft202012Validator(SCHEMA).iter_errors(doc))
    assert not errors, [e.message for e in errors[:5]]


def test_untrusted_class_names_are_escaped_in_html_and_markdown() -> None:
    evil = "<img src=x onerror=alert(1)>`\n[click](http://evil)"
    ref = pd.DataFrame({"id": range(2000), "label": np.where(np.arange(2000) % 2, evil, "ok")})
    cand = ref.copy()
    cand.loc[:200, "label"] = "ok"
    cfg = sp.from_dict(
        {
            "output": {"type": "label"},
            "columns": {"score": "label"},
            "gates": {"transitions": {"max_rate": 0.01}},
        }
    )
    report = sp.compare(ref, cand, cfg)
    html = report.to_html()
    assert "<img" not in html
    assert "&lt;img" in html
    md = report.to_markdown()
    # A name must not close its code span early or break the line: every line keeps an even
    # number of backticks and the injected link never starts a Markdown line of its own.
    assert all(line.count("`") % 2 == 0 for line in md.splitlines())
    assert not any(line.lstrip().startswith("[click]") for line in md.splitlines())


def test_cli_label_and_probabilities(tmp_path: Path) -> None:
    ref, cand = labels_frame(5000), labels_frame(5000)
    rng = np.random.default_rng(6)
    cand.loc[rng.random(len(cand)) < 0.05, "label"] = "oferta"
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    ref.to_csv(a, index=False)
    ref.to_csv(b, index=False)
    cand.to_csv(c, index=False)
    base = [
        "compare",
        "--reference",
        str(a),
        "--output-type",
        "label",
        "--score",
        "label",
        "--truth",
        "truth",
        "--quiet",
    ]
    cfg = tmp_path / "labels.yaml"
    cfg.write_text(
        "version: 1\noutput: {type: label}\ncolumns: {score: label, truth: truth}\n"
        "gates:\n  label_agreement: {min: 0.99}\n  kappa: {min: 0.97}\n",
        "utf-8",
    )
    assert main([*base[:3], "--candidate", str(b), "--config", str(cfg), "--quiet"]) == EXIT_PASS
    assert main([*base[:3], "--candidate", str(c), "--config", str(cfg), "--quiet"]) == EXIT_FAIL

    p_ref = tmp_path / "p.csv"
    probs_frame(800).to_csv(p_ref, index=False)
    code = main(
        [
            "compare",
            "--reference",
            str(p_ref),
            "--candidate",
            str(p_ref),
            "--output-type",
            "probabilities",
            "--preset",
            "exact",
            "--quiet",
        ]
    )
    assert code == EXIT_PASS
    assert main([*base, "--candidate", str(p_ref)]) == EXIT_ERROR  # no 'label' column
