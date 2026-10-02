"""v0.3: multi-label outputs (a set of classes per row)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

import scoreparity as sp
from scoreparity.cli import EXIT_FAIL, EXIT_PASS, main
from scoreparity.stats_categorical import macro_f1_sets, macro_f1_sets_difference

TAGS = ["fatura", "reclamacao", "oferta", "sinal", "portabilidade"]
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "report-v1.schema.json").read_text("utf-8")
)
ALL_GATES = {
    "label_agreement": {"min": 0.99},
    "transitions": {"max_rate": 0.01},
    "class_prevalence": {"margin": 0.01},
    "quality_difference": {"metric": "macro_f1", "margin": 0.01},
}


def tagged(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sets = [
        "|".join(
            sorted(
                rng.choice(TAGS, rng.integers(0, 4), replace=False, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            )
        )
        for _ in range(n)
    ]
    frame: pd.DataFrame = pd.DataFrame({"id": np.arange(n), "tags": sets, "truth": sets})
    return frame


def cfg(gates: dict[str, Any] | None = None, **extra: Any) -> sp.ParityConfig:
    return sp.from_dict(
        {
            "output": {"type": "labels"},
            "columns": {"score": "tags", "truth": "truth"},
            "gates": ALL_GATES if gates is None else gates,
            **extra,
        }
    )


def gate(report: sp.Report, name: str) -> Any:
    return next(g for g in report.gates if g.name == name)


def add_tag(frame: pd.DataFrame, tag: str, share: float, seed: int = 1) -> pd.DataFrame:
    out = frame.copy()
    rng = np.random.default_rng(seed)
    pick = (rng.random(len(out)) < share) & ~out["tags"].str.contains(tag)
    out.loc[pick, "tags"] = [f"{t}|{tag}" if t else tag for t in out.loc[pick, "tags"]]
    result: pd.DataFrame = out
    return result


# --------------------------------------------------------------------------- parsing


def test_order_duplicates_and_whitespace_do_not_matter() -> None:
    ref = pd.DataFrame({"id": range(4), "tags": ["a|b", "b", "", "c"]})
    cand = pd.DataFrame({"id": range(4), "tags": [" b | a |a", "b|", "", ["c"]]})
    report = sp.compare(
        ref, cand, cfg({"label_agreement": {"min": 0.0}}, columns={"score": "tags"})
    )
    assert report.summary["agreement"] == 1.0
    assert report.summary["jaccard_mean"] == 1.0


def test_lists_numbers_and_custom_separator() -> None:
    ref = pd.DataFrame({"id": range(3), "tags": [["x", "y"], np.array(["z"]), 7]})
    cand = pd.DataFrame({"id": range(3), "tags": ["y;x", "z", "7"]})
    config = sp.from_dict(
        {
            "output": {"type": "labels", "label_separator": ";"},
            "columns": {"score": "tags"},
            "gates": {"label_agreement": {"min": 0.0}},
        }
    )
    assert sp.compare(ref, cand, config).summary["agreement"] == 1.0


def test_transitions_fail_when_no_class_has_enough_rows() -> None:
    ref = pd.DataFrame({"id": range(10), "tags": ["a"] * 5 + ["b"] * 5})
    cand = ref.assign(tags=["b"] * 5 + ["a"] * 5)
    result = sp.compare(
        ref, cand, cfg({"transitions": {"max_rate": 0.01}}, columns={"score": "tags"})
    )
    transition = gate(result, "transitions")
    assert not result.passed
    assert transition.value is None
    assert transition.details["checked_checks"] == 0
    assert transition.details["skipped_checks"]


def test_missing_is_a_missing_output_and_empty_is_the_empty_set() -> None:
    ref = pd.DataFrame({"id": range(3), "tags": ["a", "", None]})
    cand = pd.DataFrame({"id": range(3), "tags": [None, "", None]})
    report = sp.compare(ref, cand, cfg({}, columns={"score": "tags"}))
    assert gate(report, "nonfinite").value == 1
    assert report.summary["n_nonfinite_both"] == 1
    assert report.summary["n_matched_finite"] == 1  # the empty sets


def test_normalisation_and_invalid_labels_in_sets() -> None:
    config = sp.from_dict(
        {
            "output": {
                "type": "labels",
                "normalize": {"lowercase": True, "map": {"bill": "fatura"}, "allowed": TAGS},
            },
            "columns": {"score": "tags"},
            "gates": {"invalid_rate": {"max": 0.6}},
        }
    )
    ref = pd.DataFrame({"id": range(3), "tags": ["fatura", "sinal", "oferta"]})
    cand = pd.DataFrame({"id": range(3), "tags": ["BILL", "sinal|banana", None]})
    report = sp.compare(ref, cand, config)
    assert report.summary["agreement"] == pytest.approx(1 / 3)
    assert gate(report, "invalid_rate").details["candidate_invalid"] == 2
    assert report.summary["label_changes"]["__invalid__"]["added"] == 2


def test_a_list_item_containing_the_separator_is_refused() -> None:
    ref = pd.DataFrame({"id": [1], "tags": [["a|b"]]})
    with pytest.raises(sp.InputError, match="separator"):
        sp.compare(ref, ref.copy(), cfg({}, columns={"score": "tags"}))


# --------------------------------------------------------------------------- gates


def test_identical_sets_pass_and_an_added_label_fails() -> None:
    ref = tagged()
    assert sp.compare(ref, ref.copy(), cfg()).passed
    report = sp.compare(ref, add_tag(ref, "oferta", 0.03), cfg())
    assert gate(report, "transitions").details["failing_pairs"] == ["(added) -> oferta"]
    assert gate(report, "class_prevalence").details["failing_classes"] == ["oferta"]
    assert not gate(report, "label_agreement").passed
    assert report.summary["label_changes"]["oferta"]["removed"] == 0


def test_a_removed_label_is_named() -> None:
    ref = tagged()
    cand = ref.copy()
    rng = np.random.default_rng(2)
    has = cand["tags"].str.contains("sinal") & (rng.random(len(cand)) < 0.1)
    cand.loc[has, "tags"] = [
        "|".join(t for t in s.split("|") if t != "sinal") for s in cand.loc[has, "tags"]
    ]
    report = sp.compare(ref, cand, cfg({"transitions": {"max_rate": 0.02}}))
    assert gate(report, "transitions").details["failing_pairs"] == ["sinal -> (removed)"]


def test_per_class_limits_for_sets() -> None:
    ref = tagged()
    cand = add_tag(ref, "portabilidade", 0.01)
    strict = {"transitions": {"max_rate": 0.005}}
    lenient = {"transitions": {"max_rate": 0.005, "per_class": {"portabilidade": 0.05}}}
    assert not sp.compare(ref, cand, cfg(strict)).passed
    assert sp.compare(ref, cand, cfg(lenient)).passed


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_quality_difference_for_sets(metric: str) -> None:
    ref = tagged()
    gates = {"quality_difference": {"metric": metric, "margin": 0.01}}
    assert sp.compare(ref, ref.copy(), cfg(gates)).passed
    g = gate(sp.compare(ref, add_tag(ref, "oferta", 0.2), cfg(gates)), "quality_difference")
    assert not g.passed
    assert g.details["candidate"] < g.details["reference"]


def test_macro_f1_sets_matches_a_per_class_loop() -> None:
    rng = np.random.default_rng(3)
    truth = rng.random((500, 4)) < 0.3
    pred = truth ^ (rng.random((500, 4)) < 0.1)
    f1s = []
    for j in range(4):
        tp = int((truth[:, j] & pred[:, j]).sum())
        fp = int((~truth[:, j] & pred[:, j]).sum())
        fn = int((truth[:, j] & ~pred[:, j]).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn))
    assert macro_f1_sets(truth, pred) == pytest.approx(float(np.mean(f1s)))


def test_macro_f1_sets_bootstrap_matches_a_row_bootstrap() -> None:
    rng = np.random.default_rng(4)
    n = 400
    truth = rng.random((n, 3)) < 0.4
    ref = truth ^ (rng.random((n, 3)) < 0.1)
    cand = truth ^ (rng.random((n, 3)) < 0.15)
    ones = np.ones(n, dtype=np.int64)
    res = macro_f1_sets_difference(truth, ref, cand, ones, alpha=0.05, n_boot=3000)
    deltas = []
    for _ in range(3000):
        i = rng.integers(0, n, n)
        deltas.append(macro_f1_sets(truth[i], cand[i]) - macro_f1_sets(truth[i], ref[i]))
    low, high = np.quantile(deltas, [0.05, 0.95])
    assert res.low == pytest.approx(low, abs=0.01)
    assert res.high == pytest.approx(high, abs=0.01)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"output": {"type": "labels"}, "gates": {"kappa": {"min": 0.9}}}, "do not apply"),
        ({"output": {"type": "labels"}, "preset": "exact"}, "presets"),
        ({"output": {"type": "labels", "label_separator": " "}}, "label_separator"),
    ],
)
def test_invalid_multilabel_configs(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(sp.ConfigError, match=message):
        sp.from_dict(raw)


# --------------------------------------------------------------------------- reports, CLI


def test_reports_render_sets_safely() -> None:
    ref = tagged(3000)
    report = sp.compare(ref, add_tag(ref, "oferta", 0.05), cfg(examples=10))
    doc = json.loads(report.to_json())
    assert not list(Draft202012Validator(SCHEMA).iter_errors(doc))
    md = report.to_markdown()
    lines = md.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Examples by id"))
    table = [line for line in lines[start:] if line.startswith("| `")]
    assert table
    for line in table:  # escaped pipes stay inside their cell: 4 cell borders per row
        assert len(re.findall(r"(?<!\\)\|", line)) == 4
    assert "Same set of labels" in md
    html = report.to_html()
    assert "Which labels changed?" in html


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_property_shuffling_rows_and_label_order_keeps_the_report(seed: int) -> None:
    ref = tagged(1500, seed)
    cand = add_tag(ref, "sinal", 0.01, seed)
    shuffled = cand.sample(frac=1.0, random_state=seed)
    shuffled["tags"] = ["|".join(reversed(t.split("|"))) for t in shuffled["tags"]]
    a = sp.compare(ref, cand, cfg())
    b = sp.compare(ref, shuffled, cfg())
    assert a.verdict == b.verdict
    assert a.summary["label_changes"] == b.summary["label_changes"]


def test_cli_multilabel(tmp_path: Path) -> None:
    ref = tagged(3000)
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    ref.to_csv(a, index=False)
    ref.to_csv(b, index=False)
    add_tag(ref, "oferta", 0.05).to_csv(c, index=False)
    path = tmp_path / "labels.yaml"
    path.write_text(
        "version: 1\noutput: {type: labels}\ncolumns: {score: tags}\n"
        "gates: {label_agreement: {min: 0.98}, transitions: {max_rate: 0.01}}\n",
        "utf-8",
    )
    args = ["compare", "--reference", str(a), "--config", str(path), "--quiet"]
    assert main([*args, "--candidate", str(b)]) == EXIT_PASS
    assert main([*args, "--candidate", str(c)]) == EXIT_FAIL


def test_parquet_list_columns(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    ref = pd.DataFrame({"id": range(3), "tags": [["a", "b"], [], ["c"]]})
    path = tmp_path / "t.parquet"
    ref.to_parquet(path)
    report = sp.compare_files(
        path, path, cfg({"label_agreement": {"min": 0.0}}, columns={"score": "tags"})
    )
    assert report.summary["agreement"] == 1.0
    assert report.summary["prevalence"]["a"]["reference"] == pytest.approx(1 / 3)
