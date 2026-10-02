"""v0.3: structured outputs (one JSON object per row, compared field by field)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

import scoreparity as sp
from scoreparity.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "report-v1.schema.json").read_text("utf-8")
)
MOTIVOS = ["fatura", "sinal", "oferta"]
FIELDS: dict[str, Any] = {
    "motivo": {
        "type": "label",
        "normalize": {"allowed": MOTIVOS},
        "gates": {"label_agreement": {"min": 0.99}, "transitions": {"max_rate": 0.01}},
    },
    "valor": {"type": "score", "gates": {"max_abs_diff": {"max": 0.01}}},
    "urgente": {"type": "label", "gates": {"label_agreement": {"min": 0.99}}},
    "tags": {"type": "labels", "required": False, "gates": {"label_agreement": {"min": 0.99}}},
}


def documents(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    motivo = rng.choice(MOTIVOS, n, p=[0.5, 0.3, 0.2])
    valor = np.round(rng.gamma(2, 60, n), 2)
    urgente = rng.random(n) < 0.2
    docs = [
        json.dumps(
            {"motivo": m, "valor": float(v), "urgente": bool(u), "tags": ["a", "b"][: u + 1]}
        )
        for m, v, u in zip(motivo, valor, urgente, strict=True)
    ]
    frame: pd.DataFrame = pd.DataFrame(
        {
            "id": np.arange(n),
            "out": docs,
            "canal": rng.choice(["app", "web"], n),
            "motivo_real": motivo,
        }
    )
    return frame


def cfg(fields: dict[str, Any] | None = None, **extra: Any) -> sp.ParityConfig:
    return sp.from_dict(
        {
            "output": {"type": "structured", "fields": fields or FIELDS},
            "columns": {"score": "out"},
            **extra,
        }
    )


def edit(frame: pd.DataFrame, rows: Any, change: Any) -> pd.DataFrame:
    out = frame.copy()
    out.loc[rows, "out"] = [json.dumps(change(json.loads(d))) for d in out.loc[rows, "out"]]
    result: pd.DataFrame = out
    return result


def gate(report: sp.Report, label: str) -> Any:
    return next(g for g in report.gates if g.label == label)


def test_identical_documents_pass_every_field() -> None:
    ref = documents()
    report = sp.compare(ref, ref.copy(), cfg())
    assert report.passed, report.failed_gates
    assert set(report.summary["fields"]) == set(FIELDS)
    assert report.summary["schema_valid"] == {"reference": 1.0, "candidate": 1.0}


def test_a_field_change_fails_that_field_only() -> None:
    ref = documents()
    oferta = ref.index[ref["out"].str.contains('"oferta"')][:60]
    cand = edit(ref, oferta, lambda d: {**d, "motivo": "fatura"})
    report = sp.compare(ref, cand, cfg())
    assert report.failed_gates == ["motivo.label_agreement", "motivo.transitions"]
    assert gate(report, "motivo.transitions").details["failing_pairs"] == ["oferta -> fatura"]
    assert gate(report, "motivo.transitions").field == "motivo"


def test_score_fields_need_json_numbers() -> None:
    ref = documents()  # enough rows for the motivo transitions to be verifiable
    cand = edit(ref, ref.index[:3], lambda d: {**d, "valor": str(d["valor"])})
    report = sp.compare(ref, cand, cfg())
    assert gate(report, "valor.nonfinite").value == 3
    assert report.failed_gates == ["valor.nonfinite"]


def test_invalid_documents_are_judged_once_at_the_top() -> None:
    ref = documents(2000)
    cand = ref.copy()
    cand.loc[:4, "out"] = ["not json", "[1, 2]", "{", None, json.dumps({"valor": 1.0})]
    report = sp.compare(ref, cand, cfg(gates={"schema_valid_rate": {"max": 0.01}}))
    assert report.summary["n_nonfinite_mismatch"] == 5  # exactly one side invalid
    assert gate(report, "schema_valid_rate").details["candidate_invalid"] == 5
    assert report.failed_gates == ["nonfinite"]  # fields only see rows valid on both sides
    assert report.summary["fields"]["motivo"]["n_matched_finite"] == 1995


def test_optional_fields_may_be_missing_but_count_as_missing_outputs() -> None:
    ref = documents(500)
    cand = edit(ref, ref.index[:2], lambda d: {k: v for k, v in d.items() if k != "tags"})
    report = sp.compare(ref, cand, cfg())
    assert report.summary["schema_valid"]["candidate"] == 1.0  # tags is optional
    assert gate(report, "tags.nonfinite").value == 2


def test_booleans_dicts_and_hostile_nesting() -> None:
    ref = pd.DataFrame(
        {
            "id": range(3),
            "out": [{"flag": True}, json.dumps({"flag": False}), json.dumps({"flag": True})],
        }
    )
    cand = ref.copy()
    cand.loc[2, "out"] = "[" * 100_000  # deeply nested: invalid, not a crash
    config = sp.from_dict(
        {
            "output": {"type": "structured", "fields": {"flag": {"type": "label"}}},
            "columns": {"score": "out"},
            "gates": {"nonfinite": None},
        }
    )
    report = sp.compare(ref, cand, config)
    assert report.summary["fields"]["flag"]["classes"] == ["false", "true"]
    assert report.summary["schema_valid"]["candidate"] == pytest.approx(2 / 3)


def test_field_segments_and_truth() -> None:
    ref = documents()
    web = ref.index[(ref["canal"] == "web") & ref["out"].str.contains('"sinal"')][:40]
    cand = edit(ref, web, lambda d: {**d, "motivo": "oferta"})
    fields = {
        "motivo": {
            "type": "label",
            "truth": "motivo_real",
            "gates": {
                "label_agreement": {"min": 0.99},
                "quality_difference": {"metric": "accuracy", "margin": 0.005},
            },
        }
    }
    report = sp.compare(ref, cand, cfg(fields, segments=["canal"]))
    agreement = gate(report, "motivo.label_agreement")
    assert agreement.details["failing_segments"] == ["canal=web"]
    assert report.summary["fields"]["motivo"]["accuracy"]["reference"] == 1.0
    assert not gate(report, "motivo.quality_difference").passed


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"output": {"type": "structured"}}, "at least one field"),
        ({"output": {"type": "structured", "fields": {"x": {"type": "probabilities"}}}}, "type"),
        (
            {"output": {"type": "structured", "fields": {"x": {"gates": {"coverage": {}}}}}},
            "top-level",
        ),
        (
            {"output": {"type": "structured", "fields": {"x": {"gates": {"oops": {}}}}}},
            "unknown gates",
        ),
        (
            {
                "output": {
                    "type": "structured",
                    "fields": {"x": {"gates": {"max_abs_diff": {"max": 1}}}},
                }
            },
            r"output\.fields\.x: gates \['max_abs_diff'\] do not apply",
        ),
        (
            {
                "output": {
                    "type": "structured",
                    "fields": {"x": {"gates": {"quality_difference": {"margin": 0.1}}}},
                }
            },
            "columns.truth",
        ),
        ({"output": {"type": "label", "fields": {"x": {}}}}, "structured"),
        ({"output": {"type": "structured", "fields": {"x": {}}}, "preset": "exact"}, "presets"),
        (
            {"output": {"type": "structured", "fields": {"x": {}}}, "gates": {"kappa": {"min": 0}}},
            "do not apply",
        ),
    ],
)
def test_invalid_structured_configs(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(sp.ConfigError, match=message):
        sp.from_dict(raw)


def test_structured_reports_are_valid_and_escaped() -> None:
    evil = "<b>x</b>`\n## injected"
    ref = pd.DataFrame({"id": range(300), "out": [json.dumps({evil: "a"})] * 300})
    cand = ref.copy()
    cand.loc[:30, "out"] = json.dumps({evil: "b"})
    config = sp.from_dict(
        {
            "output": {
                "type": "structured",
                "fields": {evil: {"type": "label", "gates": {"label_agreement": {"min": 0.99}}}},
            },
            "columns": {"score": "out"},
            "examples": 5,
        }
    )
    report = sp.compare(ref, cand, config)
    doc = json.loads(report.to_json())
    assert not list(Draft202012Validator(SCHEMA).iter_errors(doc))
    html = report.to_html()
    assert "<b>x</b>" not in html
    md = report.to_markdown()
    assert all(line.count("`") % 2 == 0 for line in md.splitlines())
    assert not any(line.startswith("## injected") for line in md.splitlines())
    names = [c.get("name") for c in ET.fromstring(report.to_junit()).iter("testcase")]
    assert f"{evil}.label_agreement" in names


def test_cli_structured(tmp_path: Path) -> None:
    ref = documents(1000)
    oferta = ref.index[ref["out"].str.contains('"oferta"')][:50]
    changed = edit(ref, oferta, lambda d: {**d, "motivo": "sinal"})
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    ref.to_csv(a, index=False)
    ref.to_csv(b, index=False)
    changed.to_csv(c, index=False)
    path = tmp_path / "s.yaml"
    path.write_text(
        "version: 1\ncolumns: {score: out}\noutput:\n  type: structured\n  fields:\n"
        "    motivo: {type: label, gates: {label_agreement: {min: 0.99}}}\n"
        "    valor: {type: score, gates: {max_abs_diff: {max: 0.001}}}\n",
        "utf-8",
    )
    args = ["compare", "--reference", str(a), "--config", str(path), "--quiet"]
    assert main([*args, "--candidate", str(b)]) == EXIT_PASS
    assert main([*args, "--candidate", str(c)]) == EXIT_FAIL
    missing_column = tmp_path / "m.yaml"
    missing_column.write_text(path.read_text("utf-8").replace("score: out", "score: nope"), "utf-8")
    bad = ["compare", "--reference", str(a), "--candidate", str(b), "--quiet"]
    assert main([*bad, "--config", str(missing_column)]) == EXIT_ERROR
