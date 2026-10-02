"""v0.3: label normalisation, invalid answers, examples by id and sample-size planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from scipy import stats

import scoreparity as sp
from scoreparity.cli import EXIT_ERROR, EXIT_PASS, main
from scoreparity.gates_categorical import rows_needed
from scoreparity.planning import plan_sample, power_at
from scoreparity.stats_categorical import clopper_pearson

ALLOWED = ["fatura", "sinal", "oferta", "cancelamento"]
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "report-v1.schema.json").read_text("utf-8")
)


def llm_cfg(gates: dict[str, Any] | None = None, **extra: Any) -> sp.ParityConfig:
    return sp.from_dict(
        {
            "output": {
                "type": "label",
                "normalize": {
                    "lowercase": True,
                    "map": {"Cancelar": "cancelamento", "cancel": "cancelamento"},
                    "allowed": ALLOWED,
                },
            },
            "columns": {"score": "answer"},
            "gates": gates or {"invalid_rate": {"max": 0.01}},
            **extra,
        }
    )


def answers(n: int = 4000, seed: int = 0, invalid: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(ALLOWED, n, p=[0.4, 0.3, 0.2, 0.1]).astype(object)
    bad = rng.random(n) < invalid
    labels[bad] = "Desculpe, não posso ajudar."
    frame: pd.DataFrame = pd.DataFrame({"id": np.arange(n), "answer": labels})
    return frame


def gate(report: sp.Report, name: str) -> Any:
    return next(g for g in report.gates if g.name == name)


# --------------------------------------------------------------------------- normalisation


def test_normalisation_maps_case_synonyms_and_invalid_answers() -> None:
    ref = pd.DataFrame({"id": range(5), "answer": ["fatura", "cancelamento", "sinal", "x", None]})
    cand = pd.DataFrame(
        {"id": range(5), "answer": [" FATURA ", "Cancelar", "cancel", "", "I cannot help"]}
    )
    report = sp.compare(ref, cand, llm_cfg({"label_agreement": {"min": 0.0}}))
    cm = report.summary["confusion"]
    assert cm["classes"] == ["__invalid__", *sorted(ALLOWED)]
    assert report.summary["agreement"] == pytest.approx(4 / 5)  # sinal -> cancelamento differs
    # Empty and missing answers are invalid answers, not missing outputs.
    assert gate(report, "nonfinite").value == 0
    assert report.summary["invalid_share"] == {"reference": 0.4, "candidate": 0.4}


def test_without_allowed_classes_only_case_and_synonyms_change() -> None:
    cfg = sp.from_dict(
        {
            "output": {"type": "label", "normalize": {"lowercase": True, "map": {"no": "não"}}},
            "columns": {"score": "answer"},
            "gates": {"label_agreement": {"min": 0.0}},
        }
    )
    ref = pd.DataFrame({"id": range(3), "answer": ["SIM", "não", ""]})
    cand = pd.DataFrame({"id": range(3), "answer": ["sim", "No", "talvez"]})
    report = sp.compare(ref, cand, cfg)
    assert report.summary["n_nonfinite_mismatch"] == 1  # "" stays a missing label
    assert report.summary["agreement"] == 1.0


def test_class_names_that_look_like_numbers_in_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "version: 1\noutput:\n  type: label\n  normalize: {allowed: [1, 2], map: {um: 1}}\n"
        "columns: {score: answer}\ngates: {invalid_rate: {max: 0.5}}\n",
        "utf-8",
    )
    cfg = sp.load(path)
    assert cfg.output.normalize is not None
    assert cfg.output.normalize.allowed == ("1", "2")
    ref = pd.DataFrame({"id": range(4), "answer": [1, 2, 1, 2]})
    cand = pd.DataFrame({"id": range(4), "answer": ["um", "2", 1, 2.5]})
    report = sp.compare(ref, cand, cfg)
    assert report.summary["invalid_share"]["candidate"] == 0.25  # "2.5" is not a class


def test_truth_outside_the_allowed_classes_is_an_input_error() -> None:
    ref = answers(100).assign(truth="fatura")
    ref.loc[3, "truth"] = "typo"
    cfg = sp.from_dict(
        {
            "output": {"type": "label", "normalize": {"allowed": ALLOWED}},
            "columns": {"score": "answer", "truth": "truth"},
            "gates": {"quality_difference": {"margin": 0.1}},
        }
    )
    with pytest.raises(sp.InputError, match=r"outside output\.normalize\.allowed"):
        sp.compare(ref, ref.copy(), cfg)


def test_truth_is_normalised_like_the_answers() -> None:
    ref = answers(2000)
    ref["truth"] = ref["answer"].str.upper()
    cfg = sp.from_dict(
        {
            "output": {"type": "label", "normalize": {"lowercase": True, "allowed": ALLOWED}},
            "columns": {"score": "answer", "truth": "truth"},
            "gates": {"quality_difference": {"margin": 0.01}},
        }
    )
    report = sp.compare(ref, ref.copy(), cfg)
    assert report.summary["accuracy"] == {"reference": 1.0, "candidate": 1.0}


# --------------------------------------------------------------------------- invalid_rate


def test_invalid_rate_passes_when_rare_and_fails_when_frequent() -> None:
    ref = answers(5000)
    assert sp.compare(ref, answers(5000, invalid=0.0005), llm_cfg()).passed
    report = sp.compare(ref, answers(5000, invalid=0.03), llm_cfg())
    g = gate(report, "invalid_rate")
    assert not g.passed
    assert g.details["candidate_invalid"] > 0
    assert g.details["ci"][1] > 0.01


def test_invalid_rate_explains_when_the_sample_is_too_small() -> None:
    ref = answers(200)
    report = sp.compare(ref, ref.copy(), llm_cfg({"invalid_rate": {"max": 0.001}}))
    g = gate(report, "invalid_rate")
    assert not g.passed
    assert g.details["rows_needed"] == 2995
    md = report.to_markdown()
    assert "no invalid answer observed" in md
    assert "plan-sample" in md


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"output": {"type": "label"}, "gates": {"invalid_rate": {"max": 0.1}}}, "allowed"),
        ({"output": {"type": "score", "normalize": {}}}, "output.normalize applies"),
        (
            {"output": {"type": "label", "normalize": {"allowed": ["a"], "map": {"x": "b"}}}},
            "not allowed",
        ),
        (
            {"output": {"type": "label", "normalize": {"allowed": ["a", "__invalid__"]}}},
            "must not be one of",
        ),
        (
            {"output": {"type": "label", "normalize": {"lowercase": True, "allowed": ["a", "A"]}}},
            "duplicates",
        ),
        ({"output": {"type": "label", "normalize": {"allowed": ["a"], "typo": 1}}}, "unknown"),
        ({"output": {"type": "label", "normalize": {"map": ["a"]}}}, "mapping"),
        ({"examples": 5000}, "examples"),
        ({"examples": -1}, "examples"),
    ],
)
def test_invalid_llm_configs(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(sp.ConfigError, match=message):
        sp.from_dict(raw)


# --------------------------------------------------------------------------- examples


def test_examples_are_off_by_default() -> None:
    ref = answers(500)
    cand = ref.assign(answer="fatura")
    assert "examples" not in sp.compare(ref, cand, llm_cfg()).summary


def test_label_examples_cover_every_kind_of_change() -> None:
    ref = answers(3000)
    cand = ref.copy()
    common = ref.index[ref["answer"] == "fatura"][:200]
    cand.loc[common, "answer"] = "sinal"
    rare = ref.index[ref["answer"] == "oferta"][:2]
    cand.loc[rare, "answer"] = "cancelamento"
    report = sp.compare(ref, cand, llm_cfg(examples=6))
    examples = report.summary["examples"]
    assert len(examples) == 6
    kinds = {(e["reference"], e["candidate"]) for e in examples}
    assert kinds == {("fatura", "sinal"), ("oferta", "cancelamento")}
    assert all(isinstance(e["id"], str) for e in examples)
    assert examples[0]["reference"] == "fatura"  # the most frequent change comes first


def test_score_and_probability_examples_rank_the_largest_differences() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"id": [f"c{i}" for i in range(1000)], "score": rng.random(1000)})
    cand = ref.copy()
    cand.loc[[10, 20], "score"] += [0.3, 0.1]
    cfg = sp.from_dict({"preset": "exact", "examples": 3})
    examples = sp.compare(ref, cand, cfg).summary["examples"]
    assert [e["id"] for e in examples] == ["c10", "c20"]  # identical rows are never listed
    assert examples[0]["difference"] == pytest.approx(0.3)

    probs = pd.DataFrame({"id": range(300), "p_a": rng.random(300)})
    probs["p_b"] = 1 - probs["p_a"]
    moved = probs.copy()
    moved.loc[7, ["p_a", "p_b"]] = [1.0, 0.0]
    cfg_p = sp.from_dict({"output": {"type": "probabilities"}, "examples": 1})
    (top,) = sp.compare(probs, moved, cfg_p).summary["examples"]
    assert top["id"] == "7"
    assert top["candidate"] == "a"


def test_examples_with_hostile_ids_are_escaped() -> None:
    evil = "<script>alert(1)</script>`|\n## injected"
    ref = pd.DataFrame({"id": [evil, "ok"] * 1, "answer": ["fatura", "sinal"]})
    cand = ref.assign(answer=["sinal", "sinal"])
    report = sp.compare(ref, cand, llm_cfg({"label_agreement": {"min": 0.0}}, examples=5))
    html = report.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    md = report.to_markdown()
    assert not any(line.startswith("## injected") for line in md.splitlines())
    assert all(line.count("`") % 2 == 0 for line in md.splitlines())


def test_llm_reports_validate_against_schema() -> None:
    ref = answers(3000)
    report = sp.compare(ref, answers(3000, seed=1, invalid=0.01), llm_cfg(examples=10))
    doc = json.loads(report.to_json())
    errors = list(Draft202012Validator(SCHEMA).iter_errors(doc))
    assert not errors, [e.message for e in errors[:5]]
    score = sp.compare(
        pd.DataFrame({"id": range(50), "score": np.linspace(0, 1, 50)}),
        pd.DataFrame({"id": range(50), "score": np.linspace(0, 1, 50) + 1e-9}),
        sp.from_dict({"preset": "float-noise", "examples": 3}),
    )
    errors = list(Draft202012Validator(SCHEMA).iter_errors(json.loads(score.to_json())))
    assert not errors, [e.message for e in errors[:5]]


# --------------------------------------------------------------------------- plan-sample


@pytest.mark.parametrize("n", [50, 299, 1000, 7777])
@pytest.mark.parametrize("max_rate", [0.001, 0.01, 0.05])
def test_planning_rule_matches_the_gate(n: int, max_rate: float) -> None:
    """The gate's Clopper-Pearson decision and the planner's binomial rule agree."""
    from scoreparity.planning import _max_events

    k_star = int(_max_events(np.array([n]), max_rate, 0.05)[0])
    for k in range(max(0, k_star - 3), k_star + 4):
        if k > n:
            break
        passes = clopper_pearson(k, n, 0.05)[1] <= max_rate
        assert passes == (k <= k_star), (n, k, k_star)


def test_zero_expected_rate_matches_rows_needed() -> None:
    for max_rate in (0.05, 0.01, 0.001):
        assert plan_sample(max_rate).rows == rows_needed(max_rate, 0.05)


@pytest.mark.parametrize(
    ("max_rate", "expected", "power"),
    [(0.01, 0.002, 0.8), (0.02, 0.01, 0.9), (0.005, 0.001, 0.8), (0.1, 0.05, 0.95)],
)
def test_planned_sizes_reach_the_power_by_simulation(
    max_rate: float, expected: float, power: float
) -> None:
    plan = plan_sample(max_rate, expected, power=power)
    rng = np.random.default_rng(0)
    for n in (plan.rows, plan.rows + 17, plan.rows + 150):
        events = rng.binomial(n, expected, 20_000)
        upper = stats.beta.ppf(0.95, events + 1, n - events)
        passed = (upper <= max_rate).mean()
        assert passed >= power - 0.01, (n, passed)
    exact = [power_at(n, max_rate, expected) for n in range(plan.rows, plan.rows + 400)]
    assert min(exact) >= power
    assert plan.smallest_rows <= plan.rows
    assert power_at(plan.smallest_rows, max_rate, expected) >= power


def test_plan_sample_costs_and_class_share() -> None:
    plan = plan_sample(0.01, 0.002, class_share=0.05, cost_per_row=0.002)
    assert plan.total_rows is not None
    assert plan.total_rows == int(np.ceil(plan.rows / 0.05))
    assert plan.estimated_cost == pytest.approx(plan.total_rows * 2 * 0.002)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rate": 0.01, "expected_rate": 0.01},
        {"max_rate": 0.0},
        {"max_rate": 0.01, "power": 1.0},
        {"max_rate": 0.01, "class_share": 0.0},
    ],
)
def test_impossible_plans_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        plan_sample(**kwargs)


def test_cli_plan_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "plan.json"
    code = main(
        [
            "plan-sample",
            "--min-agreement",
            "0.98",
            "--expected-agreement",
            "0.995",
            "--cost-per-row",
            "0.01",
            "--json",
            str(out),
        ]
    )
    assert code == EXIT_PASS
    assert "rows needed" in capsys.readouterr().out
    plan = json.loads(out.read_text("utf-8"))
    assert plan["max_rate"] == pytest.approx(0.02)
    assert plan["estimated_cost"] == pytest.approx(plan["rows"] * 2 * 0.01)
    assert main(["plan-sample", "--max-rate", "0.01", "--expected-rate", "0.02"]) == EXIT_ERROR


def test_cli_examples_flag(tmp_path: Path) -> None:
    ref, cand = answers(400), answers(400, seed=3)
    a, b, cfg, report = (tmp_path / n for n in ("a.csv", "b.csv", "c.yaml", "r.json"))
    ref.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    cfg.write_text(
        "version: 1\noutput: {type: label, normalize: {allowed: [fatura, sinal, oferta, "
        "cancelamento]}}\ncolumns: {score: answer}\ngates: {invalid_rate: {max: 0.05}}\n",
        "utf-8",
    )
    args = ["compare", "--reference", str(a), "--candidate", str(b), "--config", str(cfg)]
    assert main([*args, "--examples", "4", "--json", str(report), "--quiet"]) == EXIT_PASS
    assert len(json.loads(report.read_text("utf-8"))["summary"]["examples"]) == 4
    assert main([*args, "--examples", "100000", "--quiet"]) == EXIT_ERROR
