"""v0.3: replicas per id, the stability gate, per-class transition limits and label noise."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from jsonschema import Draft202012Validator

import scoreparity as sp
from scoreparity.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main
from scoreparity.noise import measure_noise, suggested_config_yaml

CLASSES = ["fatura", "sinal", "oferta", "atendimento", "cancelamento"]
SHARES = [0.35, 0.25, 0.2, 0.15, 0.05]
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "report-v1.schema.json").read_text("utf-8")
)
BASE = {"output": {"type": "label"}, "columns": {"score": "label"}}


def truth(n: int, seed: int) -> np.ndarray:
    # A separate stream, so the runs below never reuse the draws that chose the labels.
    return np.random.default_rng([seed, 7919]).choice(CLASSES, n, p=SHARES)


def run(
    labels: np.ndarray, noise: float, rng: np.random.Generator, shift: float = 0.0
) -> pd.DataFrame:
    """One run of a classifier: each answer is resampled with probability `noise`."""
    out = labels.copy()
    resampled = rng.random(len(labels)) < noise
    out[resampled] = rng.choice(CLASSES, int(resampled.sum()))
    if shift:
        moved = (labels == "fatura") & (rng.random(len(labels)) < shift)
        out[moved] = "cancelamento"
    frame: pd.DataFrame = pd.DataFrame({"id": np.arange(len(labels)), "label": out})
    return frame


def gate(report: sp.Report, name: str) -> Any:
    return next(g for g in report.gates if g.name == name)


# --------------------------------------------------------------------------- replicas


def replicated(labels_per_id: dict[int, list[str]]) -> pd.DataFrame:
    rows = [
        {"id": i, "replica": r, "label": label}
        for i, labels in labels_per_id.items()
        for r, label in enumerate(labels)
    ]
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


def test_replicas_vote_for_the_majority_class() -> None:
    ref = replicated({1: ["a", "a", "b"], 2: ["b", "b", "b"], 3: ["a", "b", "c", "c"]})
    cand = replicated({1: ["a", "b", "a"], 2: ["c", "b", "b"], 3: ["c", "c", "c"]})
    cfg = sp.from_dict({**BASE, "columns": {"score": "label", "replica": "replica"}, "gates": {}})
    report = sp.compare(ref, cand, cfg)
    assert report.summary["n_reference"] == 3  # ids, not rows
    assert report.summary["agreement"] == 1.0  # a/a, b/b, c/c
    stab = report.summary["stability"]
    assert stab["reference"]["unstable_share"] == pytest.approx(2 / 3)
    assert stab["candidate"]["mean"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_ties_are_broken_alphabetically_and_flagged_unstable() -> None:
    ref = replicated({1: ["b", "a"]})
    cand = replicated({1: ["a", "a"]})
    cfg = sp.from_dict({**BASE, "columns": {"score": "label", "replica": "replica"}})
    report = sp.compare(ref, cand, cfg)
    assert report.summary["agreement"] == 1.0
    assert report.summary["stability"]["reference"]["mean"] == 0.5


def test_changes_on_unstable_ids_are_reported() -> None:
    ref = replicated({i: ["a", "a", "a"] if i % 2 else ["a", "b", "b"] for i in range(100)})
    cand = replicated({i: ["a", "a", "a"] for i in range(100)})
    cfg = sp.from_dict({**BASE, "columns": {"score": "label", "replica": "replica"}})
    report = sp.compare(ref, cand, cfg)
    assert report.summary["stability"]["changes_on_unstable_ids"] == 1.0


def test_replicas_are_normalised_before_voting() -> None:
    ref = replicated({1: ["Fatura", "fatura ", "I can't"], 2: ["", "", "sinal"]})
    cfg = sp.from_dict(
        {
            "output": {
                "type": "label",
                "normalize": {"lowercase": True, "allowed": ["fatura", "sinal"]},
            },
            "columns": {"score": "label", "replica": "replica"},
            "gates": {"invalid_rate": {"max": 1.0}},
        }
    )
    report = sp.compare(ref, ref.copy(), cfg)
    assert report.summary["confusion"]["counts"][0][0] == 1  # id 2: two empty -> invalid
    assert report.summary["invalid_share"]["reference"] == 0.5


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: pd.concat([f, f.head(1)]), "duplicated"),
        (lambda f: f.assign(replica=[None, *f["replica"][1:]]), "replica column"),
        (lambda f: f.drop(columns="replica"), "missing columns"),
    ],
)
def test_replica_inputs_that_cannot_be_compared(mutate: Any, message: str) -> None:
    ref = replicated({1: ["a", "a"], 2: ["b", "b"]})
    cfg = sp.from_dict({**BASE, "columns": {"score": "label", "replica": "replica"}})
    with pytest.raises(sp.InputError, match=message):
        sp.compare(mutate(ref), ref.copy(), cfg)


def test_stability_gate() -> None:
    rng = np.random.default_rng(0)
    stable = {i: ["a"] * 3 for i in range(600)}
    shaky = {i: ["a", "a", "b"] if rng.random() < 0.2 else ["a"] * 3 for i in range(600)}
    cfg = sp.from_dict(
        {
            **BASE,
            "columns": {"score": "label", "replica": "replica"},
            "gates": {
                "stability": {"margin": 0.05},
                "label_agreement": {"min": 0.99},
            },
        }
    )
    assert sp.compare(replicated(stable), replicated(stable), cfg).passed
    report = sp.compare(replicated(stable), replicated(shaky), cfg)
    g = gate(report, "stability")
    assert not g.passed
    assert g.details["candidate_unstable"] > 0.1
    doc = json.loads(report.to_json())
    assert not list(Draft202012Validator(SCHEMA).iter_errors(doc))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"columns": {"replica": "r"}}, "columns.replica applies"),
        ({**BASE, "gates": {"stability": {"margin": 0.1}}}, "requires columns.replica"),
        ({**BASE, "columns": {"replica": "id"}}, "must differ"),
        ({**BASE, "gates": {"transitions": {"max_rate": 0.1, "per_class": {"a": 2}}}}, "per_class"),
    ],
)
def test_invalid_replica_configs(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(sp.ConfigError, match=message):
        sp.from_dict(raw)


# --------------------------------------------------------------------------- per-class limits


def test_per_class_limits_apply_to_their_source_class_only() -> None:
    rng = np.random.default_rng(101)  # not the stream that drew the labels
    labels = truth(20_000, 1)
    ref = pd.DataFrame({"id": np.arange(len(labels)), "label": labels})
    cand = ref.copy()
    small = (labels == "cancelamento") & (rng.random(len(labels)) < 0.05)
    cand.loc[small, "label"] = "fatura"
    tight = {"transitions": {"max_rate": 0.01}}
    loose_small = {"transitions": {"max_rate": 0.01, "per_class": {"cancelamento": 0.1}}}
    assert not sp.compare(ref, cand, sp.from_dict({**BASE, "gates": tight})).passed
    assert sp.compare(ref, cand, sp.from_dict({**BASE, "gates": loose_small})).passed
    big = (labels == "fatura") & (rng.random(len(labels)) < 0.05)
    cand.loc[big, "label"] = "sinal"
    report = sp.compare(ref, cand, sp.from_dict({**BASE, "gates": loose_small}))
    assert gate(report, "transitions").details["failing_pairs"] == ["fatura -> sinal"]


# --------------------------------------------------------------------------- label noise


def test_label_noise_calibration_by_simulation() -> None:
    """Harmless reruns pass the suggested gates; a real change of behaviour fails them."""
    base = sp.from_dict(BASE)
    fresh_pass, shift_pass, trials = 0, 0, 40
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        labels = truth(3000, seed)
        reference = run(labels, 0.01, rng)
        replicates = {f"r{i}": run(labels, 0.01, rng) for i in range(3)}
        profile = measure_noise(reference, replicates, base)
        cfg = sp.from_dict({**BASE, "gates": profile.suggested_gates})
        fresh_pass += sp.compare(reference, run(labels, 0.01, rng), cfg).passed
        shift_pass += sp.compare(reference, run(labels, 0.01, rng, shift=0.05), cfg).passed
    assert fresh_pass >= trials - 2, fresh_pass
    assert shift_pass == 0, shift_pass


def test_label_noise_suggests_per_class_transitions_and_invalid_rate() -> None:
    rng = np.random.default_rng(3)
    labels = truth(5000, 3)
    reference = run(labels, 0.02, rng)
    reps = {f"r{i}": run(labels, 0.02, rng) for i in range(3)}
    for rep in reps.values():
        rep.loc[rng.random(len(rep)) < 0.002, "label"] = "sorry"
    cfg = sp.from_dict(
        {
            "output": {"type": "label", "normalize": {"allowed": CLASSES}},
            "columns": {"score": "label"},
        }
    )
    profile = measure_noise(reference, reps, cfg)
    gates = profile.suggested_gates
    assert set(gates) == {
        "label_agreement",
        "transitions",
        "class_prevalence",
        "kappa",
        "invalid_rate",
    }
    per_class = gates["transitions"]["per_class"]
    assert per_class["fatura"] < gates["transitions"]["max_rate"]
    assert profile.safety == 1.75
    text = suggested_config_yaml(profile, cfg)
    loaded = sp.from_dict(yaml.safe_load(text))
    assert loaded.output.normalize is not None
    assert loaded.output.normalize.allowed == tuple(CLASSES)
    assert loaded.gates.transitions is not None
    assert loaded.gates.transitions.per_class == per_class
    assert sp.compare(reference, run(labels, 0.02, rng), loaded).passed


def test_generated_label_config_escapes_class_names() -> None:
    evil = 'x"\ngates: {coverage: null}\n#'
    labels = np.where(np.arange(400) % 2, evil, "ok")
    ref = pd.DataFrame({"id": range(400), "label": labels})
    rep = ref.copy()
    rep.loc[:3, "label"] = "ok"
    cfg = sp.from_dict({"output": {"type": "label"}, "columns": {"score": "label"}})
    text = suggested_config_yaml(measure_noise(ref, {"r1": rep}, cfg), cfg)
    loaded = sp.from_dict(yaml.safe_load(text))
    assert loaded.gates.coverage is not None  # the injected line did not take effect
    assert loaded.gates.transitions is not None


def test_label_noise_input_errors() -> None:
    ref = run(truth(500, 0), 0.0, np.random.default_rng(0))
    cfg = sp.from_dict(BASE)
    with pytest.raises(sp.InputError, match="same rows"):
        measure_noise(ref, {"r": ref.iloc[:-1]}, cfg)
    with pytest.raises(sp.InputError, match="'score' and 'label'"):
        measure_noise(ref, {"r": ref}, sp.from_dict({"output": {"type": "probabilities"}}))


def test_cli_label_noise_and_replicas(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    labels = truth(3000, 5)
    paths = {}
    for name, frame in {
        "ref": run(labels, 0.01, rng),
        "r1": run(labels, 0.01, rng),
        "r2": run(labels, 0.01, rng),
        "r3": run(labels, 0.01, rng),
        "fresh": run(labels, 0.01, rng),
        "changed": run(labels, 0.01, rng, shift=0.1),
    }.items():
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    out = tmp_path / "noise.yaml"
    code = main(
        [
            "noise",
            "--reference",
            str(paths["ref"]),
            *[a for r in ("r1", "r2", "r3") for a in ("--replicate", str(paths[r]))],
            "--output-type",
            "label",
            "--score",
            "label",
            "--out",
            str(out),
        ]
    )
    assert code == EXIT_PASS
    cmp = ["compare", "--reference", str(paths["ref"]), "--config", str(out), "--quiet"]
    assert main([*cmp, "--candidate", str(paths["fresh"])]) == EXIT_PASS
    assert main([*cmp, "--candidate", str(paths["changed"])]) == EXIT_FAIL

    reps = pd.concat(
        [run(labels, 0.01, rng).assign(replica=i) for i in range(3)], ignore_index=True
    )
    rep_path = tmp_path / "replicas.csv"
    reps.to_csv(rep_path, index=False)
    replica_config = tmp_path / "replica_parity.yaml"
    replica_config.write_text(
        "version: 1\noutput: {type: label}\ncolumns: {score: label, replica: replica}\n"
        "gates: {label_agreement: {min: 0.99}}\n",
        encoding="utf-8",
    )
    args = [
        "compare",
        "--reference",
        str(rep_path),
        "--candidate",
        str(rep_path),
        "--output-type",
        "label",
        "--score",
        "label",
        "--replica",
        "replica",
        "--config",
        str(replica_config),
        "--quiet",
    ]
    assert main(args) == EXIT_PASS
    assert main([*args[:-1], "--replica", "missing", "--quiet"]) == EXIT_ERROR
