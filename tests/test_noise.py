"""`scoreparity noise`: tolerances calibrated from replicate runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from scoreparity import InputError, compare, from_dict
from scoreparity.cli import EXIT_ERROR, EXIT_PASS, main
from scoreparity.noise import MIN_MARGIN, _ceil_sig, measure_noise, suggested_config_yaml
from tests.conftest import make_scores

BASE = from_dict({"segments": ["plan"], "columns": {"label": "label"}})


def _rerun(reference: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    """A harmless rerun: same rows, float noise of scale sigma, shuffled order."""
    rng = np.random.default_rng(seed)
    out = reference[["id", "score"]].copy()
    out["score"] = out["score"] + rng.normal(0, sigma, len(out))
    shuffled: pd.DataFrame = out.sample(frac=1.0, random_state=seed)
    return shuffled


@given(st.floats(min_value=1e-300, max_value=1e300, allow_nan=False))
def test_ceil_sig_rounds_up_to_two_significant_digits(value: float) -> None:
    rounded = _ceil_sig(value)
    assert rounded >= value
    assert rounded <= value * 1.11
    assert rounded == float(f"{rounded:.1e}")  # representable with two significant digits


def test_ceil_sig_is_decimal_exact() -> None:
    assert _ceil_sig(3e-7) == 3e-7
    assert repr(_ceil_sig(2.01e-7)) == "2.1e-07"
    assert _ceil_sig(0.0) == 0.0


def test_deterministic_pipeline_gets_exact_gates(reference: pd.DataFrame) -> None:
    profile = measure_noise(reference, {"r1": reference[["id", "score"]].copy()}, BASE)
    gates = profile.suggested_gates
    assert gates["max_abs_diff"]["max"] == 0
    assert gates["mean_diff_equivalence"]["margin"] == MIN_MARGIN
    cfg = from_dict(yaml.safe_load(suggested_config_yaml(profile, BASE)))
    assert compare(reference, reference.copy(), cfg).passed
    assert not compare(reference, reference.assign(score=reference["score"] + 1e-9), cfg).passed


def test_calibrated_tolerances_pass_new_reruns_and_fail_real_changes(
    reference: pd.DataFrame,
) -> None:
    sigma = 1e-7
    replicates = {f"r{i}": _rerun(reference, sigma, seed=i) for i in range(3)}
    profile = measure_noise(reference, replicates, BASE, safety=3.0)
    cfg = from_dict(yaml.safe_load(suggested_config_yaml(profile, BASE)))

    fresh = [compare(reference, _rerun(reference, sigma, seed=100 + i), cfg) for i in range(30)]
    pass_rate = sum(r.passed for r in fresh) / len(fresh)
    assert pass_rate >= 0.95, [r.failed_gates for r in fresh if not r.passed][:3]

    worst = profile.worst("max_abs_diff")
    shifted = reference.assign(score=reference["score"] + 10 * worst)
    assert not compare(reference, shifted, cfg).passed


def test_segment_noise_is_covered(reference: pd.DataFrame) -> None:
    """Small segments have wider confidence intervals; the margin must cover them too."""
    ref = reference.assign(plan=np.where(np.arange(len(reference)) < 60, "tiny", reference["plan"]))
    replicates = {f"r{i}": _rerun(ref, 1e-7, seed=i) for i in range(3)}
    profile = measure_noise(ref, replicates, BASE)
    cfg = from_dict(yaml.safe_load(suggested_config_yaml(profile, BASE)))
    results = [compare(ref, _rerun(ref, 1e-7, seed=200 + i), cfg) for i in range(20)]
    assert sum(r.passed for r in results) >= 19


def test_replicate_must_score_the_same_rows(reference: pd.DataFrame) -> None:
    with pytest.raises(InputError, match="same rows"):
        measure_noise(reference, {"short": reference.iloc[10:][["id", "score"]]}, BASE)


def test_safety_below_one_is_rejected(reference: pd.DataFrame) -> None:
    with pytest.raises(InputError, match="safety"):
        measure_noise(reference, {"r": reference.copy()}, BASE, safety=0.5)


def test_untrusted_names_cannot_inject_yaml(reference: pd.DataFrame) -> None:
    evil_name = 'x"\ngates: {max_abs_diff: {max: 999}}\n#'
    base = from_dict({"columns": {"id": "id"}, "segments": ["plan"]})
    profile = measure_noise(reference, {evil_name: _rerun(reference, 1e-7, 1)}, base)
    doc = yaml.safe_load(suggested_config_yaml(profile, base))
    assert doc["gates"]["max_abs_diff"]["max"] < 1
    assert doc["segments"] == ["plan"]


def test_cli_noise_then_compare_round_trip(tmp_path: Path) -> None:
    ref = make_scores(3000, seed=4)
    ref_path = tmp_path / "ref.csv"
    ref.to_csv(ref_path, index=False)
    reps = []
    for i in range(3):
        p = tmp_path / f"rerun{i}.csv"
        _rerun(ref, 1e-8, seed=i).to_csv(p, index=False)
        reps += ["--replicate", str(p)]
    cfg_path, profile_path = tmp_path / "parity.yaml", tmp_path / "noise.json"
    args = [
        "noise",
        "--reference",
        str(ref_path),
        *reps,
        "--segment",
        "plan",
        "--label",
        "label",
        "--out",
        str(cfg_path),
        "--json",
        str(profile_path),
    ]
    assert main(args) == EXIT_PASS
    assert main(args) == EXIT_ERROR  # refuses to overwrite without --force
    assert len(json.loads(profile_path.read_text(encoding="utf-8"))["replicates"]) == 3
    assert "auc_difference" in yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["gates"]

    fresh = tmp_path / "fresh.csv"
    _rerun(ref, 1e-8, seed=99).to_csv(fresh, index=False)
    code = main(
        [
            "compare",
            "--reference",
            str(ref_path),
            "--candidate",
            str(fresh),
            "--config",
            str(cfg_path),
            "--quiet",
        ]
    )
    assert code == EXIT_PASS


def test_suggestions_scale_with_the_noise(reference: pd.DataFrame) -> None:
    small = measure_noise(reference, {"r": _rerun(reference, 1e-8, 1)}, BASE)
    large = measure_noise(reference, {"r": _rerun(reference, 1e-5, 1)}, BASE)
    for gate, key in (("max_abs_diff", "max"), ("mean_diff_equivalence", "margin")):
        ratio = large.suggested_gates[gate][key] / small.suggested_gates[gate][key]
        assert math.isclose(math.log10(ratio), 3, abs_tol=0.5)
