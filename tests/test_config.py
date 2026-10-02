from __future__ import annotations

from pathlib import Path

import pytest

from scoreparity.cli import CONFIG_TEMPLATE
from scoreparity.config import from_dict, load
from scoreparity.errors import ConfigError
from scoreparity.presets import PRESETS


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_loads(name: str) -> None:
    cfg = from_dict({"preset": name})
    assert cfg.preset == name
    assert cfg.gates.coverage is not None  # always-on defaults survive presets


def test_explicit_gate_overrides_preset_and_null_disables() -> None:
    cfg = from_dict(
        {"preset": "float-noise", "gates": {"max_abs_diff": {"max": 0.5}, "top_k_overlap": None}}
    )
    assert cfg.gates.max_abs_diff is not None
    assert cfg.gates.max_abs_diff.max == 0.5
    assert cfg.gates.top_k_overlap is None
    assert cfg.gates.quantile_abs_diff is not None  # untouched preset gate stays


def test_yaml_1_1_exponent_strings_are_numbers(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "gates:\n  max_abs_diff: {max: 1e-5}\n  decision_flips: {thresholds: [5e-1], max_rate: 0}\n"
    )
    cfg = load(path)
    assert cfg.gates.max_abs_diff is not None
    assert cfg.gates.max_abs_diff.max == 1e-5
    assert cfg.gates.decision_flips is not None
    assert cfg.gates.decision_flips.thresholds == (0.5,)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"gates": {"max_abs_dif": {"max": 1}}}, "unknown gates"),
        ({"gates": {"max_abs_diff": {"maximum": 1}}}, "unknown keys"),
        ({"segmnets": ["a"]}, "unknown top-level keys"),
        ({"gates": {"max_abs_diff": {"max": "abc"}}}, "expected a number"),
        ({"gates": {"max_abs_diff": {"max": -1}}}, "must be >= 0"),
        ({"gates": {"quantile_abs_diff": {"max": 1, "q": 1.5}}}, r"q must be in \(0, 1\)"),
        ({"gates": {"auc_difference": {"margin": 0.01}}}, "requires columns.label"),
        ({"alpha": 0.7}, "alpha must be in"),
        ({"preset": "nope"}, "unknown preset"),
        ({"version": 2}, "unsupported config version"),
        ({"segments": "plan"}, "list of column names"),
        (
            {"gates": {"mean_diff_equivalence": {"margin": 1, "per_segment": "yes"}}},
            "true or false",
        ),
    ],
)
def test_invalid_configs_fail_loudly(raw: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        from_dict(raw)


def test_init_template_is_a_valid_config(tmp_path: Path) -> None:
    for preset in PRESETS:
        path = tmp_path / f"{preset}.yaml"
        path.write_text(CONFIG_TEMPLATE.format(preset=preset))
        assert load(path).preset == preset


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("gates: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load(path)


def test_output_type_defaults_to_score_and_is_validated() -> None:
    assert from_dict({}).output.type == "score"
    assert from_dict({"output": {"type": "score"}}).output.type == "score"
    with pytest.raises(ConfigError, match=r"output\.type must be one of"):
        from_dict({"output": {"type": "unknown"}})
    with pytest.raises(ConfigError, match="unknown keys"):
        from_dict({"output": {"kind": "score"}})
