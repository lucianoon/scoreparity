"""The JSON report is a public contract: every report must validate against the published schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from scoreparity import compare, from_dict
from scoreparity.report import SCHEMA_VERSION

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schema" / f"report-v{SCHEMA_VERSION}.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_itself_is_valid() -> None:
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"preset": "exact"},
        {
            "preset": "float-noise",
            "segments": ["plan"],
            "columns": {"label": "label"},
            "gates": {"auc_difference": {"margin": 0.01}},
        },
        {"preset": "quantization", "segments": ["plan", "region"]},
    ],
)
@pytest.mark.parametrize("shift", [0.0, 1e-3])
def test_reports_validate_against_schema(
    reference: pd.DataFrame, config: dict[str, Any], shift: float
) -> None:
    cand = reference.assign(score=reference["score"] + shift)
    cand.loc[:3, "score"] = np.nan
    doc = json.loads(compare(reference, cand, from_dict(config)).to_json())
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(doc), key=str)
    assert not errors, [e.message for e in errors[:5]]


def test_schema_lists_every_gate_the_code_knows() -> None:
    from scoreparity.config import _GATE_TYPES

    names = SCHEMA["properties"]["gates"]["items"]["properties"]["name"]["enum"]
    assert set(names) == set(_GATE_TYPES) | {"parity_gate_configured"}
