from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scoreparity.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main
from tests.conftest import make_scores


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    ref = make_scores(500)
    paths = {
        "ref": tmp_path / "ref.parquet",
        "same": tmp_path / "same.csv",
        "worse": tmp_path / "worse.csv",
    }
    ref.to_parquet(paths["ref"])
    ref[["id", "score"]].to_csv(paths["same"], index=False)
    ref.assign(score=1 - ref["score"])[["id", "score"]].to_csv(paths["worse"], index=False)
    return paths


def test_exit_code_pass(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "compare",
            "--reference",
            str(files["ref"]),
            "--candidate",
            str(files["same"]),
            "--preset",
            "float-noise",
            "--segment",
            "plan",
        ]
    )
    assert code == EXIT_PASS
    assert "PASS" in capsys.readouterr().out


def test_exit_code_fail_and_reports_written(files: dict[str, Path], tmp_path: Path) -> None:
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    code = main(
        [
            "compare",
            "--reference",
            str(files["ref"]),
            "--candidate",
            str(files["worse"]),
            "--preset",
            "float-noise",
            "--label",
            "label",
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
            "--quiet",
        ]
    )
    assert code == EXIT_FAIL
    doc = json.loads(out_json.read_text())
    assert doc["verdict"] == "FAIL"
    assert doc["inputs"]["reference"]["sha256"]
    assert "FAIL" in out_md.read_text()
    # --label alone (no AUC gate in the preset) still reports AUC.
    assert doc["summary"]["auc"]["candidate"] < doc["summary"]["auc"]["reference"]
    assert "AUC: reference" in out_md.read_text()


@pytest.mark.parametrize(
    "args",
    [
        ["compare", "--reference", "missing.csv", "--candidate", "missing.csv"],
        ["compare", "--reference", "x.txt", "--candidate", "x.txt"],
        ["compare"],  # missing required arguments
        ["nonsense"],
    ],
)
def test_exit_code_error(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(args) == EXIT_ERROR


def test_bad_column_is_an_error_not_a_failure(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "compare",
            "--reference",
            str(files["ref"]),
            "--candidate",
            str(files["same"]),
            "--score",
            "nope",
        ]
    )
    assert code == EXIT_ERROR
    assert "missing columns" in capsys.readouterr().err


def test_config_file_and_cli_overrides(files: dict[str, Path], tmp_path: Path) -> None:
    cfg = tmp_path / "parity.yaml"
    assert main(["init", str(cfg), "--preset", "exact"]) == EXIT_PASS
    assert main(["init", str(cfg)]) == EXIT_ERROR  # refuses to overwrite
    code = main(
        [
            "compare",
            "--reference",
            str(files["ref"]),
            "--candidate",
            str(files["same"]),
            "--config",
            str(cfg),
            "--quiet",
        ]
    )
    assert code == EXIT_PASS


def test_csv_reading_is_bit_exact(tmp_path: Path) -> None:
    """Regression: pandas' default CSV float parser is off by 1 ulp for ~25% of values."""
    from scoreparity.compare import read_table

    ref = make_scores(2000)
    path = tmp_path / "s.csv"
    ref[["id", "score"]].to_csv(path, index=False)
    back = read_table(path)
    assert (back["score"].to_numpy() == ref["score"].to_numpy()).all()


def test_separate_score_columns(tmp_path: Path) -> None:
    ref = make_scores(200)
    ref_path, cand_path = tmp_path / "a.csv", tmp_path / "b.csv"
    ref[["id", "score"]].rename(columns={"score": "p_old"}).to_csv(ref_path, index=False)
    ref[["id", "score"]].rename(columns={"score": "p_new"}).to_csv(cand_path, index=False)
    code = main(
        [
            "compare",
            "--reference",
            str(ref_path),
            "--candidate",
            str(cand_path),
            "--reference-score",
            "p_old",
            "--candidate-score",
            "p_new",
            "--preset",
            "exact",
            "--quiet",
        ]
    )
    assert code == EXIT_PASS
    pd.testing.assert_frame_equal(pd.read_csv(ref_path), pd.read_csv(ref_path))
