"""HTML and JUnit renderings, and the `render` command."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scoreparity import compare, from_dict
from scoreparity.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main
from scoreparity.report_html import render_html
from scoreparity.report_junit import render_junit

CFG = from_dict({"preset": "float-noise", "segments": ["plan"], "columns": {"label": "label"}})


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.scripts = 0
        self.handlers: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.scripts += tag == "script"
        self.handlers += [name for name, _ in attrs if name.startswith("on")]


def _doc(reference: pd.DataFrame, shift: float = 0.0) -> dict[str, Any]:
    cand = reference.assign(score=reference["score"] + shift)
    return compare(reference, cand, CFG).to_dict()


def test_html_is_self_contained_and_has_the_sections(reference: pd.DataFrame) -> None:
    html = render_html(_doc(reference, shift=1e-3))
    assert "http://" not in html
    assert "https://" not in html
    assert "✕ FAIL" in html
    assert "How large are the differences?" in html
    assert "Does any segment move?" in html
    parser = _Collector()
    parser.feed(html)
    assert parser.scripts == 0
    assert parser.tags.count("svg") == 2


def test_untrusted_segment_names_are_escaped(reference: pd.DataFrame) -> None:
    """A segment value from the data must never become markup in the report."""
    evil = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    ref = reference.assign(plan=np.where(reference.index % 2 == 0, evil, "ok"))
    doc = compare(ref, ref.assign(score=ref["score"] + 1e-3), CFG).to_dict()
    html = render_html(doc)
    parser = _Collector()
    parser.feed(html)
    assert parser.scripts == 0
    assert parser.handlers == []
    assert "&lt;script&gt;" in html


def test_histogram_buckets_cover_every_row(reference: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    cand = reference.assign(score=reference["score"] + rng.normal(0, 1e-6, len(reference)))
    doc = compare(reference, cand, CFG).to_dict()
    hist = doc["summary"]["abs_diff_histogram"]
    total = hist["zero"] + sum(d["count"] for d in hist["decades"])
    assert total == doc["summary"]["n_matched_finite"]


def test_junit_has_one_case_per_gate_and_marks_failures(reference: pd.DataFrame) -> None:
    doc = _doc(reference, shift=1e-3)
    root = ET.fromstring(render_junit(doc))
    cases = root.findall("testcase")
    assert len(cases) == len(doc["gates"])
    failed = {c.get("name") for c in cases if c.find("failure") is not None}
    assert failed == set(doc["failed_gates"])
    assert root.get("failures") == str(len(failed))


def test_junit_passing_report_has_no_failures(reference: pd.DataFrame) -> None:
    root = ET.fromstring(render_junit(_doc(reference)))
    assert root.get("failures") == "0"
    assert all(c.find("failure") is None for c in root.findall("testcase"))


def test_cli_writes_every_format_into_new_directories(
    reference: pd.DataFrame, tmp_path: Path
) -> None:
    ref = tmp_path / "ref.parquet"
    reference.to_parquet(ref)
    out = tmp_path / "nested" / "out"
    code = main(
        [
            "compare",
            "--reference",
            str(ref),
            "--candidate",
            str(ref),
            "--preset",
            "exact",
            "--json",
            str(out / "r.json"),
            "--markdown",
            str(out / "r.md"),
            "--html",
            str(out / "r.html"),
            "--junit",
            str(out / "r.xml"),
            "--quiet",
        ]
    )
    assert code == EXIT_PASS
    assert {p.name for p in out.iterdir()} == {"r.json", "r.md", "r.html", "r.xml"}


def test_render_command_mirrors_the_verdict(reference: pd.DataFrame, tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_doc(reference, shift=1e-3)), encoding="utf-8")
    html = tmp_path / "again.html"
    assert main(["render", str(report), "--html", str(html), "--quiet"]) == EXIT_FAIL
    assert "✕ FAIL" in html.read_text(encoding="utf-8")


@pytest.mark.parametrize("content", ["not json", '{"schema_version": 99}', "[1, 2]"])
def test_render_rejects_unreadable_reports(tmp_path: Path, content: str) -> None:
    report = tmp_path / "r.json"
    report.write_text(content, encoding="utf-8")
    assert main(["render", str(report), "--quiet"]) == EXIT_ERROR
