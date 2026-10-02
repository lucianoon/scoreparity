"""JUnit XML rendering: one test case per gate, so any CI test UI shows parity gates natively."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from scoreparity.report import _fmt


def render_junit(doc: dict[str, Any], suite_name: str = "scoreparity") -> str:
    gates = doc["gates"]
    failures = sum(not g["passed"] for g in gates)
    suite = ET.Element(
        "testsuite",
        name=suite_name,
        tests=str(len(gates)),
        failures=str(failures),
        errors="0",
        skipped="0",
    )
    props = ET.SubElement(suite, "properties")
    for key, value in (
        ("verdict", doc["verdict"]),
        ("rows_compared", doc["summary"]["n_matched_finite"]),
        ("scoreparity_version", doc["environment"]["scoreparity"]),
        ("schema_version", doc["schema_version"]),
    ):
        ET.SubElement(props, "property", name=key, value=str(value))
    for g in gates:
        case = ET.SubElement(suite, "testcase", classname=suite_name, name=g["name"])
        summary = f"value {_fmt(g['value'])}, threshold {_fmt(g['threshold'])}: {g['description']}"
        if not g["passed"]:
            failure = ET.SubElement(case, "failure", message=summary, type="ParityGateFailed")
            failing = g["details"].get("failing_segments")
            failure.text = summary + (
                "\nfailing segments: " + ", ".join(map(str, failing)) if failing else ""
            )
        else:
            ET.SubElement(case, "system-out").text = summary
    ET.indent(suite)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(suite, encoding="unicode") + "\n"
    )
