"""The comparison report and its renderings.

The JSON document is a public contract, versioned by `schema_version`. Fields may be added in a
minor release; renaming or removing a field requires a new schema version.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass, field
from typing import Any

from scoreparity import __version__
from scoreparity.gates import GateResult

SCHEMA_VERSION = 1


def _clean(value: Any) -> Any:
    """Make a value strict-JSON safe: NaN/inf become null, tuples become lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return _clean(value.item())
    return value


@dataclass(frozen=True)
class Report:
    gates: list[GateResult]
    summary: dict[str, Any]
    config: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def failed_gates(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = _clean(
            {
                "schema_version": SCHEMA_VERSION,
                "verdict": self.verdict,
                "failed_gates": self.failed_gates,
                "gates": [g.to_dict() for g in self.gates],
                "summary": self.summary,
                "config": self.config,
                "inputs": self.inputs,
                "environment": {
                    "scoreparity": __version__,
                    "python": platform.python_version(),
                },
            }
        )
        return doc

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False, ensure_ascii=False)

    def to_markdown(self) -> str:
        return render_markdown(self.to_dict())

    def to_html(self) -> str:
        from scoreparity.report_html import render_html

        return render_html(self.to_dict())

    def to_junit(self) -> str:
        from scoreparity.report_junit import render_junit

        return render_junit(self.to_dict())


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e6:
            return f"{value:.3g}"
        return f"{value:.6g}"
    return str(value)


def render_markdown(doc: dict[str, Any]) -> str:
    """Markdown suited to a pull-request comment or a CI job summary."""
    icon = "✅" if doc["verdict"] == "PASS" else "❌"
    s = doc["summary"]
    lines = [
        f"## {icon} Score parity: **{doc['verdict']}**",
        "",
        f"{s['n_matched_finite']:,} rows compared "
        f"(reference {s['n_reference']:,}, candidate {s['n_candidate']:,}, "
        f"missing {s['n_missing']:,}, extra {s['n_extra']:,}).",
        "",
        "| Gate | Result | Value | Threshold | What it checks |",
        "|---|---|---|---|---|",
    ]
    for g in doc["gates"]:
        ci = g["details"].get("ci")
        value = _fmt(g["value"]) + (f" (CI {_fmt(ci[0])} to {_fmt(ci[1])})" if ci else "")
        lines.append(
            f"| `{g['name']}` | {'pass' if g['passed'] else '**FAIL**'} | {value} | "
            f"{_fmt(g['threshold'])} | {g['description']} |"
        )
    quantiles = s.get("abs_diff_quantiles")
    if quantiles:
        lines += [
            "",
            "Absolute difference quantiles: "
            + ", ".join(f"q{q} = {_fmt(v)}" for q, v in quantiles.items())
            + f". Identical scores: {100 * s['identical_share']:.2f}%.",
        ]
    auc = s.get("auc")
    if auc:
        lo, hi = auc["delta_ci90"]
        lines += [
            "",
            f"AUC: reference {auc['reference']:.5f}, candidate {auc['candidate']:.5f} "
            f"(difference {_fmt(auc['delta'])}, 90% CI {_fmt(lo)} to {_fmt(hi)}).",
        ]
    for g in doc["gates"]:
        failing = g["details"].get("failing_segments")
        if failing:
            lines += [
                "",
                f"Segments failing `{g['name']}`: " + ", ".join(f"`{x}`" for x in failing),
            ]
    lines += ["", f"<sub>scoreparity {doc['environment']['scoreparity']}</sub>"]
    return "\n".join(lines) + "\n"
