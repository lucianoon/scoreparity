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
MARKDOWN_EXAMPLES = 20  # a pull-request comment stays short; the JSON and HTML list them all


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
        return [g.label for g in self.gates if not g.passed]

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


def _code(value: Any) -> str:
    """Inline code for a user-provided name (segment, class) in a PR comment.

    Backticks and line breaks would let a name close the code span and inject Markdown, so
    they are replaced before wrapping.
    """
    text = str(value).replace("`", "'").replace("\r", " ").replace("\n", " ")
    return f"`{text[:120]}`"


def _cell(value: Any) -> str:
    """A user-provided value inside a Markdown table cell: code, pipes escaped, never empty."""
    text = _fmt(value)
    return _code(text).replace("|", r"\|") if text != "" else "(empty)"


def gate_name(g: dict[str, Any]) -> str:
    """Display name of a gate in a report document: `field.gate` for structured fields."""
    return f"{g['field']}.{g['name']}" if g.get("field") else str(g["name"])


def _summary_lines(s: dict[str, Any]) -> list[str]:
    lines: list[str] = []
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
    if "jaccard_mean" in s:
        lines += [
            "",
            f"Same set of labels: {100 * s['agreement']:.3f}% of rows over "
            f"{len(s.get('classes', []))} classes; mean Jaccard similarity "
            f"{_fmt(s['jaccard_mean'])}.",
        ]
    elif "agreement" in s:
        lines += [
            "",
            f"Class agreement: {100 * s['agreement']:.3f}% over {len(s.get('classes', []))} "
            f"classes; Cohen's kappa {_fmt(s.get('kappa'))}.",
        ]
    stab = s.get("stability")
    if stab:
        on_unstable = stab.get("changes_on_unstable_ids")
        lines += [
            "",
            "Ids whose replicas disagree: reference "
            f"{100 * stab['reference']['unstable_share']:.2f}%, candidate "
            f"{100 * stab['candidate']['unstable_share']:.2f}%"
            + (
                f"; {100 * on_unstable:.1f}% of the class changes are on such ids."
                if on_unstable is not None
                else "."
            ),
        ]
    invalid = s.get("invalid_share")
    if invalid:
        lines += [
            "",
            f"Answers outside the allowed classes: reference {100 * invalid['reference']:.3f}%, "
            f"candidate {100 * invalid['candidate']:.3f}%.",
        ]
    valid = s.get("schema_valid")
    if valid:
        lines += [
            "",
            f"Valid documents: reference {100 * valid['reference']:.3f}%, "
            f"candidate {100 * valid['candidate']:.3f}%.",
        ]
    return lines


def _hint_lines(gates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    events = {"label_agreement": "disagreement", "invalid_rate": "invalid answer"}
    count_key = {"label_agreement": "disagreements", "invalid_rate": "candidate_invalid"}
    for g in gates:
        details, name = g["details"], _code(gate_name(g))
        for key, what in (
            ("failing_segments", "Segments failing"),
            ("failing_classes", "Classes failing"),
            ("failing_pairs", "Class transitions failing"),
        ):
            items = details.get(key)
            if items:
                shown = ", ".join(_code(x) for x in items[:20])
                more = f" (+{len(items) - 20} more)" if len(items) > 20 else ""
                lines += ["", f"{what} {name}: {shown}{more}"]
        if not g["passed"] and details.get("underpowered_classes"):
            lines += [
                "",
                f"{name}: not enough rows to verify the threshold for "
                + ", ".join(_code(x) for x in details["underpowered_classes"][:20])
                + f" (each needs at least {details['rows_needed_per_class']} rows; collect more "
                "rows or raise `min_class_size` to exclude them deliberately).",
            ]
        if (
            g["name"] in events
            and not g["passed"]
            and details.get(count_key[g["name"]]) == 0
            and "rows_needed" in details
        ):
            lines += [
                "",
                f"{name}: no {events[g['name']]} observed, but {details['rows']} rows "
                f"cannot prove the threshold; at least {details['rows_needed']} rows are needed "
                "(`scoreparity plan-sample` sizes a sample before scoring it).",
            ]
    return lines


def _example_lines(s: dict[str, Any]) -> list[str]:
    examples = s.get("examples")
    if not examples:
        return []
    shown = examples[:MARKDOWN_EXAMPLES]
    extra = "difference" if "difference" in shown[0] else None
    lines = [
        "",
        f"Examples by id ({len(shown)} of {len(examples)} listed in the report):",
        "",
        "| id | reference | candidate |" + (" difference |" if extra else ""),
        "|---|---|---|" + ("---|" if extra else ""),
    ]
    for e in shown:
        lines.append(
            f"| {_cell(e['id'])} | {_cell(e['reference'])} | {_cell(e['candidate'])} |"
            + (f" {_fmt(e[extra])} |" if extra else "")
        )
    return lines


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
            f"| {_cell(gate_name(g))} | {'pass' if g['passed'] else '**FAIL**'} | {value} | "
            f"{_fmt(g['threshold'])} | {g['description']} |"
        )
    lines += _summary_lines(s)
    lines += _hint_lines(doc["gates"])
    lines += _example_lines(s)
    for name, sub in (s.get("fields") or {}).items():
        lines += ["", f"### Field {_code(name)} ({sub.get('type', '')})"]
        lines += _summary_lines(sub) or ["", "No rows to summarise."]
        lines += _example_lines(sub)
    lines += ["", f"<sub>scoreparity {doc['environment']['scoreparity']}</sub>"]
    return "\n".join(lines) + "\n"
