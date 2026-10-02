"""Self-contained HTML report: no external scripts, styles or fonts, so it works as a CI artifact
opened offline. Charts are inline SVG built from the JSON report alone.

Every string that can come from user data (segment names, file paths, column names) goes through
`html.escape`; the report is safe to open even when the data is not.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

from scoreparity.report import _fmt

_CSS = """
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --series: #2a78d6; --band: rgba(42,120,214,0.10);
  --good: #0ca30c; --good-text: #006300; --critical: #d03b3b; --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --series: #3987e5; --band: rgba(57,135,229,0.16);
    --good: #0ca30c; --good-text: #0ca30c; --critical: #d03b3b; --border: rgba(255,255,255,0.10);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; }
p.lead { color: var(--ink-2); margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; overflow-x: auto; }
.verdict { display: inline-flex; gap: 8px; align-items: center; font-weight: 700;
  font-size: 18px; padding: 6px 14px; border-radius: 999px; border: 2px solid currentColor; }
.verdict.pass { color: var(--good-text); } .verdict.fail { color: var(--critical); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 16px 0; }
.fact { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px; }
.fact b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; }
.fact span { color: var(--ink-2); font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; font-size: 13px; }
td.num { text-align: right; white-space: nowrap; }
.status { font-weight: 700; white-space: nowrap; }
.status.pass { color: var(--good-text); } .status.fail { color: var(--critical); }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--ink-2); }
svg { display: block; max-width: 100%; height: auto; }
svg text { fill: var(--ink-2); font: 12px system-ui, sans-serif; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .bar { fill: var(--series); } svg .bar.beyond { fill: var(--critical); }
svg .mark:hover, svg .mark:focus { outline: none; opacity: 0.75; }
svg .band { fill: var(--band); } svg .ci { stroke: var(--series); stroke-width: 2; }
svg .ci.fail { stroke: var(--critical); } svg .ci.ungated { stroke: var(--muted); }
svg .dot { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
svg .dot.fail { fill: var(--critical); } svg .dot.ungated { fill: var(--muted); }
svg .limit { stroke: var(--critical); stroke-width: 1.5; stroke-dasharray: 4 3; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--ink-2);
  margin-bottom: 8px; }
.key { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
  vertical-align: -1px; margin-right: 6px; }
footer { color: var(--muted); font-size: 12px; margin-top: 32px; word-break: break-all; }
code { font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
@media (forced-colors: active) {
  svg .bar, svg .dot { fill: CanvasText; } svg .ci { stroke: CanvasText; }
}
"""


def _status(passed: bool) -> str:
    return (
        '<span class="status pass">✓ pass</span>'
        if passed
        else '<span class="status fail">✕ FAIL</span>'
    )


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = float(10.0 ** math.floor(math.log10(value)))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * exp:
            return step * exp
    return 10 * exp


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """A bar with rounded top corners, anchored flat on the baseline."""
    r = max(0.0, min(r, w / 2, h))
    return (
        f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z"
    )


def _histogram(doc: dict[str, Any]) -> str:
    hist = doc["summary"].get("abs_diff_histogram")
    if not hist:
        return ""
    decades = {d["exponent"]: d["count"] for d in hist["decades"]}
    exps = list(range(min(decades), max(decades) + 1)) if decades else []
    buckets: list[tuple[str, int, int | None]] = [("0", hist["zero"], None)]
    buckets += [(f"1e{e}", decades.get(e, 0), e) for e in exps]
    limit = next((g["threshold"] for g in doc["gates"] if g["name"] == "max_abs_diff"), None)

    width, height = 720, 260
    left, right, top, bottom = 56, 16, 16, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(buckets)
    gap_zero = 16 if n > 1 else 0
    slot = (plot_w - gap_zero) / n
    bar_w = max(4.0, slot - 2)  # 2px surface gap between adjacent bars
    ymax = _nice_max(max(c for _, c, _ in buckets))
    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="hist-title"><title id="hist-title">Rows per order of magnitude '
        "of the absolute score difference</title>"
    ]
    for i in range(5):
        v = ymax * i / 4
        y = top + plot_h - plot_h * v / ymax
        svg.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        svg.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt(float(v))}</text>'
        )
    beyond_any = False
    for i, (label, count, exp) in enumerate(buckets):
        x = left + i * slot + (gap_zero if i > 0 else 0) + (slot - bar_w) / 2
        h = plot_h * count / ymax
        beyond = limit is not None and exp is not None and 10.0**exp >= limit and limit > 0
        beyond = beyond or (limit == 0 and exp is not None)
        beyond_any = beyond_any or (beyond and count > 0)
        if count > 0:
            rng = "exactly 0" if exp is None else f"[1e{exp}, 1e{exp + 1})"
            svg.append(
                f'<path class="bar mark{" beyond" if beyond else ""}" tabindex="0" '
                f'd="{_bar_path(x, top + plot_h - h, bar_w, h)}">'
                f"<title>|diff| {rng}: {count:,} rows</title></path>"
            )
        svg.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - bottom + 16}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )
    if limit is not None and limit > 0 and exps:
        pos = math.log10(limit)
        e0 = math.floor(pos)
        if e0 in exps:
            i = exps.index(e0) + 1
            x = left + gap_zero + i * slot + (pos - e0) * slot
            svg.append(
                f'<line class="limit" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}"/>'
            )
    svg.append(
        f'<line class="axis" x1="{left}" x2="{width - right}" '
        f'y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle">'
        "|candidate &minus; reference| (bucket = lower bound)</text></svg>"
    )
    legend = ['<span><i class="key" style="background:var(--series)"></i>rows</span>']
    if beyond_any:
        legend.append(
            '<span><i class="key" style="background:var(--critical)"></i>'
            "✕ beyond the <code>max_abs_diff</code> tolerance</span>"
        )
    rows = "".join(
        f"<tr><td>{'exactly 0' if e is None else f'[1e{e}, 1e{e + 1})'}</td>"
        f'<td class="num">{c:,}</td></tr>'
        for _, c, e in buckets
    )
    return (
        "<h2>How large are the differences?</h2>"
        '<p class="lead">Rows per order of magnitude of |candidate &minus; reference|. '
        "Floating-point noise sits far left; behaviour changes sit far right.</p>"
        f'<div class="card"><div class="legend">{"".join(legend)}</div>{"".join(svg)}'
        "<details><summary>Show data</summary><table><tr><th>|diff|</th>"
        f'<th class="num">rows</th></tr>{rows}</table></details></div>'
    )


def _segments(doc: dict[str, Any]) -> str:
    gate = next((g for g in doc["gates"] if g["name"] == "mean_diff_equivalence"), None)
    if not gate or not gate["details"].get("segments"):
        return ""
    margin = float(gate["threshold"])
    segs = sorted(
        gate["details"]["segments"],
        key=lambda s: -max(abs(s["ci"][0] or 0), abs(s["ci"][1] or 0)),
    )
    shown = segs[:40]
    extent = max([margin * 1.5] + [abs(v or 0) for s in shown for v in s["ci"]]) * 1.1
    width, row_h = 720, 24
    left, right, top = 200, 24, 28
    plot_w = width - left - right
    height = top + row_h * len(shown) + 24

    def sx(v: float) -> float:
        return float(left + plot_w * (v + extent) / (2 * extent))

    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="seg-title">'
        '<title id="seg-title">Mean score difference per segment with confidence interval'
        "</title>",
        f'<rect class="band" x="{sx(-margin):.1f}" y="{top - 8}" '
        f'width="{sx(margin) - sx(-margin):.1f}" height="{row_h * len(shown) + 8}"/>',
        f'<line class="axis" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top - 8}" '
        f'y2="{top + row_h * len(shown)}"/>',
    ]
    # Axis labels: 0 and the extremes always; the margin only if it does not collide.
    placed: list[float] = []
    for v in (0.0, -extent, extent, -margin, margin):
        x = sx(v)
        if any(abs(x - p) < 56 for p in placed):
            continue
        placed.append(x)
        svg.append(f'<text x="{x:.1f}" y="{top - 14}" text-anchor="middle">{_fmt(v)}</text>')
    for i, s in enumerate(shown):
        y = top + i * row_h + row_h / 2
        state = "" if s["equivalent"] else (" fail" if s["gated"] else " ungated")
        mark = "" if s["equivalent"] else (" ✕" if s["gated"] else " (too small)")
        lo, hi, mean = s["ci"][0], s["ci"][1], s["mean_diff"]
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">'
            f"{escape(str(s['segment']))}{mark}</text>"
        )
        svg.append(
            f'<g class="mark" tabindex="0"><title>{escape(str(s["segment"]))}: mean '
            f"{_fmt(mean)}, CI {_fmt(lo)} to {_fmt(hi)}, n = {s['n']:,}</title>"
            f'<line class="ci{state}" x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<circle class="dot{state}" cx="{sx(mean):.1f}" cy="{y:.1f}" r="5"/></g>'
        )
    svg.append("</svg>")
    rows = "".join(
        f"<tr><td>{escape(str(s['segment']))}</td><td class='num'>{s['n']:,}</td>"
        f"<td class='num'>{_fmt(s['mean_diff'])}</td>"
        f"<td class='num'>{_fmt(s['ci'][0])} to {_fmt(s['ci'][1])}</td>"
        f"<td>{_status(s['equivalent']) if s['gated'] else 'not gated (too small)'}</td></tr>"
        for s in segs
    )
    more = f" Showing the {len(shown)} widest of {len(segs)}." if len(segs) > len(shown) else ""
    level = 100 * (1 - 2 * float(doc["config"]["alpha"]))
    return (
        "<h2>Does any segment move?</h2>"
        f'<p class="lead">Mean difference per segment with its {level:g}% '
        f"confidence interval. The shaded band is the tolerance (±{_fmt(margin)}); a segment "
        f"passes only if its whole interval fits inside.{more}</p>"
        '<div class="card"><div class="legend">'
        '<span><i class="key" style="background:var(--series)"></i>equivalent</span>'
        '<span><i class="key" style="background:var(--critical)"></i>✕ not equivalent</span>'
        '<span><i class="key" style="background:var(--muted)"></i>too small to gate</span>'
        f'<span><i class="key" style="background:var(--band);outline:1px solid var(--series)">'
        f"</i>tolerance ±{_fmt(margin)}</span>"
        f"</div>{''.join(svg)}<details><summary>Show data</summary><table><tr>"
        "<th>Segment</th><th class='num'>rows</th><th class='num'>mean diff</th>"
        f"<th class='num'>CI</th><th>result</th></tr>{rows}</table></details></div>"
    )


MAX_HEATMAP_CLASSES = 25


def _compact(count: int) -> str:
    return (
        str(count)
        if count < 1000
        else f"{count / 1000:.1f}k"
        if count < 1_000_000
        else f"{count / 1e6:.1f}M"
    )


def _confusion(doc: dict[str, Any]) -> str:
    """Transition matrix (reference class -> candidate class) as a heatmap plus a data table.

    Off-diagonal cells are shaded by the share of the reference class that moved (one hue,
    light to dark); cells of transitions that failed the gate get a critical outline and a mark.
    """
    conf = doc["summary"].get("confusion")
    if not conf:
        return ""
    classes = [str(c) for c in conf["classes"]]
    counts = conf["counts"]
    k = len(classes)
    gate = next((g for g in doc["gates"] if g["name"] == "transitions"), None)
    failing = set(gate["details"].get("failing_pairs", [])) if gate else set()
    rows_total = [sum(r) for r in counts]
    max_rate = max(
        (
            counts[i][j] / rows_total[i]
            for i in range(k)
            for j in range(k)
            if i != j and rows_total[i]
        ),
        default=0.0,
    )
    table_rows = "".join(
        f"<tr><td>{escape(classes[i])}</td>"
        + "".join(f"<td class='num'>{counts[i][j]:,}</td>" for j in range(k))
        + f"<td class='num'>{rows_total[i]:,}</td></tr>"
        for i in range(k)
    )
    table = (
        "<details><summary>Show data</summary><table><tr><th>reference \\ candidate</th>"
        + "".join(f"<th class='num'>{escape(c)}</th>" for c in classes)
        + f"<th class='num'>rows</th></tr>{table_rows}</table></details>"
    )
    intro = (
        "<h2>Which classes changed?</h2>"
        '<p class="lead">Rows are the reference class, columns the candidate class. The diagonal '
        "is agreement; off the diagonal, darker means a larger share of that reference class "
        "moved.</p>"
    )
    if k > MAX_HEATMAP_CLASSES:
        return intro + f'<div class="card"><p>{k} classes: table only.</p>{table}</div>'

    cell, left, top = 44, 140, 110
    size_w, size_h = left + cell * k + 8, top + cell * k + 8
    svg = [
        f'<svg viewBox="0 0 {size_w} {size_h}" width="{size_w}" height="{size_h}" '
        'role="img" aria-labelledby="cm-title">'
        '<title id="cm-title">Transition matrix from reference class to candidate class</title>'
    ]
    for j, name in enumerate(classes):
        x = left + j * cell + cell / 2
        svg.append(
            f'<text x="{x:.1f}" y="{top - 8}" text-anchor="start" '
            f'transform="rotate(-45 {x:.1f} {top - 8})">{escape(name[:18])}</text>'
        )
    for i, name in enumerate(classes):
        y = top + i * cell + cell / 2 + 4
        svg.append(f'<text x="{left - 8}" y="{y:.1f}" text-anchor="end">{escape(name[:18])}</text>')
        for j in range(k):
            x, y0 = left + j * cell, top + i * cell
            count = counts[i][j]
            share = count / rows_total[i] if rows_total[i] else 0.0
            label = f"{classes[i]} -> {classes[j]}"
            bad = label in failing
            if i == j:
                fill, opacity = "var(--grid)", 1.0
            else:
                fill = "var(--series)"
                opacity = 0.0 if count == 0 else 0.15 + 0.85 * (share / max_rate if max_rate else 0)
            stroke = ' stroke="var(--critical)" stroke-width="2.5"' if bad else ""
            svg.append(
                f'<g class="mark" tabindex="0"><title>{escape(label)}: {count:,} rows '
                f"({100 * share:.2f}% of {escape(classes[i])})</title>"
                f'<rect x="{x + 1}" y="{y0 + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" '
                f'fill="{fill}" fill-opacity="{opacity:.3f}"{stroke}/>'
                + (
                    f'<text x="{x + cell / 2:.1f}" y="{y0 + cell / 2 + 4:.1f}" '
                    f'text-anchor="middle">{"✕" if bad else ""}{_compact(count)}</text>'
                    if count
                    else ""
                )
                + "</g>"
            )
    svg.append("</svg>")
    legend = (
        '<div class="legend"><span><i class="key" style="background:var(--grid)"></i>'
        'agreement</span><span><i class="key" style="background:var(--series)"></i>'
        "share of the reference class that moved</span>"
        + (
            '<span><i class="key" style="background:transparent;outline:2px solid '
            'var(--critical)"></i>✕ transition above the tolerance</span>'
            if failing
            else ""
        )
        + "</div>"
    )
    return intro + f'<div class="card">{legend}{"".join(svg)}{table}</div>'


def _examples(doc: dict[str, Any]) -> str:
    examples = doc["summary"].get("examples")
    if not examples:
        return ""
    extra = "difference" if "difference" in examples[0] else None
    head = "<tr><th>id</th><th>reference</th><th>candidate</th>" + (
        "<th class='num'>difference</th></tr>" if extra else "</tr>"
    )
    rows = "".join(
        f"<tr><td><code>{escape(str(e['id']))}</code></td>"
        f"<td>{escape(_fmt(e['reference']))}</td><td>{escape(_fmt(e['candidate']))}</td>"
        + (f"<td class='num'>{escape(_fmt(e[extra]))}</td>" if extra else "")
        + "</tr>"
        for e in examples
    )
    what = (
        "the largest differences first"
        if extra
        else "rows whose class changed, one kind of change at a time"
    )
    return (
        "<h2>Examples</h2>"
        f'<p class="lead">Rows by id only ({what}), to look up in the source system.</p>'
        f"<div class='card'><table>{head}{rows}</table></div>"
    )


def render_html(doc: dict[str, Any]) -> str:
    """Render a JSON report document (see `Report.to_dict`) as a standalone HTML page."""
    passed = doc["verdict"] == "PASS"
    s = doc["summary"]
    facts = [(f"{s['n_matched_finite']:,}", "rows compared")]
    if "identical_share" in s:
        facts.append((f"{100 * s['identical_share']:.2f}%", "identical outputs"))
        facts.append((_fmt((s.get("abs_diff_quantiles") or {}).get("1.0")), "largest |difference|"))
    if "agreement" in s:
        facts.append((f"{100 * s['agreement']:.3f}%", "same class"))
        facts.append((_fmt(s.get("kappa")), "Cohen's kappa"))
    if "stability" in s:
        unstable = s["stability"]["candidate"]["unstable_share"]
        facts.append((f"{100 * unstable:.2f}%", "unstable ids (candidate)"))
    if "invalid_share" in s:
        facts.append((f"{100 * s['invalid_share']['candidate']:.3f}%", "invalid answers"))
    auc = s.get("auc")
    if auc:
        facts.append((_fmt(auc["delta"]), "AUC difference"))
    gates_rows = "".join(
        f"<tr><td><code>{escape(g['name'])}</code></td><td>{_status(g['passed'])}</td>"
        f"<td class='num'>{escape(_fmt(g['value']))}</td>"
        f"<td class='num'>{escape(_fmt(g['threshold']))}</td>"
        f"<td>{escape(g['description'])}</td></tr>"
        for g in doc["gates"]
    )
    inputs = "".join(
        f"<div>{escape(name)}: <code>{escape(str(meta.get('path', '')))}</code> "
        f"sha256 <code>{escape(str(meta.get('sha256', '')))}</code></div>"
        for name, meta in doc.get("inputs", {}).items()
    )
    preset = doc["config"].get("preset")
    preset_note = f" Preset: <code>{escape(str(preset))}</code>." if preset else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Score parity: {doc['verdict']}</title><style>{_CSS}</style></head><body><main>"
        "<h1>Score parity report</h1>"
        f'<p class="lead">Are the candidate scores equivalent to the reference within the '
        f"declared tolerance?{preset_note}</p>"
        f'<div class="verdict {"pass" if passed else "fail"}">'
        f"{'✓ PASS' if passed else '✕ FAIL'}</div>"
        '<div class="facts">'
        + "".join(
            f'<div class="fact"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for v, k in facts
        )
        + "</div><h2>Gates</h2><div class='card'><table><tr><th>Gate</th><th>Result</th>"
        "<th class='num'>Value</th><th class='num'>Threshold</th><th>What it checks</th></tr>"
        f"{gates_rows}</table></div>"
        f"{_histogram(doc)}{_confusion(doc)}{_segments(doc)}{_examples(doc)}"
        f"<footer>{inputs}<div>scoreparity {escape(doc['environment']['scoreparity'])} · "
        f"report schema {doc['schema_version']}</div></footer></main></body></html>"
    )
