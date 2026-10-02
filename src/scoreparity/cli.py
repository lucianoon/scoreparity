"""Command-line interface.

Exit codes are part of the public contract, so CI systems can rely on them:
    0  every gate passed
    1  at least one gate failed (the candidate is not equivalent)
    2  usage or input error (the comparison could not be made)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from scoreparity import __version__
from scoreparity.config import ParityConfig, from_dict, load
from scoreparity.errors import InputError, ScoreParityError
from scoreparity.presets import PRESETS
from scoreparity.report import SCHEMA_VERSION

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

CONFIG_TEMPLATE = """\
# scoreparity configuration. Declare what "equivalent" means BEFORE looking at results.
# Docs: https://github.com/lucianoon/scoreparity
version: 1

# Starting point: exact | float-noise | quantization. Gates below override it;
# set a gate to null to disable it.
preset: {preset}

columns:
  id: id              # record identifier present in both tables
  score: score        # score column (or set reference_score / candidate_score)
  # label: label      # binary outcome, enables the AUC gate

# Columns used to check every segment separately (a global average can hide
# segments that moved in opposite directions).
segments: []
min_segment_size: 30
alpha: 0.05

gates:
  # max_abs_diff: {{max: 1.0e-5}}
  # quantile_abs_diff: {{q: 0.99, max: 1.0e-6}}
  # mean_diff_equivalence: {{margin: 1.0e-6, per_segment: true}}
  # decision_flips: {{thresholds: [0.5], max_rate: 0.0}}
  # top_k_overlap: {{k_pct: [1, 10], min_overlap: 0.99}}
  # auc_difference: {{margin: 0.001}}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoreparity",
        description="Prove that two versions of a model produce equivalent scores.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    cmp = sub.add_parser(
        "compare",
        help="compare reference and candidate scores",
        description="Compare two score tables. Exit code 0 = equivalent, 1 = not, 2 = error.",
    )
    cmp.add_argument("--reference", required=True, help="reference scores (.csv or .parquet)")
    cmp.add_argument("--candidate", required=True, help="candidate scores (.csv or .parquet)")
    cmp.add_argument("--context", help="optional table with labels/segments keyed by id")
    src = cmp.add_mutually_exclusive_group()
    src.add_argument("--config", help="YAML configuration file")
    src.add_argument("--preset", choices=sorted(PRESETS), help="use a preset without a config file")
    cmp.add_argument("--id", help="id column (overrides the config)")
    cmp.add_argument("--score", help="score column in both tables (overrides the config)")
    cmp.add_argument("--reference-score", help="score column in the reference table")
    cmp.add_argument("--candidate-score", help="score column in the candidate table")
    cmp.add_argument("--label", help="binary label column (enables AUC reporting/gate)")
    cmp.add_argument("--html", metavar="PATH", help="write the self-contained HTML report here")
    cmp.add_argument("--junit", metavar="PATH", help="write a JUnit XML file (one test per gate)")
    cmp.add_argument(
        "--segment",
        action="append",
        default=None,
        metavar="COLUMN",
        help="segment column; repeat for several (overrides the config)",
    )
    cmp.add_argument("--json", metavar="PATH", help="write the JSON report here")
    cmp.add_argument("--markdown", metavar="PATH", help="write the Markdown report here")
    cmp.add_argument("--quiet", action="store_true", help="do not print the report to stdout")

    ren = sub.add_parser(
        "render",
        help="re-render a saved JSON report as Markdown, HTML or JUnit",
        description="Exit code mirrors the report verdict: 0 PASS, 1 FAIL, 2 unreadable report.",
    )
    ren.add_argument("report", help="JSON report written by `compare --json`")
    ren.add_argument("--markdown", metavar="PATH")
    ren.add_argument("--html", metavar="PATH")
    ren.add_argument("--junit", metavar="PATH")
    ren.add_argument("--quiet", action="store_true", help="do not print the report to stdout")

    init = sub.add_parser("init", help="write a commented configuration file to start from")
    init.add_argument("path", nargs="?", default="scoreparity.yaml")
    init.add_argument("--preset", choices=sorted(PRESETS), default="float-noise")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    return parser


def _resolve_config(args: argparse.Namespace) -> ParityConfig:
    if args.config:
        cfg = load(args.config)
    else:
        cfg = from_dict({"preset": args.preset} if args.preset else {})
    columns = dataclasses.replace(
        cfg.columns,
        **{
            name: value
            for name, value in {
                "id": args.id,
                "score": args.score,
                "reference_score": args.reference_score,
                "candidate_score": args.candidate_score,
                "label": args.label,
            }.items()
            if value is not None
        },
    )
    segments = tuple(args.segment) if args.segment is not None else cfg.segments
    return dataclasses.replace(cfg, columns=columns, segments=segments)


def _write_stdout(text: str) -> None:
    """Write to stdout even when its encoding cannot represent the text.

    Windows consoles and CI runners often use cp1252, which cannot encode the report's status
    icons. A parity gate must never crash on output, so unencodable characters are replaced.
    """
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.flush()
        sys.stdout.buffer.write(text.encode(encoding, errors="replace"))
        sys.stdout.buffer.flush()


def _write_outputs(doc: dict[str, object], args: argparse.Namespace) -> None:
    from scoreparity.report import render_markdown
    from scoreparity.report_html import render_html
    from scoreparity.report_junit import render_junit

    renderers: dict[str, Callable[[], str]] = {
        "json": lambda: json.dumps(doc, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        "markdown": lambda: render_markdown(doc),
        "html": lambda: render_html(doc),
        "junit": lambda: render_junit(doc),
    }
    for name, render in renderers.items():
        target = getattr(args, name, None)
        if target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(), encoding="utf-8")
    if not args.quiet:
        _write_stdout(render_markdown(doc))


def _cmd_compare(args: argparse.Namespace) -> int:
    from scoreparity.compare import compare_files

    report = compare_files(args.reference, args.candidate, _resolve_config(args), args.context)
    _write_outputs(report.to_dict(), args)
    return EXIT_PASS if report.passed else EXIT_FAIL


def _cmd_render(args: argparse.Namespace) -> int:
    """Re-render a saved JSON report (e.g. produced by a pipeline job) in other formats."""
    try:
        doc = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{args.report}: cannot read JSON report: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        raise InputError(
            f"{args.report}: not a scoreparity report with schema_version {SCHEMA_VERSION}"
        )
    args.json = None
    _write_outputs(doc, args)
    return EXIT_PASS if doc["verdict"] == "PASS" else EXIT_FAIL


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        return EXIT_ERROR
    path.write_text(CONFIG_TEMPLATE.format(preset=args.preset), encoding="utf-8")
    load(path)  # the template must always be a valid configuration
    print(f"wrote {path}")
    return EXIT_PASS


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits with 2 on usage errors; keep 0 for --help
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    try:
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "render":
            return _cmd_render(args)
        return _cmd_init(args)
    except ScoreParityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
