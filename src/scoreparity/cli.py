"""Command-line interface.

Exit codes are part of the public contract, so CI systems can rely on them:
    0  every gate passed
    1  at least one gate failed (the candidate is not equivalent)
    2  usage or input error (the comparison could not be made)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from scoreparity import __version__
from scoreparity.config import ParityConfig, from_dict, load
from scoreparity.errors import ScoreParityError
from scoreparity.presets import PRESETS

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


def _cmd_compare(args: argparse.Namespace) -> int:
    from scoreparity.compare import compare_files

    report = compare_files(args.reference, args.candidate, _resolve_config(args), args.context)
    if args.json:
        Path(args.json).write_text(report.to_json() + "\n", encoding="utf-8")
    markdown = report.to_markdown()
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(markdown)
    return EXIT_PASS if report.passed else EXIT_FAIL


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
        return _cmd_init(args)
    except ScoreParityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
