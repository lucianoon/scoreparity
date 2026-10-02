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
    cmp.add_argument(
        "--output-type",
        choices=["score", "label", "probabilities", "labels"],
        help="what each row holds (overrides the config; default score)",
    )
    cmp.add_argument("--truth", help="class ground-truth column (label/probabilities outputs)")
    cmp.add_argument("--prob-prefix", help="prefix of the probability columns (default p_)")
    cmp.add_argument(
        "--replica",
        metavar="COLUMN",
        help="replica column: several rows per id, compared by majority class (labels)",
    )
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
    cmp.add_argument(
        "--examples",
        type=int,
        metavar="N",
        help="list up to N changed rows by id in the report (overrides the config; default 0)",
    )
    cmp.add_argument("--quiet", action="store_true", help="do not print the report to stdout")

    pln = sub.add_parser(
        "plan-sample",
        help="how many rows to score so a class gate can pass (before paying for them)",
        description=(
            "Sample size for the rate gates (label_agreement, transitions, invalid_rate): the "
            "smallest number of rows from which a gate proving 'rate <= max' passes with the "
            "requested power when the true rate is --expected-rate. Exact binomial computation."
        ),
    )
    target = pln.add_mutually_exclusive_group(required=True)
    target.add_argument("--max-rate", type=float, help="tolerance to prove (transitions, invalid)")
    target.add_argument(
        "--min-agreement", type=float, help="label_agreement threshold (max rate = 1 - this)"
    )
    expect = pln.add_mutually_exclusive_group()
    expect.add_argument(
        "--expected-rate", type=float, help="true rate you expect (e.g. from a noise run)"
    )
    expect.add_argument("--expected-agreement", type=float, help="agreement you expect")
    pln.add_argument("--power", type=float, default=0.8, help="probability of passing (0.8)")
    pln.add_argument("--alpha", type=float, default=0.05, help="as in the config (0.05)")
    pln.add_argument(
        "--class-share",
        type=float,
        help="share of rows in the smallest class (transitions are verified per class)",
    )
    pln.add_argument("--cost-per-row", type=float, help="price of scoring one row once")
    pln.add_argument("--versions", type=int, default=2, help="versions to score (2)")
    pln.add_argument("--json", metavar="PATH", help="write the plan as JSON")

    noi = sub.add_parser(
        "noise",
        help="measure the noise floor from replicate runs and suggest tolerances",
        description=(
            "Score the SAME reference model again under conditions that should not matter "
            "(another batch size, machine, thread count) and pass those files as replicates. "
            "Writes a configuration whose tolerances cover the measured noise x --safety."
        ),
    )
    noi.add_argument("--reference", required=True, help="reference scores (.csv or .parquet)")
    noi.add_argument(
        "--replicate",
        action="append",
        required=True,
        metavar="PATH",
        help="outputs of a rerun of the reference model; repeat (3+ recommended)",
    )
    noi.add_argument("--context", help="optional table with labels/segments keyed by id")
    noi.add_argument("--config", help="base config: columns, segments, alpha (gates are replaced)")
    noi.add_argument("--id")
    noi.add_argument("--score")
    noi.add_argument("--reference-score")
    noi.add_argument("--candidate-score", help="score column in the replicate tables")
    noi.add_argument("--label")
    noi.add_argument("--segment", action="append", default=None, metavar="COLUMN")
    noi.add_argument(
        "--output-type",
        choices=["score", "label"],
        help="what each row holds (overrides the config; default score)",
    )
    noi.add_argument(
        "--safety",
        type=float,
        default=None,
        help="multiplier over the noise, >= 1 (default 3 for scores, 1.75 for labels)",
    )
    noi.add_argument("--q", type=float, default=0.99, help="quantile for quantile_abs_diff")
    noi.add_argument("--threshold", action="append", type=float, help="decision threshold(s)")
    noi.add_argument("--top-k", action="append", type=float, help="top-k percentage(s)")
    noi.add_argument("--out", metavar="PATH", help="write the suggested YAML here (default stdout)")
    noi.add_argument("--json", metavar="PATH", help="write the measured noise profile as JSON")
    noi.add_argument("--force", action="store_true", help="overwrite --out if it exists")

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
        # Build with the output type so a preset is filtered to the gates that apply to it.
        raw: dict[str, object] = {"preset": args.preset} if args.preset else {}
        if getattr(args, "output_type", None):
            raw["output"] = {"type": args.output_type}
        cfg = from_dict(raw)
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
                "truth": getattr(args, "truth", None),
                "replica": getattr(args, "replica", None),
            }.items()
            if value is not None
        },
    )
    output = cfg.output
    if getattr(args, "output_type", None):
        output = dataclasses.replace(output, type=args.output_type)
    if getattr(args, "prob_prefix", None):
        output = dataclasses.replace(output, prob_prefix=args.prob_prefix)
    segments = tuple(args.segment) if args.segment is not None else cfg.segments
    examples = getattr(args, "examples", None)
    return dataclasses.replace(
        cfg,
        columns=columns,
        segments=segments,
        output=output,
        examples=cfg.examples if examples is None else examples,
    )


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


def _cmd_noise(args: argparse.Namespace) -> int:
    from scoreparity.compare import read_table
    from scoreparity.noise import measure_noise, suggested_config_yaml

    args.preset = None
    base = _resolve_config(args)
    if not 0 < args.q < 1:
        raise InputError("--q must be in (0, 1)")
    out = Path(args.out) if args.out else None
    if out is not None and out.exists() and not args.force:
        raise InputError(f"{out} already exists (use --force to overwrite)")
    replicates = {}
    for path in args.replicate:
        name = Path(path).name if Path(path).name not in replicates else path
        replicates[name] = read_table(path)
    profile = measure_noise(
        read_table(args.reference),
        replicates,
        base,
        context=read_table(args.context) if args.context else None,
        safety=args.safety,
        q=args.q,
        thresholds=tuple(args.threshold or [0.5]),
        k_pct=tuple(args.top_k or [10.0]),
    )
    text = suggested_config_yaml(profile, base)
    if args.json:
        Path(args.json).write_text(
            json.dumps(profile.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        _write_stdout(text)
    return EXIT_PASS


def _cmd_plan_sample(args: argparse.Namespace) -> int:
    from scoreparity.planning import describe, plan_sample

    max_rate = args.max_rate if args.max_rate is not None else 1 - args.min_agreement
    if args.expected_agreement is not None:
        expected = 1 - args.expected_agreement
    else:
        expected = args.expected_rate or 0.0
    try:
        plan = plan_sample(
            max_rate,
            expected,
            alpha=args.alpha,
            power=args.power,
            class_share=args.class_share,
            cost_per_row=args.cost_per_row,
            versions=args.versions,
        )
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    if args.json:
        Path(args.json).write_text(
            json.dumps(plan.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    _write_stdout(describe(plan))
    return EXIT_PASS


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
        if args.command == "noise":
            return _cmd_noise(args)
        if args.command == "plan-sample":
            return _cmd_plan_sample(args)
        return _cmd_init(args)
    except ScoreParityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
