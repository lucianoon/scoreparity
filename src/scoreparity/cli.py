"""Command-line interface.

Exit codes are part of the public contract, so CI systems can rely on them:
    0  every gate passed
    1  at least one gate failed (the candidate is not equivalent)
    2  usage or input error (the comparison could not be made)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from scoreparity import __version__

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoreparity",
        description="Prove that two versions of a model produce equivalent scores.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR  # pragma: no cover - subcommands arrive in the next phase


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
