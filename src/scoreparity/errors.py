"""Exceptions raised by scoreparity.

Every exception here means the comparison *could not be made* (CLI exit code 2). A candidate
that is simply not equivalent is never an exception: it is a failed gate in the report.
"""

from __future__ import annotations


class ScoreParityError(Exception):
    """Base class for all scoreparity errors."""


class ConfigError(ScoreParityError):
    """The parity configuration is invalid."""


class InputError(ScoreParityError):
    """The score tables cannot be compared (missing columns, duplicate ids, wrong types)."""
