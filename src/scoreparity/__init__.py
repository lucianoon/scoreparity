"""scoreparity: score parity gates for ML model changes."""

__version__ = "0.1.0"

from scoreparity.compare import compare, compare_files, read_table
from scoreparity.config import ParityConfig, from_dict, load
from scoreparity.errors import ConfigError, InputError, ScoreParityError
from scoreparity.report import Report

__all__ = [
    "ConfigError",
    "InputError",
    "ParityConfig",
    "Report",
    "ScoreParityError",
    "__version__",
    "compare",
    "compare_files",
    "from_dict",
    "load",
    "read_table",
]
