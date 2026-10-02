"""Output kinds: how a given kind of model output is aligned, gated and summarised.

`compare` dispatches on `config.output.type`. Adding a new kind of output (class labels,
probability vectors, ...) means adding a module here and registering it; the comparison entry
point, the report and the CLI do not change.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from scoreparity.config import ParityConfig
from scoreparity.gates import GateResult


class OutputKind(Protocol):
    """The contract every output kind implements."""

    name: str

    def compare(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        cfg: ParityConfig,
        context: pd.DataFrame | None,
    ) -> tuple[list[GateResult], dict[str, Any]]:
        """Align both tables, evaluate the configured gates and build the summary.

        Raises `InputError` when the tables cannot be compared at all.
        """
        ...


def get_output_kind(name: str) -> OutputKind:
    from scoreparity.outputs.label import LabelOutput
    from scoreparity.outputs.probabilities import ProbabilitiesOutput
    from scoreparity.outputs.score import ScoreOutput

    kinds: dict[str, OutputKind] = {
        "score": ScoreOutput(),
        "label": LabelOutput(),
        "probabilities": ProbabilitiesOutput(),
    }
    return kinds[name]
