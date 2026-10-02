"""Continuous scores (one number per row): the original scoreparity behaviour."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scoreparity import gates
from scoreparity.align import DIFF, align
from scoreparity.config import ParityConfig
from scoreparity.gates import GateResult


class ScoreOutput:
    name = "score"

    def compare(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        cfg: ParityConfig,
        context: pd.DataFrame | None,
    ) -> tuple[list[GateResult], dict[str, Any]]:
        alignment = align(reference, candidate, cfg.columns, cfg.segments, context)
        out = gates.summary(alignment)
        if cfg.examples:
            abs_diff = alignment.frame[DIFF].abs().to_numpy(np.float64)
            out["examples"] = gates.examples(alignment, cfg, abs_diff)
        return gates.evaluate(alignment, cfg), out
