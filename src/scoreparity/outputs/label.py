"""Class labels (one predicted class per row)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scoreparity import gates, gates_categorical
from scoreparity.align import align_labels
from scoreparity.config import ParityConfig
from scoreparity.gates import GateResult


class LabelOutput:
    name = "label"

    def compare(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        cfg: ParityConfig,
        context: pd.DataFrame | None,
    ) -> tuple[list[GateResult], dict[str, Any]]:
        alignment = align_labels(reference, candidate, cfg.columns, cfg.segments, context)
        results = gates.common_gates(alignment, cfg, missing_text="a missing label")
        results += gates_categorical.evaluate(alignment, cfg)
        return results, gates_categorical.summary(alignment)
