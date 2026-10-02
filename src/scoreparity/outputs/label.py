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
        alignment = align_labels(
            reference, candidate, cfg.columns, cfg.segments, context, cfg.output.normalize
        )
        results = gates.common_gates(alignment, cfg, missing_text="a missing label")
        results += gates_categorical.evaluate(alignment, cfg)
        out = gates_categorical.summary(alignment, cfg)
        if cfg.examples:
            out["examples"] = gates_categorical.examples(alignment, cfg)
        return results, out
