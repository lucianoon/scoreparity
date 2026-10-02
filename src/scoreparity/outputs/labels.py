"""Sets of class labels (multi-label outputs)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scoreparity import gates, gates_categorical, gates_multilabel
from scoreparity.align import align_label_sets
from scoreparity.config import ParityConfig
from scoreparity.gates import GateResult


class LabelsOutput:
    name = "labels"

    def compare(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        cfg: ParityConfig,
        context: pd.DataFrame | None,
    ) -> tuple[list[GateResult], dict[str, Any]]:
        alignment = align_label_sets(
            reference,
            candidate,
            cfg.columns,
            cfg.segments,
            context,
            cfg.output.normalize,
            cfg.output.label_separator,
        )
        results = gates.common_gates(alignment, cfg, missing_text="no labels at all")
        results += gates_multilabel.evaluate(alignment, cfg)
        out = gates_multilabel.summary(alignment, cfg)
        if cfg.examples:
            out["examples"] = gates_categorical.examples(alignment, cfg)
        return results, out
