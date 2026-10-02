"""Named starting points for common kinds of change.

Presets are *starting points*, not universal truths: the right tolerance depends on how the
score is used. They assume probability-like scores in [0, 1]. Explicit gates in a config
override the preset, and setting a gate to null disables it.

- exact: the change must not alter a single bit (pure refactors on the same hardware and
  library versions, deterministic pipelines).
- float-noise: only floating-point reordering is allowed (different batch sizes, thread counts,
  BLAS builds, CPU vs CPU). Differences of this kind are typically below 1e-6 in float32.
- quantization: deliberate precision loss (fp16, bf16, int8). Individual scores may move
  noticeably, so the gates focus on decisions, ranking and aggregate quality.
"""

from __future__ import annotations

from typing import Any

from scoreparity.errors import ConfigError

PRESETS: dict[str, dict[str, Any]] = {
    "exact": {
        "max_abs_diff": {"max": 0.0},
        "decision_flips": {"thresholds": [0.5], "max_rate": 0.0},
    },
    "float-noise": {
        "max_abs_diff": {"max": 1e-5},
        "quantile_abs_diff": {"q": 0.99, "max": 1e-6},
        "mean_diff_equivalence": {"margin": 1e-6, "per_segment": True},
        "decision_flips": {"thresholds": [0.5], "max_rate": 0.0001},
        "top_k_overlap": {"k_pct": [10], "min_overlap": 0.999},
    },
    "quantization": {
        "max_abs_diff": {"max": 0.05},
        "quantile_abs_diff": {"q": 0.99, "max": 0.01},
        "mean_diff_equivalence": {"margin": 0.002, "per_segment": True},
        "decision_flips": {"thresholds": [0.5], "max_rate": 0.005},
        "top_k_overlap": {"k_pct": [10], "min_overlap": 0.98},
    },
}


def preset_gates(name: str) -> dict[str, Any]:
    try:
        return PRESETS[name]
    except KeyError:
        raise ConfigError(
            f"unknown preset {name!r}; available presets: {sorted(PRESETS)}"
        ) from None
