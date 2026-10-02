"""Regenerate the golden reports used by tests/test_golden.py.

Run it with the RELEASED version whose behaviour must be preserved, not with the working tree:

    uv run --isolated \
        --with "scoreparity[parquet] @ git+https://github.com/lucianoon/scoreparity@v0.1.0" \
        python tests/golden/make_golden.py

The scenarios are deterministic (fixed seeds) and only use public API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scoreparity as sp

HERE = Path(__file__).resolve().parent


def scenario_frames() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    rng = np.random.default_rng(20261002)
    n = 3000
    latent = rng.normal(0, 1, n)
    ref = pd.DataFrame(
        {
            "id": [f"r{i:05d}" for i in range(n)],
            "score": 1 / (1 + np.exp(-latent)),
            "label": (rng.random(n) < 1 / (1 + np.exp(-(1.4 * latent - 1)))).astype(int),
            "plan": rng.choice(["a", "b", "c"], n),
        }
    )
    noise = ref.assign(score=ref["score"] + rng.normal(0, 1e-8, n))
    drift = ref.assign(score=ref["score"] + np.where(ref["plan"] == "c", 2e-3, 0.0))
    broken = ref.copy()
    broken.loc[:9, "score"] = np.nan
    broken = broken.iloc[20:]
    return {"noise": (ref, noise), "drift": (ref, drift), "broken": (ref, broken)}


CONFIG = {
    "preset": "float-noise",
    "columns": {"label": "label"},
    "segments": ["plan"],
    "gates": {"auc_difference": {"margin": 0.001}},
}


def stable(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that legitimately differ between versions (environment, config echo)."""
    return {k: v for k, v in doc.items() if k not in ("environment", "config", "inputs")}


def main() -> None:
    cfg = sp.from_dict(CONFIG)
    out: dict[str, Any] = {
        name: stable(json.loads(sp.compare(ref, cand, cfg).to_json()))
        for name, (ref, cand) in scenario_frames().items()
    }
    out["_generated_with"] = sp.__version__
    path = HERE / "reports-v0.1.0.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path} with scoreparity {sp.__version__}", file=sys.stderr)


if __name__ == "__main__":
    main()
