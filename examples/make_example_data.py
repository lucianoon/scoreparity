"""Generate example score files: a reference and three candidates.

    python examples/make_example_data.py OUTPUT_DIR

- reference.csv        current model: id, score, label, plan
- candidate_same.csv   same model after a refactor (float32 round-off only)  -> PASS float-noise
- candidate_drift.csv  one segment shifted by +0.002                          -> FAIL
- candidate_broken.csv a bug: 1% of rows NaN and 2% of rows dropped           -> FAIL

Deterministic (fixed seed) and dependency-light (numpy + pandas), so it doubles as a demo
and as a fixture for the GitHub Action self-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    n = 20_000
    latent = rng.normal(0, 1, n)
    reference = pd.DataFrame(
        {
            "id": [f"cust-{i:06d}" for i in range(n)],
            "score": 1 / (1 + np.exp(-latent)),
            "label": (rng.random(n) < 1 / (1 + np.exp(-(1.5 * latent - 1)))).astype(int),
            "plan": rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    reference.to_csv(out / "reference.csv", index=False)

    same = reference[["id", "score"]].copy()
    same["score"] = same["score"].astype(np.float32).astype(np.float64)
    same.to_csv(out / "candidate_same.csv", index=False)

    drift = reference[["id", "score"]].copy()
    drift.loc[reference["plan"] == "pro", "score"] += 0.002
    drift.to_csv(out / "candidate_drift.csv", index=False)

    broken = reference[["id", "score"]].copy()
    broken.loc[rng.random(n) < 0.01, "score"] = np.nan
    broken = broken[rng.random(n) >= 0.02]
    broken.to_csv(out / "candidate_broken.csv", index=False)
    print(f"wrote 4 files to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "example-data")
