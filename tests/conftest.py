from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_scores(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Reference scores with a label and two segment columns, like a real scoring table."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    label = rng.random(n) < 1 / (1 + np.exp(-(latent - 1)))
    frame: pd.DataFrame = pd.DataFrame(
        {
            "id": [f"c{i:06d}" for i in range(n)],
            "score": 1 / (1 + np.exp(-latent)),
            "label": label.astype(int),
            "plan": rng.choice(["pre", "pos", "controle"], n),
            "region": rng.choice(["N", "NE", "S", "SE"], n),
        }
    )
    return frame


@pytest.fixture
def reference() -> pd.DataFrame:
    return make_scores()
