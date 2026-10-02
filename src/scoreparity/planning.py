"""Sample-size planning for the class gates: how many rows to score before comparing.

When every row costs money (an LLM call per row and per version), the two classic mistakes are
running too few rows, so the gate cannot pass even when nothing changed, and running far more
than needed. The rate gates (`label_agreement`, `transitions`, `invalid_rate`) all prove a
claim of the form "the event rate is at most `max_rate`" with a one-sided Clopper-Pearson upper
bound. For a true rate `expected_rate` below `max_rate`, this module finds how many rows give
that proof with the requested power (probability of passing).

Exactness: the Clopper-Pearson upper bound for k events in n rows is at most `max_rate` exactly
when P(X <= k | n, max_rate) <= alpha. The gate therefore passes when at most k*(n) events are
observed, with k*(n) the largest such k, and the power is P(X <= k*(n) | n, expected_rate),
computed from the binomial distribution, without approximation.

The power is not monotone in n (the binomial "sawtooth"): one more row can lower it slightly.
The plan returns the smallest n from which *every* larger sample reaches the requested power, so
collecting a few extra rows never hurts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

# Hard cap on the search; far beyond any sensible plan (1e-6 rates and below are not testable
# with sampled data at a reasonable cost).
MAX_ROWS = 5_000_000


@dataclass(frozen=True)
class SamplePlan:
    max_rate: float
    expected_rate: float
    alpha: float
    power: float
    rows: int  # per class for `transitions`, in total for the other gates
    max_events: int  # the gate passes with at most this many events in `rows` rows
    achieved_power: float
    smallest_rows: int  # first size reaching the power; some sizes between it and `rows` do not
    class_share: float | None = None
    total_rows: int | None = None
    versions: int = 2
    cost_per_row: float | None = None
    estimated_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _max_events(n: np.ndarray, max_rate: float, alpha: float) -> np.ndarray:
    """Largest k with P(X <= k | n, max_rate) <= alpha, or -1 when even k = 0 is too many."""
    k = stats.binom.ppf(alpha, n, max_rate).astype(np.int64)  # smallest k with cdf >= alpha
    over = stats.binom.cdf(k, n, max_rate) > alpha
    return np.asarray(np.where(over, k - 1, k))


def power_at(n: int, max_rate: float, expected_rate: float, alpha: float = 0.05) -> float:
    """Probability that a rate gate with `n` rows passes when the true rate is `expected_rate`."""
    k = _max_events(np.array([n]), max_rate, alpha)[0]
    return float(stats.binom.cdf(k, n, expected_rate)) if k >= 0 else 0.0


def plan_sample(
    max_rate: float,
    expected_rate: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    class_share: float | None = None,
    cost_per_row: float | None = None,
    versions: int = 2,
) -> SamplePlan:
    """Rows needed so that a rate gate passes with probability `power` at `expected_rate`."""
    if not 0 < max_rate < 1:
        raise ValueError("max_rate must be in (0, 1)")
    if not 0 <= expected_rate < max_rate:
        raise ValueError(
            "expected_rate must be in [0, max_rate): a tolerance at or below the true rate "
            "cannot be proven with any sample size"
        )
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if class_share is not None and not 0 < class_share <= 1:
        raise ValueError("class_share must be in (0, 1]")
    if versions < 1:
        raise ValueError("versions must be >= 1")

    if expected_rate == 0:
        # Zero events are certain, so the first n where zero events suffice is the answer and
        # every larger n also works.
        rows = math.ceil(math.log(alpha) / math.log(1 - max_rate))
        k_max, achieved, smallest = 0, 1.0, rows
    else:
        # Normal approximation for the order of magnitude, then an exact scan well past it.
        z_a, z_b = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
        sd_null = math.sqrt(max_rate * (1 - max_rate))
        sd_alt = math.sqrt(expected_rate * (1 - expected_rate))
        approx = ((z_a * sd_null + z_b * sd_alt) / (max_rate - expected_rate)) ** 2
        end = int(min(MAX_ROWS, 3 * approx + 2000))
        n = np.arange(1, end + 1)
        k = _max_events(n, max_rate, alpha)
        pw = np.where(k >= 0, stats.binom.cdf(k, n, expected_rate), 0.0)
        short = np.flatnonzero(pw < power)
        if len(short) and short[-1] == end - 1:
            raise ValueError(f"more than {end:,} rows would be needed; loosen the tolerance")
        first = int(short[-1]) + 1 if len(short) else 0  # every n after the last shortfall
        rows, k_max, achieved = int(n[first]), int(k[first]), float(pw[first])
        smallest = int(n[np.flatnonzero(pw >= power)[0]])

    total = math.ceil(rows / class_share) if class_share is not None else None
    billed = total if total is not None else rows
    return SamplePlan(
        max_rate=max_rate,
        expected_rate=expected_rate,
        alpha=alpha,
        power=power,
        rows=rows,
        max_events=k_max,
        achieved_power=achieved,
        smallest_rows=smallest,
        class_share=class_share,
        total_rows=total,
        versions=versions,
        cost_per_row=cost_per_row,
        estimated_cost=billed * versions * cost_per_row if cost_per_row is not None else None,
    )


def describe(plan: SamplePlan) -> str:
    """A short human-readable explanation of a plan."""
    level = f"{100 * (1 - plan.alpha):g}%"
    lines = [
        f"To show that the rate is at most {plan.max_rate:.4g} (one-sided {level} bound) when "
        f"the true rate is {plan.expected_rate:.4g}, with {100 * plan.power:g}% power:",
        f"  rows needed: {plan.rows:,}"
        + (" per class" if plan.class_share is not None else "")
        + f" (the gate passes with at most {plan.max_events:,} events; "
        f"power at this size {100 * plan.achieved_power:.1f}%)",
    ]
    if plan.smallest_rows < plan.rows:
        lines.append(
            f"  {plan.smallest_rows:,} rows already reach the power, but some sizes between "
            f"{plan.smallest_rows:,} and {plan.rows:,} fall just short (binomial sawtooth)"
        )
    if plan.total_rows is not None:
        lines.append(
            f"  smallest class is {100 * (plan.class_share or 0):g}% of the rows: "
            f"{plan.total_rows:,} rows in total"
        )
    if plan.estimated_cost is not None:
        billed = plan.total_rows if plan.total_rows is not None else plan.rows
        lines.append(
            f"  estimated cost: {billed:,} rows x {plan.versions} version(s) x "
            f"{plan.cost_per_row:g} = {plan.estimated_cost:,.2f}"
        )
    lines.append(
        "  Every sample of at least the rows needed reaches the power. "
        "label_agreement: max rate = 1 - min; transitions: per source class; invalid_rate: max."
    )
    return "\n".join(lines) + "\n"
