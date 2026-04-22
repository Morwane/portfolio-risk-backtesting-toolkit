"""Rebalancing schedule generation.

Produces the dates at which the portfolio is rebalanced to target weights,
and computes transaction costs for each rebalancing event.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_rebalancing_dates(
    index: pd.DatetimeIndex,
    frequency: str = "quarterly",
) -> pd.DatetimeIndex:
    """Return the rebalancing dates within *index*.

    Args:
        index: Full business-day DatetimeIndex of the backtest.
        frequency: One of ``"monthly"``, ``"quarterly"``, ``"annual"``,
                   ``"buy_and_hold"`` (rebalance only at start).

    Returns:
        DatetimeIndex of rebalancing dates (subset of *index*).
    """
    if frequency == "buy_and_hold":
        return pd.DatetimeIndex([index[0]])

    freq_map = {
        "monthly": "MS",   # month start
        "quarterly": "QS", # quarter start
        "annual": "YS",    # year start
    }
    if frequency not in freq_map:
        raise ValueError(
            f"Unknown rebalancing frequency '{frequency}'. "
            f"Choose from: {list(freq_map.keys()) + ['buy_and_hold']}"
        )

    calendar_dates = pd.date_range(
        start=index.min(),
        end=index.max(),
        freq=freq_map[frequency],
    )

    # Snap each calendar date to the nearest business day in the index
    rebal_dates = []
    for cal_date in calendar_dates:
        candidates = index[index >= cal_date]
        if len(candidates) > 0:
            rebal_dates.append(candidates[0])

    result = pd.DatetimeIndex(sorted(set(rebal_dates)))
    logger.debug(
        "Rebalancing schedule: %s -> %d dates between %s and %s",
        frequency,
        len(result),
        result[0].date() if len(result) else "N/A",
        result[-1].date() if len(result) else "N/A",
    )
    return result


def compute_rebalancing_costs(
    weights_before: Dict[str, float],
    weights_after: Dict[str, float],
    portfolio_value: float,
    cost_bps: float = 2.0,
) -> float:
    """Compute one-way transaction cost for a rebalancing event.

    Cost = sum(|w_after - w_before|) * portfolio_value * cost_bps / 10000.
    Only buys (or sells) are counted once (half-turn assumption).

    Args:
        weights_before: Current drift weights before rebalancing.
        weights_after: Target weights after rebalancing.
        portfolio_value: Current NAV.
        cost_bps: One-way cost in basis points.

    Returns:
        Total transaction cost in NAV units.
    """
    all_keys = set(weights_before) | set(weights_after)
    total_turnover = sum(
        abs(weights_after.get(k, 0.0) - weights_before.get(k, 0.0))
        for k in all_keys
    )
    # Divide by 2: each rebalance creates both a buy and a sell of equal size
    one_way_turnover = total_turnover / 2.0
    cost = one_way_turnover * portfolio_value * cost_bps / 10_000.0
    return cost
