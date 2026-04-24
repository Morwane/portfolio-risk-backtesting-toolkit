"""Turnover and transaction cost analysis.

Builds a per-rebalancing-event report from the weights history produced by
build_portfolio(..., track_rebalances=True).

Outputs:
  outputs/tables/turnover_report.csv
  outputs/tables/rebalance_trade_summary.csv
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_turnover_report(
    weights_history: pd.DataFrame,
    target_weights: Dict[str, float],
    portfolio_id: str = "portfolio",
) -> pd.DataFrame:
    """Compute turnover and cost statistics per rebalancing event.

    Args:
        weights_history: Output of build_portfolio() with track_rebalances=True.
                         Must contain a ``_rebal_cost`` column.
        target_weights: The target sleeve weights (for labelling).
        portfolio_id: Portfolio identifier for the report.

    Returns:
        DataFrame with one row per rebalancing event containing:
        date, one_way_turnover, cost_deducted, cumulative_cost, nav_at_rebal.
    """
    sleeve_cols = [c for c in weights_history.columns if not c.startswith("_")]

    if "_rebal_cost" not in weights_history.columns:
        logger.warning(
            "weights_history for '%s' has no _rebal_cost column. "
            "Rebuild with track_rebalances=True for a full turnover report.",
            portfolio_id,
        )
        return pd.DataFrame(columns=["date", "one_way_turnover_pct", "cost_deducted", "cumulative_cost"])

    # Rebalancing events are rows where _rebal_cost > 0
    rebal_mask = weights_history["_rebal_cost"] > 0
    rebal_dates = weights_history[rebal_mask].index

    if len(rebal_dates) == 0:
        logger.info("No rebalancing events found for '%s' (buy-and-hold).", portfolio_id)
        return pd.DataFrame(columns=["date", "one_way_turnover_pct", "cost_deducted", "cumulative_cost"])

    rows: List[Dict] = []
    cumulative_cost = 0.0

    for date in rebal_dates:
        cost = float(weights_history.loc[date, "_rebal_cost"])
        cumulative_cost += cost

        # Turnover = ½ Σ |w_i,t - w_i,target| (one-way)
        pre_rebal_idx = weights_history.index.get_loc(date)
        if pre_rebal_idx > 0:
            prev_date = weights_history.index[pre_rebal_idx - 1]
            pre_weights = weights_history.loc[prev_date, sleeve_cols]
        else:
            pre_weights = pd.Series(target_weights)

        drift = sum(
            abs(float(pre_weights.get(s, 0.0)) - target_weights.get(s, 0.0))
            for s in sleeve_cols
        )
        one_way_turnover = drift / 2.0

        rows.append({
            "portfolio_id": portfolio_id,
            "date": date,
            "one_way_turnover_pct": round(one_way_turnover * 100, 2),
            "cost_deducted": round(cost, 6),
            "cumulative_cost": round(cumulative_cost, 6),
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Turnover report [%s]: %d rebalancing events, avg one-way turnover %.1f%%",
        portfolio_id,
        len(df),
        df["one_way_turnover_pct"].mean() if len(df) > 0 else 0.0,
    )
    return df


def build_turnover_summary(
    turnover_reports: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate summary statistics across all portfolios.

    Args:
        turnover_reports: {portfolio_id: turnover_df from build_turnover_report()}.

    Returns:
        Summary DataFrame with one row per portfolio.
    """
    rows: List[Dict] = []
    for pid, df in turnover_reports.items():
        if df.empty:
            rows.append({
                "portfolio_id": pid,
                "n_rebalances": 0,
                "avg_one_way_turnover_pct": None,
                "total_cost_deducted": 0.0,
                "note": "buy_and_hold or no rebalancing events",
            })
        else:
            rows.append({
                "portfolio_id": pid,
                "n_rebalances": len(df),
                "avg_one_way_turnover_pct": round(float(df["one_way_turnover_pct"].mean()), 2),
                "total_cost_deducted": round(float(df["cumulative_cost"].iloc[-1]), 4),
                "note": "OK",
            })
    return pd.DataFrame(rows)
