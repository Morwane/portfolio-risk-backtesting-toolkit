"""Portfolio construction engine.

Builds a portfolio NAV series from asset returns and target weights,
handling periodic rebalancing and optional transaction costs.

Key design:
  - All returns are simple (arithmetic) returns.
  - Between rebalancing dates, each asset's weight drifts with its price.
  - At each rebalancing date, weights are reset to target.
  - Transaction costs are deducted from the portfolio at rebalancing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.portfolio.rebalancing import get_rebalancing_dates, compute_rebalancing_costs
from src.utils.logging_utils import get_logger
from src.utils.validation import filter_available_sleeves, validate_weights

logger = get_logger(__name__)


def build_portfolio(
    returns: pd.DataFrame,
    target_weights: Dict[str, float],
    rebalancing_frequency: str = "quarterly",
    initial_value: float = 100.0,
    cost_bps: float = 2.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Backtest a portfolio with periodic rebalancing.

    Args:
        returns: Daily simple returns DataFrame (dates × sleeves).
        target_weights: Dict of sleeve_id -> weight (must sum to ~1.0).
        rebalancing_frequency: ``"monthly"``, ``"quarterly"``, ``"annual"``,
                               or ``"buy_and_hold"``.
        initial_value: Starting portfolio value (default 100 for indexing).
        cost_bps: One-way transaction cost per rebalancing.

    Returns:
        Tuple of:
          - ``nav``: pd.Series of daily portfolio values.
          - ``weights_history``: pd.DataFrame of daily portfolio weights.
    """
    # Filter to available sleeves
    available = list(returns.columns)
    weights = filter_available_sleeves(target_weights, available, renormalise=True)
    validate_weights(weights)

    rebal_dates = get_rebalancing_dates(returns.index, rebalancing_frequency)

    # Initialise
    nav_values = [initial_value]
    current_value = initial_value
    current_weights = dict(weights)  # copy
    weight_rows: List[Dict] = []

    for i, date in enumerate(returns.index):
        if i == 0:
            weight_rows.append({**current_weights, "date": date})
            continue

        # Daily return for each sleeve
        day_returns = returns.loc[date]

        # Update current weights based on price moves (drift)
        new_values = {
            s: current_weights.get(s, 0.0) * current_value * (1 + day_returns.get(s, 0.0))
            for s in available
        }
        new_portfolio_value = sum(new_values.values())
        if new_portfolio_value <= 0:
            logger.warning("Portfolio value became non-positive on %s. Stopping.", date)
            break

        drifted_weights = {s: v / new_portfolio_value for s, v in new_values.items()}

        # Rebalance if this is a rebalancing date (skip the very first)
        if date in rebal_dates and i > 0:
            cost = compute_rebalancing_costs(
                drifted_weights, weights, new_portfolio_value, cost_bps
            )
            new_portfolio_value -= cost
            current_weights = dict(weights)
            logger.debug(
                "Rebalanced on %s. Cost: %.4f. Portfolio value: %.4f",
                date.date(), cost, new_portfolio_value,
            )
        else:
            current_weights = drifted_weights

        current_value = new_portfolio_value
        nav_values.append(current_value)
        weight_rows.append({**current_weights, "date": date})

    nav = pd.Series(
        nav_values,
        index=returns.index[:len(nav_values)],
        name="portfolio_nav",
    )
    nav.index.name = "date"

    weights_history = pd.DataFrame(weight_rows).set_index("date")

    return nav, weights_history


def build_multiple_portfolios(
    returns: pd.DataFrame,
    portfolio_configs: Dict[str, Dict],
    rebalancing_frequency: str = "quarterly",
    initial_value: float = 100.0,
    cost_bps: float = 2.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Build several portfolios in one call for comparison.

    Args:
        returns: Daily returns DataFrame.
        portfolio_configs: Dict of {portfolio_id: {"weights": {...}, "name": str}}.
        rebalancing_frequency: Shared rebalancing frequency.
        initial_value: Starting value for each portfolio.
        cost_bps: Transaction cost.

    Returns:
        Tuple of:
          - ``nav_df``: DataFrame with one column per portfolio.
          - ``weights_dict``: Dict of {portfolio_id: weights_history DataFrame}.
    """
    nav_dict: Dict[str, pd.Series] = {}
    weights_dict: Dict[str, pd.DataFrame] = {}

    for pid, config in portfolio_configs.items():
        weights = config.get("weights", {})
        if not weights:
            logger.warning("Portfolio '%s' has empty weights. Skipping.", pid)
            continue
        logger.info("Building portfolio: %s", config.get("name", pid))
        nav, wh = build_portfolio(
            returns, weights, rebalancing_frequency, initial_value, cost_bps
        )
        nav.name = config.get("name", pid)
        nav_dict[pid] = nav
        weights_dict[pid] = wh

    nav_df = pd.DataFrame(nav_dict)
    nav_df.index.name = "date"
    return nav_df, weights_dict


def compute_portfolio_returns(nav: pd.Series) -> pd.Series:
    """Compute daily simple returns from a NAV series."""
    r = nav.pct_change().dropna()
    r.name = "portfolio_return"
    return r
