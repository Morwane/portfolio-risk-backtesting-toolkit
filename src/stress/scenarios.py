"""Combined stress testing orchestrator.

Runs both historical windows and custom shocks in a single call,
returning the full stress test results table used in reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.stress.historical import run_historical_stress, run_historical_stress_multi
from src.stress.shocks import run_custom_shocks, run_custom_shocks_multi
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_full_stress_suite(
    portfolio_returns: pd.Series,
    weights: Dict[str, float],
    scenarios_path: Optional[Path] = None,
    portfolio_name: str = "portfolio",
) -> Dict[str, pd.DataFrame]:
    """Run the complete stress testing suite for one portfolio.

    Combines:
      1. Historical window analysis (actual returns during crisis periods)
      2. Custom shock scenarios (synthetic factor shocks)

    Args:
        portfolio_returns: Daily simple return Series.
        weights: Portfolio weights dict (for shock decomposition).
        scenarios_path: Override config path.
        portfolio_name: Label for logging.

    Returns:
        Dict with keys:
          - ``historical``: DataFrame from run_historical_stress
          - ``custom_shocks``: DataFrame from run_custom_shocks
    """
    logger.info("Running full stress suite for '%s' ...", portfolio_name)

    historical_df = run_historical_stress(portfolio_returns, scenarios_path=scenarios_path)
    shocks_df = run_custom_shocks(weights, scenarios_path=scenarios_path)

    logger.info(
        "Stress suite complete: %d historical scenarios, %d custom shocks.",
        len(historical_df),
        len(shocks_df),
    )

    return {
        "historical": historical_df,
        "custom_shocks": shocks_df,
    }


def build_stress_comparison_table(
    portfolio_nav_df: pd.DataFrame,
    portfolio_weights_dict: Dict[str, Dict[str, float]],
    scenarios_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Run stress testing across all portfolios for comparison charts.

    Args:
        portfolio_nav_df: DataFrame of NAV series (column per portfolio).
        portfolio_weights_dict: Dict of {portfolio_id: weights}.
        scenarios_path: Override config path.

    Returns:
        Dict with keys ``historical`` and ``custom_shocks``,
        each a long-form DataFrame with a ``portfolio_id`` column.
    """
    historical = run_historical_stress_multi(portfolio_nav_df, scenarios_path)
    shocks = run_custom_shocks_multi(portfolio_weights_dict, scenarios_path)

    return {
        "historical": historical,
        "custom_shocks": shocks,
    }
