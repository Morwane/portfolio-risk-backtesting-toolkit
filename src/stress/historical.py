"""Historical stress scenario analysis.

Computes portfolio performance during predefined historical crisis windows.
Each window is defined by a start/end date in config/stress_scenarios.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.analytics.performance import compute_summary
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SCENARIOS_PATH = Path(__file__).parents[2] / "config" / "stress_scenarios.yaml"


def load_historical_windows(path: Optional[Path] = None) -> Dict:
    """Load historical stress window definitions from YAML."""
    p = path or _SCENARIOS_PATH
    with open(p, "r") as fh:
        config = yaml.safe_load(fh)
    return config.get("historical_windows", {})


def run_historical_stress(
    portfolio_returns: pd.Series,
    asset_returns: Optional[pd.DataFrame] = None,
    scenarios_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute portfolio return statistics for each historical stress window.

    Args:
        portfolio_returns: Daily portfolio return Series (DatetimeIndex).
        asset_returns: Optional per-sleeve returns for decomposition.
        scenarios_path: Override path to stress_scenarios.yaml.

    Returns:
        DataFrame with one row per scenario and columns:
        scenario_name, start, end, portfolio_total_return,
        portfolio_annualised_vol, max_drawdown_in_window,
        n_trading_days, status.
    """
    windows = load_historical_windows(scenarios_path)
    rows = []

    for scenario_id, spec in windows.items():
        start = spec["start"]
        end = spec["end"]
        name = spec["name"]

        # Slice returns to the window
        window_returns = portfolio_returns.loc[start:end]

        if window_returns.empty:
            logger.warning(
                "Scenario '%s' (%s to %s): no data in portfolio history. Skipping.",
                name, start, end,
            )
            rows.append({
                "scenario_id": scenario_id,
                "scenario_name": name,
                "start": start,
                "end": end,
                "portfolio_total_return": None,
                "portfolio_annualised_vol": None,
                "max_drawdown_in_window": None,
                "n_trading_days": 0,
                "status": "no_data",
            })
            continue

        total_ret = (1 + window_returns).prod() - 1
        ann_vol = window_returns.std() * (252 ** 0.5) if len(window_returns) > 1 else float("nan")

        from src.analytics.drawdown import max_drawdown
        mdd = max_drawdown(window_returns)

        actual_start = window_returns.index[0].date()
        actual_end = window_returns.index[-1].date()

        rows.append({
            "scenario_id": scenario_id,
            "scenario_name": name,
            "start": str(actual_start),
            "end": str(actual_end),
            "portfolio_total_return": round(float(total_ret), 6),
            "portfolio_annualised_vol": round(float(ann_vol), 6),
            "max_drawdown_in_window": round(float(mdd), 6),
            "n_trading_days": len(window_returns),
            "status": "ok",
        })

        logger.info(
            "Scenario '%s': return=%.2f%%  vol=%.2f%%  mdd=%.2f%%",
            name,
            total_ret * 100,
            ann_vol * 100,
            mdd * 100,
        )

    df = pd.DataFrame(rows)
    return df


def run_historical_stress_multi(
    portfolio_nav_df: pd.DataFrame,
    scenarios_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run historical stress analysis across multiple portfolios.

    Args:
        portfolio_nav_df: DataFrame where each column is a portfolio's NAV series.
        scenarios_path: Override path.

    Returns:
        Long-form DataFrame with columns: scenario_name, portfolio_id,
        portfolio_total_return, max_drawdown_in_window.
    """
    rows = []
    for portfolio_id in portfolio_nav_df.columns:
        nav = portfolio_nav_df[portfolio_id].dropna()
        port_returns = nav.pct_change().dropna()
        port_returns.name = portfolio_id

        stress_df = run_historical_stress(port_returns, scenarios_path=scenarios_path)
        stress_df["portfolio_id"] = portfolio_id
        rows.append(stress_df)

    return pd.concat(rows, ignore_index=True)
