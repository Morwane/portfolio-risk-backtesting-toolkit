"""Drawdown analytics.

Provides peak-to-trough drawdown series, maximum drawdown, and a
structured drawdown event table.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Compute the running drawdown series from daily simple returns.

    At each point, reports current loss from the most recent peak (0 = at peak).

    Args:
        daily_returns: Daily simple return Series.

    Returns:
        pd.Series of drawdown values (negative numbers, 0 at peaks).
    """
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1
    dd.name = "drawdown"
    return dd


def max_drawdown(daily_returns: pd.Series) -> float:
    """Return the maximum drawdown (most negative value in the drawdown series).

    Returns a negative float (e.g., -0.34 for a 34% drawdown).
    """
    return float(drawdown_series(daily_returns).min())


def drawdown_table(
    daily_returns: pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build a table of the worst distinct drawdown episodes.

    Each episode is defined as a peak-to-trough-to-recovery period.

    Args:
        daily_returns: Daily simple return Series.
        top_n: Number of episodes to return (ranked by depth).

    Returns:
        DataFrame with columns: peak_date, trough_date, recovery_date,
        drawdown_pct, duration_peak_to_trough_days,
        duration_trough_to_recovery_days.
    """
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1

    episodes = []
    in_drawdown = False
    peak_date = None
    peak_val = None

    for date, val in cum.items():
        if not in_drawdown:
            peak_val = running_max.loc[date]
            if val < peak_val * 0.9999:  # threshold to start episode
                in_drawdown = True
                peak_date = running_max.loc[:date].idxmax()
        else:
            # Check for recovery
            if val >= running_max.loc[date] * 0.9999:
                trough_date = dd.loc[peak_date:date].idxmin()
                trough_pct = dd.loc[trough_date]
                recovery_date = date
                pt_days = (trough_date - peak_date).days
                tr_days = (recovery_date - trough_date).days
                episodes.append({
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": recovery_date,
                    "drawdown_pct": round(trough_pct * 100, 2),
                    "duration_peak_to_trough_days": pt_days,
                    "duration_trough_to_recovery_days": tr_days,
                })
                in_drawdown = False
                peak_val = None

    # Handle ongoing drawdown at end of series
    if in_drawdown and peak_date is not None:
        last_date = cum.index[-1]
        trough_date = dd.loc[peak_date:].idxmin()
        trough_pct = dd.loc[trough_date]
        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": None,  # not yet recovered
            "drawdown_pct": round(trough_pct * 100, 2),
            "duration_peak_to_trough_days": (trough_date - peak_date).days,
            "duration_trough_to_recovery_days": None,
        })

    if not episodes:
        return pd.DataFrame(columns=[
            "peak_date", "trough_date", "recovery_date",
            "drawdown_pct", "duration_peak_to_trough_days",
            "duration_trough_to_recovery_days",
        ])

    df = pd.DataFrame(episodes)
    df = df.sort_values("drawdown_pct").head(top_n).reset_index(drop=True)
    return df


def underwater_equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Alias for drawdown_series — used for chart labelling."""
    return drawdown_series(daily_returns)
