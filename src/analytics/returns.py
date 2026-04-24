"""Return computation utilities.

Converts prices to returns and handles aggregation across frequencies.
Separate from cleaner.py which handles data engineering concerns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return log returns from price DataFrame."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return simple (arithmetic) returns from price DataFrame."""
    return prices.pct_change().iloc[1:]


def compound_returns(daily_returns: pd.Series, periods: int) -> float:
    """Compound *periods* consecutive daily returns."""
    return (1 + daily_returns).prod() - 1


def annualise_return(total_return: float, n_days: int) -> float:
    """Annualise a total return over n_days of calendar history.

    Uses geometric annualisation: (1 + R)^(252/n) - 1.
    """
    if n_days <= 0:
        return float("nan")
    return (1 + total_return) ** (TRADING_DAYS / n_days) - 1


def annualise_volatility(daily_returns: pd.Series, ddof: int = 1) -> float:
    """Annualise daily return volatility."""
    return daily_returns.std(ddof=ddof) * np.sqrt(TRADING_DAYS)


def monthly_returns_table(daily_returns: pd.Series) -> pd.DataFrame:
    """Build a months × years pivot table of monthly compounded returns.

    Args:
        daily_returns: Daily simple return Series.

    Returns:
        DataFrame with year columns and month rows (Jan=1 … Dec=12).
    """
    monthly = (1 + daily_returns).resample("M").prod() - 1
    monthly.index = pd.to_datetime(monthly.index)
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack(level=0)
    table.index.name = "month"
    table.columns.name = "year"
    return table


def cumulative_return_series(daily_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Build a cumulative return index from a simple returns series.

    Args:
        daily_returns: Daily simple return Series.
        base: Starting index value.

    Returns:
        Cumulative return index Series.
    """
    cumret = base * (1 + daily_returns).cumprod()
    cumret.name = daily_returns.name or "cumulative_return"
    return cumret
