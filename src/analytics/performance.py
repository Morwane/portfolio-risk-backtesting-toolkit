"""Summary performance metrics for a return series.

Computes the full suite of KPIs used in the portfolio summary table.
All metrics use daily simple returns as input unless stated otherwise.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analytics.returns import TRADING_DAYS, annualise_return, annualise_volatility

_RISK_FREE_RATE = 0.045  # annual; override via function arguments


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_annual: float = _RISK_FREE_RATE,
) -> float:
    """Annualised Sharpe ratio using daily returns."""
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    if excess.std() == 0:
        return float("nan")
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_annual: float = _RISK_FREE_RATE,
) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    downside_std = downside.std() * np.sqrt(TRADING_DAYS)
    return (excess.mean() * TRADING_DAYS) / downside_std


def calmar_ratio(
    daily_returns: pd.Series,
    max_dd: Optional[float] = None,
) -> float:
    """Calmar ratio: annualised return / |max drawdown|."""
    from src.analytics.drawdown import max_drawdown as _max_dd
    ann_ret = annualise_return(
        (1 + daily_returns).prod() - 1,
        len(daily_returns),
    )
    dd = max_dd if max_dd is not None else _max_dd(daily_returns)
    if dd >= 0:
        return float("nan")
    return ann_ret / abs(dd)


def compute_summary(
    daily_returns: pd.Series,
    risk_free_annual: float = _RISK_FREE_RATE,
    name: Optional[str] = None,
) -> Dict[str, float]:
    """Compute the full performance summary for one return series.

    Args:
        daily_returns: Daily simple return Series.
        risk_free_annual: Annual risk-free rate.
        name: Optional label for logging.

    Returns:
        Dict with keys: total_return, annualised_return, annualised_volatility,
        sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
        best_day, worst_day, positive_days_pct.
    """
    from src.analytics.drawdown import max_drawdown

    n = len(daily_returns)
    total_ret = (1 + daily_returns).prod() - 1
    ann_ret = annualise_return(total_ret, n)
    ann_vol = annualise_volatility(daily_returns)
    sr = sharpe_ratio(daily_returns, risk_free_annual)
    so = sortino_ratio(daily_returns, risk_free_annual)
    mdd = max_drawdown(daily_returns)
    cal = ann_ret / abs(mdd) if mdd < 0 else float("nan")

    return {
        "total_return": round(total_ret, 6),
        "annualised_return": round(ann_ret, 6),
        "annualised_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(so, 4),
        "calmar_ratio": round(cal, 4),
        "max_drawdown": round(mdd, 6),
        "best_day": round(daily_returns.max(), 6),
        "worst_day": round(daily_returns.min(), 6),
        "positive_days_pct": round((daily_returns > 0).mean(), 4),
        "n_days": n,
    }


def compute_all_summaries(
    returns_df: pd.DataFrame,
    risk_free_annual: float = _RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compute performance summaries for all columns in *returns_df*.

    Args:
        returns_df: DataFrame where each column is a return series.
        risk_free_annual: Annual risk-free rate.

    Returns:
        DataFrame with one row per column (metric columns).
    """
    rows = {}
    for col in returns_df.columns:
        rows[col] = compute_summary(returns_df[col].dropna(), risk_free_annual, name=col)
    return pd.DataFrame(rows).T
