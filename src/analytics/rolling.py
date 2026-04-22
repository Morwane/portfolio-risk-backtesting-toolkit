"""Rolling risk and performance metrics.

Produces time-series of metrics computed over rolling windows, used for
the rolling charts and monitoring dashboards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_volatility(
    daily_returns: pd.Series,
    window: int = TRADING_DAYS,
    min_periods: int = 63,
) -> pd.Series:
    """Compute annualised rolling volatility.

    Args:
        daily_returns: Daily simple return Series.
        window: Rolling window in trading days.
        min_periods: Minimum observations required (default ~1 quarter).

    Returns:
        Annualised rolling volatility Series.
    """
    vol = (
        daily_returns
        .rolling(window=window, min_periods=min_periods)
        .std()
        * np.sqrt(TRADING_DAYS)
    )
    vol.name = "rolling_volatility"
    return vol


def rolling_sharpe(
    daily_returns: pd.Series,
    window: int = TRADING_DAYS,
    risk_free_annual: float = 0.045,
    min_periods: int = 63,
) -> pd.Series:
    """Compute rolling annualised Sharpe ratio.

    Args:
        daily_returns: Daily simple return Series.
        window: Rolling window in trading days.
        risk_free_annual: Annual risk-free rate.
        min_periods: Minimum observations required.

    Returns:
        Rolling Sharpe ratio Series.
    """
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = daily_returns - rf_daily

    roll_mean = excess.rolling(window=window, min_periods=min_periods).mean()
    roll_std = excess.rolling(window=window, min_periods=min_periods).std()

    sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)
    sharpe.name = "rolling_sharpe"
    return sharpe


def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = TRADING_DAYS,
    min_periods: int = 63,
) -> pd.Series:
    """Compute rolling beta of *asset_returns* vs *benchmark_returns*.

    Args:
        asset_returns: Daily return Series for the asset/portfolio.
        benchmark_returns: Daily return Series for the benchmark.
        window: Rolling window in trading days.
        min_periods: Minimum observations required.

    Returns:
        Rolling beta Series.
    """
    def _beta(df: pd.DataFrame) -> float:
        if df.shape[0] < 2:
            return float("nan")
        cov = df.cov()
        var_bench = cov.iloc[1, 1]
        if var_bench == 0:
            return float("nan")
        return cov.iloc[0, 1] / var_bench

    combined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    combined.columns = ["asset", "benchmark"]

    beta = combined.rolling(window=window, min_periods=min_periods).apply(
        lambda x: x  # placeholder — use covariance approach below
    )

    # Use rolling cov / rolling var approach (more stable)
    roll_cov = combined["asset"].rolling(window, min_periods=min_periods).cov(combined["benchmark"])
    roll_var = combined["benchmark"].rolling(window, min_periods=min_periods).var()
    beta = roll_cov / roll_var
    beta.name = "rolling_beta"
    return beta


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = TRADING_DAYS,
    min_periods: int = 63,
) -> pd.Series:
    """Compute rolling correlation between two return series."""
    corr = series_a.rolling(window, min_periods=min_periods).corr(series_b)
    corr.name = f"rolling_corr_{series_a.name}_{series_b.name}"
    return corr


def compute_rolling_metrics_df(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    window: int = TRADING_DAYS,
    risk_free_annual: float = 0.045,
) -> pd.DataFrame:
    """Compute all rolling metrics and return as a single DataFrame.

    Args:
        portfolio_returns: Daily portfolio return Series.
        benchmark_returns: Optional benchmark for beta/correlation.
        window: Rolling window in trading days.
        risk_free_annual: Annual risk-free rate.

    Returns:
        DataFrame with columns: rolling_volatility, rolling_sharpe,
        and optionally rolling_beta, rolling_correlation.
    """
    result = pd.DataFrame(index=portfolio_returns.index)
    result["rolling_volatility"] = rolling_volatility(portfolio_returns, window)
    result["rolling_sharpe"] = rolling_sharpe(portfolio_returns, window, risk_free_annual)

    if benchmark_returns is not None:
        result["rolling_beta"] = rolling_beta(portfolio_returns, benchmark_returns, window)
        result["rolling_correlation"] = rolling_correlation(portfolio_returns, benchmark_returns, window)

    return result
