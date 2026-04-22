"""Value at Risk and Expected Shortfall / CVaR computation.

Implements two VaR methodologies and historical ES as specified in the brief.
All metrics are expressed as positive loss values (losses are reported as
positive numbers to align with risk reporting conventions).

Methodology:
  Historical VaR:    Empirical quantile of the return distribution.
                     No distributional assumption. Preserves fat tails
                     and skewness. Requires sufficient history (>252 obs).

  Parametric VaR:    Gaussian VaR: µ - z_α * σ (annualised from daily).
                     Assumes normality — will underestimate tail risk
                     in practice. Reported alongside historical for comparison.

  Historical ES:     Mean of returns below the VaR threshold.
                     Also called CVaR or Conditional VaR.
                     More coherent than VaR for risk aggregation.

Returns are expressed as daily P&L fractions. Multiply by portfolio value
to convert to dollar amounts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(
    daily_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute historical VaR at *confidence* level.

    Args:
        daily_returns: Daily simple return series.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        VaR as a positive loss fraction (e.g. 0.02 = 2% one-day loss).
    """
    if len(daily_returns.dropna()) < 50:
        raise ValueError("Insufficient data for historical VaR (need ≥50 observations).")
    quantile = daily_returns.dropna().quantile(1 - confidence)
    return -quantile  # convert to positive loss


def parametric_var(
    daily_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute Gaussian (parametric) VaR at *confidence* level.

    Args:
        daily_returns: Daily simple return series.
        confidence: Confidence level.

    Returns:
        VaR as a positive loss fraction.
    """
    r = daily_returns.dropna()
    mu = r.mean()
    sigma = r.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return float(var)


def historical_es(
    daily_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute Historical Expected Shortfall (CVaR) at *confidence* level.

    ES = -E[R | R < VaR_threshold]

    Args:
        daily_returns: Daily simple return series.
        confidence: Confidence level (must match the VaR level).

    Returns:
        ES as a positive loss fraction.
    """
    r = daily_returns.dropna()
    threshold = r.quantile(1 - confidence)
    tail = r[r <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def compute_var_es_summary(
    daily_returns: pd.Series,
    confidence_levels: List[float] = None,
    name: str = "portfolio",
) -> pd.DataFrame:
    """Compute VaR and ES at multiple confidence levels.

    Args:
        daily_returns: Daily simple return Series.
        confidence_levels: List of confidence levels. Defaults to [0.95, 0.99].
        name: Label for the returned DataFrame index.

    Returns:
        DataFrame with rows for each metric × confidence level.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    rows = []
    for cl in confidence_levels:
        h_var = historical_var(daily_returns, cl)
        p_var = parametric_var(daily_returns, cl)
        h_es = historical_es(daily_returns, cl)
        rows.append({
            "asset": name,
            "confidence": f"{int(cl * 100)}%",
            "historical_var_1d": round(h_var, 6),
            "parametric_var_1d": round(p_var, 6),
            "historical_es_1d": round(h_es, 6),
            # Approximate 10-day VaR via √10 scaling (Basel convention, rough only)
            "historical_var_10d_approx": round(h_var * np.sqrt(10), 6),
        })

    return pd.DataFrame(rows)


def compute_all_var_es(
    returns_df: pd.DataFrame,
    confidence_levels: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute VaR / ES for every column in *returns_df*.

    Args:
        returns_df: DataFrame where each column is a daily return series.
        confidence_levels: Confidence levels to compute.

    Returns:
        Stacked DataFrame with results for all assets.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    frames = []
    for col in returns_df.columns:
        df = compute_var_es_summary(
            returns_df[col].dropna(), confidence_levels, name=col
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
