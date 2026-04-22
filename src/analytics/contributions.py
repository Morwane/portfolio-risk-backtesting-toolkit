"""Asset-class return and risk contribution analytics.

Computes:
  - Return contribution: weight × asset return (Brinson-style)
  - Volatility contribution: marginal risk contribution (percentage of risk)
  - Risk budget: contribution to total portfolio variance
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.data.mapping import get_asset_class_map, get_name_map


def return_contribution(
    weights: Dict[str, float],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute each asset's contribution to portfolio return over time.

    Contribution_i(t) = w_i(t) × r_i(t)

    This is the Brinson attribution framework simplified to individual assets.

    Args:
        weights: Dict of sleeve_id -> weight (assumed static here).
        asset_returns: Daily simple returns DataFrame.

    Returns:
        DataFrame of daily return contributions, same shape as asset_returns
        (subset to sleeves in weights).
    """
    avail = [s for s in weights if s in asset_returns.columns]
    w = pd.Series({s: weights[s] for s in avail})
    contrib = asset_returns[avail].multiply(w, axis=1)
    contrib.index.name = "date"
    return contrib


def cumulative_return_contribution(
    weights: Dict[str, float],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Cumulative sum of daily return contributions per sleeve."""
    daily = return_contribution(weights, asset_returns)
    return daily.cumsum()


def marginal_risk_contribution(
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute each asset's marginal contribution to portfolio volatility.

    MRC_i = (Σw)_i / σ_p  where Σ is the covariance matrix.

    Args:
        weights: Dict of sleeve_id -> weight.
        cov_matrix: Annualised covariance matrix (sleeves × sleeves).

    Returns:
        Series of marginal risk contributions (sum = portfolio volatility).
    """
    avail = [s for s in weights if s in cov_matrix.columns]
    w = np.array([weights[s] for s in avail])
    sigma = cov_matrix.loc[avail, avail].values

    portfolio_var = float(w @ sigma @ w)
    portfolio_vol = np.sqrt(portfolio_var)
    if portfolio_vol == 0:
        return pd.Series(0.0, index=avail)

    mrc = (sigma @ w) / portfolio_vol
    return pd.Series(mrc, index=avail, name="marginal_risk_contribution")


def risk_contribution_pct(
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute each asset's percentage contribution to total portfolio variance.

    RiskContrib_i = w_i * (Σw)_i / σ_p²
    Sum = 1 by construction.

    Args:
        weights: Dict of sleeve_id -> weight.
        cov_matrix: Annualised covariance matrix.

    Returns:
        Series of risk contribution percentages (sum ~1.0).
    """
    avail = [s for s in weights if s in cov_matrix.columns]
    w = np.array([weights[s] for s in avail])
    sigma = cov_matrix.loc[avail, avail].values

    portfolio_var = float(w @ sigma @ w)
    if portfolio_var == 0:
        return pd.Series(0.0, index=avail)

    individual = w * (sigma @ w)
    pct = individual / portfolio_var
    return pd.Series(pct, index=avail, name="risk_contribution_pct")


def asset_class_summary(
    contributions: pd.Series,
    asset_class_map: Dict[str, str] = None,
) -> pd.Series:
    """Aggregate contributions by asset class.

    Args:
        contributions: Per-sleeve contribution Series.
        asset_class_map: Dict of sleeve_id -> asset_class label.
                         If None, fetches from universe config.

    Returns:
        Series of summed contributions keyed by asset class.
    """
    if asset_class_map is None:
        asset_class_map = get_asset_class_map()

    mapped = contributions.copy()
    mapped.index = [asset_class_map.get(s, s) for s in contributions.index]
    return mapped.groupby(level=0).sum().rename("asset_class_contribution")


def build_contribution_table(
    weights: Dict[str, float],
    asset_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build a unified contribution table with return and risk contributions.

    Args:
        weights: Dict of sleeve_id -> weight.
        asset_returns: Daily returns DataFrame.
        cov_matrix: Annualised covariance matrix. If None, computed from returns.

    Returns:
        DataFrame indexed by sleeve_id with columns:
        weight, cumulative_return_contribution,
        risk_contribution_pct (if cov_matrix available).
    """
    avail = [s for s in weights if s in asset_returns.columns]
    w = {s: weights[s] for s in avail}

    # Cumulative return contribution
    cum_contrib = return_contribution(w, asset_returns).sum()

    # Risk contribution
    if cov_matrix is None:
        cov_matrix = asset_returns[avail].cov() * 252  # annualise

    risk_pct = risk_contribution_pct(w, cov_matrix)
    mrc = marginal_risk_contribution(w, cov_matrix)

    name_map = get_name_map()
    ac_map = get_asset_class_map()

    rows = []
    for s in avail:
        rows.append({
            "sleeve_id": s,
            "name": name_map.get(s, s),
            "asset_class": ac_map.get(s, ""),
            "weight": round(w[s], 4),
            "return_contribution": round(float(cum_contrib.get(s, 0.0)), 4),
            "risk_contribution_pct": round(float(risk_pct.get(s, 0.0)), 4),
            "marginal_risk_contribution": round(float(mrc.get(s, 0.0)), 4),
        })

    return pd.DataFrame(rows).set_index("sleeve_id")
