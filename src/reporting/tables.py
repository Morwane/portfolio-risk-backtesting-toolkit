"""Table builders for all structured outputs.

Each function returns a clean, presentation-ready DataFrame that can be
passed directly to export.py or printed to a notebook.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.analytics.performance import compute_all_summaries
from src.analytics.var_es import compute_all_var_es
from src.data.mapping import get_asset_class_map, get_name_map


def fmt_pct(x: float, decimals: int = 2) -> str:
    """Format a fraction as a percentage string."""
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def fmt_ratio(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


# ── Portfolio Summary ─────────────────────────────────────────────────────────

def build_portfolio_summary(
    returns_dict: Dict[str, pd.Series],
    risk_free_annual: float = 0.045,
) -> pd.DataFrame:
    """Build a cross-portfolio performance summary table.

    Args:
        returns_dict: Dict of {portfolio_name: daily_returns_series}.
        risk_free_annual: Annual risk-free rate.

    Returns:
        DataFrame with one row per portfolio and metric columns.
    """
    df = pd.DataFrame(returns_dict)
    summary = compute_all_summaries(df, risk_free_annual)

    # Format for presentation
    display = pd.DataFrame()
    display["Annualised Return"] = summary["annualised_return"].map(fmt_pct)
    display["Annualised Volatility"] = summary["annualised_volatility"].map(fmt_pct)
    display["Sharpe Ratio"] = summary["sharpe_ratio"].map(fmt_ratio)
    display["Sortino Ratio"] = summary["sortino_ratio"].map(fmt_ratio)
    display["Calmar Ratio"] = summary["calmar_ratio"].map(fmt_ratio)
    display["Max Drawdown"] = summary["max_drawdown"].map(fmt_pct)
    display["Best Day"] = summary["best_day"].map(lambda x: fmt_pct(x, 2))
    display["Worst Day"] = summary["worst_day"].map(lambda x: fmt_pct(x, 2))
    display["% Positive Days"] = summary["positive_days_pct"].map(fmt_pct)

    return display


# ── Monthly Returns Heatmap ───────────────────────────────────────────────────

def build_monthly_returns_table(
    daily_returns: pd.Series,
    portfolio_name: str = "Portfolio",
) -> pd.DataFrame:
    """Build a calendar-style monthly returns table (months × years).

    Args:
        daily_returns: Daily simple return Series.
        portfolio_name: Used for logging only.

    Returns:
        DataFrame with months (1–12) as rows and years as columns.
        Values are decimal fractions (multiply by 100 for percentages).
    """
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    monthly.index = pd.to_datetime(monthly.index)

    table = monthly.groupby([monthly.index.year, monthly.index.month]).first()
    table = table.unstack(level=0)
    table.index.name = "month"
    table.columns.name = "year"
    table.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(table)]

    return table


# ── Drawdown Table ────────────────────────────────────────────────────────────

def build_drawdown_table(
    daily_returns: pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-N drawdown episodes in presentation format."""
    from src.analytics.drawdown import drawdown_table

    df = drawdown_table(daily_returns, top_n=top_n)
    if df.empty:
        return df

    df = df.rename(columns={
        "peak_date": "Peak Date",
        "trough_date": "Trough Date",
        "recovery_date": "Recovery Date",
        "drawdown_pct": "Drawdown (%)",
        "duration_peak_to_trough_days": "Days to Trough",
        "duration_trough_to_recovery_days": "Recovery Days",
    })
    return df


# ── VaR / ES Summary ─────────────────────────────────────────────────────────

def build_var_es_table(
    returns_dict: Dict[str, pd.Series],
    confidence_levels: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Build a VaR / ES summary table across all portfolios/assets.

    Args:
        returns_dict: Dict of {label: daily_returns_series}.
        confidence_levels: e.g. [0.95, 0.99].

    Returns:
        Wide-format DataFrame suitable for display.
    """
    from src.analytics.var_es import compute_var_es_summary

    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    rows = []
    for name, series in returns_dict.items():
        df = compute_var_es_summary(series.dropna(), confidence_levels, name=name)
        rows.append(df)

    result = pd.concat(rows, ignore_index=True)
    # Convert to percentages for display
    for col in ["historical_var_1d", "parametric_var_1d", "historical_es_1d",
                "historical_var_10d_approx"]:
        result[col] = result[col].map(lambda x: fmt_pct(x, 3))

    result = result.rename(columns={
        "asset": "Portfolio / Asset",
        "confidence": "Confidence",
        "historical_var_1d": "Hist. VaR (1D)",
        "parametric_var_1d": "Param. VaR (1D)",
        "historical_es_1d": "Hist. ES (1D)",
        "historical_var_10d_approx": "Hist. VaR (10D, √10)",
    })
    return result


# ── Stress Test Results ───────────────────────────────────────────────────────

def build_stress_test_table(
    historical_df: pd.DataFrame,
    custom_shocks_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Format stress test DataFrames for presentation.

    Args:
        historical_df: Output from run_historical_stress().
        custom_shocks_df: Output from run_custom_shocks().

    Returns:
        Dict with ``historical`` and ``custom_shocks`` display DataFrames.
    """
    hist = historical_df.copy()
    if "portfolio_total_return" in hist.columns:
        hist["portfolio_total_return"] = hist["portfolio_total_return"].map(
            lambda x: fmt_pct(x) if pd.notna(x) else "—"
        )
    if "max_drawdown_in_window" in hist.columns:
        hist["max_drawdown_in_window"] = hist["max_drawdown_in_window"].map(
            lambda x: fmt_pct(x) if pd.notna(x) else "—"
        )
    if "portfolio_annualised_vol" in hist.columns:
        hist["portfolio_annualised_vol"] = hist["portfolio_annualised_vol"].map(
            lambda x: fmt_pct(x) if pd.notna(x) else "—"
        )

    hist = hist.rename(columns={
        "scenario_name": "Scenario",
        "start": "Start",
        "end": "End",
        "portfolio_total_return": "Portfolio Return",
        "portfolio_annualised_vol": "Ann. Vol",
        "max_drawdown_in_window": "Max DD",
        "n_trading_days": "Trading Days",
    })

    shocks = custom_shocks_df[["scenario_name", "description", "total_portfolio_impact"]].copy()
    shocks["total_portfolio_impact"] = shocks["total_portfolio_impact"].map(fmt_pct)
    shocks = shocks.rename(columns={
        "scenario_name": "Scenario",
        "description": "Description",
        "total_portfolio_impact": "Est. Portfolio Impact",
    })

    return {"historical": hist, "custom_shocks": shocks}


# ── Asset Class Contribution Table ───────────────────────────────────────────

def build_contribution_table(
    contribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """Format the contribution table from analytics.contributions."""
    df = contribution_df.copy()
    if "weight" in df.columns:
        df["weight"] = df["weight"].map(lambda x: fmt_pct(x, 1))
    if "return_contribution" in df.columns:
        df["return_contribution"] = df["return_contribution"].map(fmt_pct)
    if "risk_contribution_pct" in df.columns:
        df["risk_contribution_pct"] = df["risk_contribution_pct"].map(lambda x: fmt_pct(x, 1))

    df = df.rename(columns={
        "name": "Asset",
        "asset_class": "Class",
        "weight": "Weight",
        "return_contribution": "Return Contrib.",
        "risk_contribution_pct": "Risk Contrib. %",
        "marginal_risk_contribution": "Marginal Risk",
    })
    return df
