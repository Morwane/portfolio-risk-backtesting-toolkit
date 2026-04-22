"""High-level risk analytics aggregator.

Bundles all risk metrics into a single callable function for use by
the backtest runner and reporting layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.analytics.drawdown import drawdown_series, drawdown_table, max_drawdown
from src.analytics.performance import compute_summary
from src.analytics.rolling import compute_rolling_metrics_df
from src.analytics.var_es import compute_var_es_summary


def compute_risk_report(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_annual: float = 0.045,
    var_confidence_levels: Optional[List[float]] = None,
    rolling_window: int = 252,
) -> Dict:
    """Compute the complete risk report for one portfolio.

    Args:
        portfolio_returns: Daily simple return Series.
        benchmark_returns: Optional benchmark for beta computation.
        risk_free_annual: Annual risk-free rate.
        var_confidence_levels: VaR/ES confidence levels.
        rolling_window: Rolling metrics window (trading days).

    Returns:
        Dict with keys:
          - ``summary``: dict of KPIs
          - ``var_es``: DataFrame from compute_var_es_summary
          - ``drawdown_series``: Series
          - ``drawdown_table``: DataFrame of top drawdown episodes
          - ``rolling_metrics``: DataFrame of rolling vol/Sharpe/beta
    """
    if var_confidence_levels is None:
        var_confidence_levels = [0.95, 0.99]

    summary = compute_summary(portfolio_returns, risk_free_annual)
    var_es = compute_var_es_summary(
        portfolio_returns,
        confidence_levels=var_confidence_levels,
        name=portfolio_returns.name or "portfolio",
    )
    dd_series = drawdown_series(portfolio_returns)
    dd_table = drawdown_table(portfolio_returns, top_n=10)
    rolling = compute_rolling_metrics_df(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        window=rolling_window,
        risk_free_annual=risk_free_annual,
    )

    return {
        "summary": summary,
        "var_es": var_es,
        "drawdown_series": dd_series,
        "drawdown_table": dd_table,
        "rolling_metrics": rolling,
    }
