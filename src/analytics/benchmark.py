"""Benchmark-relative analytics.

Computes tracking error, information ratio, and benchmark-relative return series
for a portfolio against a reference benchmark NAV.

Design constraints:
  - Benchmarks are defined as other portfolios in the same backtest run
    (e.g. balanced_60_40 serves as the reference for strategic_diversified).
  - No external benchmark index is fetched: this avoids entitlement complexity
    and keeps the analysis self-contained.
  - Where no benchmark is defined for a portfolio, benchmark-relative metrics
    are omitted with an explicit note.
  - FX decomposition is NOT performed: all returns are in USD base currency
    with FX risk embedded in non-USD sleeve prices.

Outputs:
  outputs/tables/benchmark_relative_report.csv
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default benchmark mapping: {portfolio_id: benchmark_portfolio_id}
# Can be overridden via settings.yaml or function argument.
DEFAULT_BENCHMARK_MAP: Dict[str, str] = {
    "strategic_diversified": "balanced_60_40",
    "defensive": "balanced_60_40",
    "equal_weight": "balanced_60_40",
    # balanced_60_40 is its own baseline — no benchmark assigned by default
}


def compute_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualise: bool = True,
    trading_days: int = 252,
) -> float:
    """Compute annualised tracking error (std of active return).

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        annualise: If True, annualise by √trading_days.
        trading_days: Trading days per year.

    Returns:
        Tracking error as a decimal (e.g. 0.035 for 3.5%).
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std()
    if annualise:
        te *= np.sqrt(trading_days)
    return float(te)


def compute_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualise: bool = True,
    trading_days: int = 252,
) -> float:
    """Compute Information Ratio = annualised active return / tracking error.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        annualise: Annualise both return and TE.
        trading_days: Trading days per year.

    Returns:
        Information ratio (dimensionless). NaN if TE is zero.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")

    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    active_ann = active.mean() * trading_days
    te = active.std() * np.sqrt(trading_days)
    if te < 1e-10:
        return float("nan")
    return float(active_ann / te)


def compute_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_periods: int = 63,
) -> float:
    """Compute OLS beta of portfolio vs benchmark.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        min_periods: Minimum observations for computation.

    Returns:
        Beta (float). NaN if insufficient data.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < min_periods:
        return float("nan")
    cov_matrix = aligned.cov()
    bench_var = float(cov_matrix.iloc[1, 1])
    if bench_var < 1e-12:
        return float("nan")
    return float(cov_matrix.iloc[0, 1] / bench_var)


def build_benchmark_report(
    port_returns_dict: Dict[str, pd.Series],
    benchmark_map: Optional[Dict[str, str]] = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Build benchmark-relative metrics for all mapped portfolios.

    Args:
        port_returns_dict: {portfolio_id: daily_returns_series}.
        benchmark_map: {portfolio_id: benchmark_portfolio_id}. Uses DEFAULT_BENCHMARK_MAP if None.
        trading_days: Trading days per year.

    Returns:
        DataFrame with one row per portfolio-benchmark pair containing:
        tracking_error, information_ratio, beta, active_return_ann.
    """
    mapping = benchmark_map or DEFAULT_BENCHMARK_MAP
    rows: List[Dict] = []

    for pid, port_ret in port_returns_dict.items():
        bm_id = mapping.get(pid)
        if bm_id is None or bm_id not in port_returns_dict:
            rows.append({
                "portfolio_id": pid,
                "benchmark_id": bm_id or "none",
                "tracking_error_ann": None,
                "information_ratio": None,
                "beta_vs_benchmark": None,
                "active_return_ann": None,
                "note": "No benchmark assigned or benchmark not in run.",
            })
            continue

        bm_ret = port_returns_dict[bm_id]
        aligned = pd.concat([port_ret, bm_ret], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        active = aligned["portfolio"] - aligned["benchmark"]
        active_ann = float(active.mean() * trading_days)
        te = compute_tracking_error(aligned["portfolio"], aligned["benchmark"], trading_days=trading_days)
        ir = compute_information_ratio(aligned["portfolio"], aligned["benchmark"], trading_days=trading_days)
        beta = compute_beta(aligned["portfolio"], aligned["benchmark"])

        rows.append({
            "portfolio_id": pid,
            "benchmark_id": bm_id,
            "tracking_error_ann": round(te, 4) if not np.isnan(te) else None,
            "information_ratio": round(ir, 4) if not np.isnan(ir) else None,
            "beta_vs_benchmark": round(beta, 4) if not np.isnan(beta) else None,
            "active_return_ann": round(active_ann, 4),
            "note": "OK",
        })

        logger.info(
            "Benchmark-relative [%s vs %s]: TE=%.2f%%  IR=%.2f  beta=%.2f",
            pid, bm_id,
            te * 100 if not np.isnan(te) else float("nan"),
            ir if not np.isnan(ir) else float("nan"),
            beta if not np.isnan(beta) else float("nan"),
        )

    return pd.DataFrame(rows)
