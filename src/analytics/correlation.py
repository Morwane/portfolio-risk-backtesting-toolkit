"""Correlation and covariance analytics.

Computes full and crisis-period correlation matrices across sleeves,
with helpers for concentration / diversification metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 63,
) -> pd.DataFrame:
    """Compute pairwise return correlations.

    Args:
        returns: Daily returns DataFrame (dates × sleeves).
        method: ``"pearson"`` or ``"spearman"``.
        min_periods: Minimum overlapping observations per pair.

    Returns:
        Correlation matrix as a square DataFrame.
    """
    corr = returns.corr(method=method, min_periods=min_periods)
    corr.index.name = "sleeve"
    corr.columns.name = "sleeve"
    return corr


def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    return s1.rolling(window, min_periods=min_periods).corr(s2)


def effective_n_bets(corr: pd.DataFrame) -> float:
    """Estimate effective number of independent bets via eigenvalue decomposition.

    A fully correlated portfolio has effective N = 1.
    A perfectly diversified portfolio has effective N = n_assets.

    Reference: Meucci (2009) — "Managing diversification".

    Args:
        corr: Correlation matrix (n × n).

    Returns:
        Scalar effective N in [1, n].
    """
    eigenvalues = np.linalg.eigvalsh(corr.values)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # drop numerical noise
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(entropy))


def concentration_herfindahl(weights: Dict[str, float]) -> float:
    """Herfindahl–Hirschman Index of weight concentration.

    Returns 1 for a single asset, 1/n for equal weight.
    """
    w = np.array(list(weights.values()))
    return float(np.sum(w**2))


def build_correlation_report(
    returns: pd.DataFrame,
    name_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Build full correlation matrix and summary statistics.

    Args:
        returns: Daily returns DataFrame.
        name_map: Optional {sleeve_id: display_name} for renaming columns.

    Returns:
        Tuple of (corr_df_with_display_names, summary_dict).
    """
    corr = correlation_matrix(returns)

    if name_map:
        rename = {k: v for k, v in name_map.items() if k in corr.columns}
        corr = corr.rename(index=rename, columns=rename)

    # Summary statistics
    n = len(corr)
    # Upper-triangle pairs only
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    vals = upper.stack().values

    eff_n = effective_n_bets(correlation_matrix(returns))

    summary = {
        "n_sleeves": n,
        "mean_pairwise_correlation": round(float(vals.mean()), 4),
        "median_pairwise_correlation": round(float(np.median(vals)), 4),
        "max_pairwise_correlation": round(float(vals.max()), 4),
        "min_pairwise_correlation": round(float(vals.min()), 4),
        "effective_n_bets": round(eff_n, 2),
        "diversification_ratio": round(eff_n / n, 3),
    }

    logger.info(
        "Correlation report: mean=%.2f  eff_N=%.1f / %d sleeves",
        summary["mean_pairwise_correlation"],
        eff_n,
        n,
    )
    return corr, summary
