"""Data cleaning and alignment pipeline.

Takes raw price DataFrames from the loader and returns clean, aligned series
ready for return computation and analytics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def drop_stale_columns(
    prices: pd.DataFrame,
    max_missing_pct: float = 0.20,
) -> pd.DataFrame:
    """Drop columns where more than *max_missing_pct* of values are NaN.

    Args:
        prices: Raw price DataFrame.
        max_missing_pct: Maximum allowed fraction of missing values (0–1).

    Returns:
        Cleaned DataFrame with problematic columns removed.
    """
    missing_frac = prices.isnull().mean()
    to_drop = missing_frac[missing_frac > max_missing_pct].index.tolist()
    if to_drop:
        logger.warning(
            "Dropping %d column(s) with >%.0f%% missing values: %s",
            len(to_drop),
            max_missing_pct * 100,
            to_drop,
        )
        prices = prices.drop(columns=to_drop)
    return prices


def align_to_common_dates(
    prices: pd.DataFrame,
    method: str = "ffill",
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Align all price series to a common business-day index.

    Fills small gaps (≤ ffill_limit days) by forward-filling. Remaining NaN
    at the start of a series are left as-is (the series simply starts later).

    Args:
        prices: Price DataFrame (may have gaps from weekends/holidays).
        method: Fill method — ``"ffill"`` (forward fill) only for now.
        ffill_limit: Max consecutive NaNs to forward-fill.

    Returns:
        Aligned price DataFrame on a continuous business-day index.
    """
    if prices.empty:
        return prices

    full_idx = pd.bdate_range(start=prices.index.min(), end=prices.index.max())
    prices = prices.reindex(full_idx)
    prices.index.name = "date"

    if method == "ffill":
        prices = prices.ffill(limit=ffill_limit)

    return prices


def trim_to_common_start(
    prices: pd.DataFrame,
    min_sleeves: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Trim prices to the date when at least *min_sleeves* columns have data.

    If min_sleeves is None, trims to when ALL columns have data (strict inner join).

    Args:
        prices: Aligned price DataFrame.
        min_sleeves: Minimum number of non-NaN columns required per row.

    Returns:
        (trimmed_prices, common_start_date)
    """
    threshold = min_sleeves if min_sleeves is not None else len(prices.columns)
    valid = prices.count(axis=1) >= threshold
    if not valid.any():
        raise ValueError(
            f"No date has >= {threshold} non-NaN sleeves. "
            "Reduce min_sleeves or check data."
        )
    common_start = prices[valid].index[0]
    trimmed = prices.loc[common_start:]
    logger.info(
        "Common history start: %s  (%d sleeves, %d rows)",
        common_start.date(),
        len(trimmed.columns),
        len(trimmed),
    )
    return trimmed, common_start


def handle_outliers(
    returns: pd.DataFrame,
    zscore_threshold: float = 10.0,
) -> pd.DataFrame:
    """Cap return outliers at ±zscore_threshold standard deviations.

    This is a conservative winsorisation to catch data errors (e.g., dividend
    adjustments, splits) without masking genuine extreme returns.
    Flags and logs any values that were capped.

    Args:
        returns: Daily returns DataFrame.
        zscore_threshold: Z-score cutoff for outlier flagging.

    Returns:
        Winsorised returns DataFrame.
    """
    mean = returns.mean()
    std = returns.std()
    zscores = (returns - mean).abs() / std
    outliers = (zscores > zscore_threshold) & returns.notna()

    n_outliers = outliers.sum().sum()
    if n_outliers > 0:
        logger.warning(
            "Capping %d outlier return(s) beyond ±%.0f σ.",
            n_outliers,
            zscore_threshold,
        )
        upper = mean + zscore_threshold * std
        lower = mean - zscore_threshold * std
        returns = returns.clip(lower=lower, upper=upper, axis=1)

    return returns


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """Compute daily returns from prices.

    Args:
        prices: Price DataFrame.
        method: ``"simple"`` (arithmetic) or ``"log"`` (geometric).

    Returns:
        Returns DataFrame (first row is NaN and should be dropped by callers).
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}. Use 'simple' or 'log'.")

    returns.index.name = "date"
    return returns


def clean_pipeline(
    raw_prices: pd.DataFrame,
    max_missing_pct: float = 0.20,
    ffill_limit: int = 5,
    min_sleeves: Optional[int] = None,
    return_method: str = "simple",
    winsorise: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline on raw prices.

    Steps:
      1. Drop columns with excessive missing data
      2. Align to business-day index (forward fill small gaps)
      3. Trim to common start date
      4. Compute daily returns
      5. Optionally winsorise outliers

    Args:
        raw_prices: Output from loader.load_prices().
        max_missing_pct: Threshold for column removal.
        ffill_limit: Max consecutive NaN days to forward-fill.
        min_sleeves: Passed to trim_to_common_start.
        return_method: ``"simple"`` or ``"log"``.
        winsorise: Whether to cap return outliers.

    Returns:
        (clean_prices, daily_returns) — both share the same DatetimeIndex.
    """
    logger.info("Running data cleaning pipeline ...")

    prices = drop_stale_columns(raw_prices, max_missing_pct)
    prices = align_to_common_dates(prices, ffill_limit=ffill_limit)
    prices, _ = trim_to_common_start(prices, min_sleeves)

    returns = compute_returns(prices, method=return_method).iloc[1:]

    if winsorise:
        returns = handle_outliers(returns)

    logger.info(
        "Cleaning complete: %d sleeves, %d daily return observations.",
        len(returns.columns),
        len(returns),
    )
    return prices.iloc[1:], returns


def resample_to_monthly(
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compound daily returns to monthly frequency.

    Args:
        daily_returns: Daily simple return DataFrame.

    Returns:
        Monthly compounded returns DataFrame.
    """
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    monthly.index.name = "date"
    return monthly
