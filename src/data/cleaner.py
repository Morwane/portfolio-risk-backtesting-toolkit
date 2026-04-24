"""Data cleaning and alignment pipeline.

Takes raw price DataFrames from the loader and returns clean, aligned series
ready for return computation and analytics.

Use clean_pipeline() for the standard 2-tuple (prices, returns).
Use clean_pipeline_with_report() to also receive a CleaningReport audit record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Audit record produced by the cleaning pipeline."""

    # Input state
    input_sleeves: List[str] = field(default_factory=list)
    input_rows: int = 0

    # Drop stale step
    dropped_sleeves: List[str] = field(default_factory=list)
    drop_reasons: Dict[str, str] = field(default_factory=dict)  # {sleeve_id: reason}
    pre_inception_rows: Dict[str, int] = field(default_factory=dict)   # {sleeve_id: n_leading_nan}
    in_sample_missing_pct: Dict[str, float] = field(default_factory=dict)  # {sleeve_id: fraction}

    # Alignment step
    bdate_gap_fills: Dict[str, int] = field(default_factory=dict)  # {sleeve_id: n_filled}

    # Trim step
    pre_trim_start: Optional[str] = None
    common_start: Optional[str] = None
    rows_lost_to_trim: int = 0

    # Outlier step
    outliers_capped: Dict[str, int] = field(default_factory=dict)  # {sleeve_id: n_capped}

    # Final state
    output_sleeves: List[str] = field(default_factory=list)
    output_rows: int = 0

    def to_dict(self) -> Dict:
        return {
            "input_sleeves": len(self.input_sleeves),
            "input_rows": self.input_rows,
            "dropped_sleeves": self.dropped_sleeves,
            "drop_reasons": self.drop_reasons,
            "pre_inception_rows_by_sleeve": self.pre_inception_rows,
            "in_sample_missing_pct_by_sleeve": self.in_sample_missing_pct,
            "bdate_gap_fills_total": sum(self.bdate_gap_fills.values()),
            "bdate_gap_fills_by_sleeve": self.bdate_gap_fills,
            "pre_trim_start": self.pre_trim_start,
            "common_start": self.common_start,
            "rows_lost_to_trim": self.rows_lost_to_trim,
            "outliers_capped_total": sum(self.outliers_capped.values()),
            "outliers_capped_by_sleeve": self.outliers_capped,
            "output_sleeves": len(self.output_sleeves),
            "output_rows": self.output_rows,
        }

    def save(self, path: Path = Path("outputs/tables/data_quality_report.json")) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Data quality report saved: %s", path)
        return path

    def to_csv(self, path: Path = Path("outputs/tables/data_quality_report.csv")) -> Path:
        """Export per-sleeve summary as CSV."""
        rows = []
        all_sleeves = set(self.input_sleeves)
        for sid in sorted(all_sleeves):
            rows.append({
                "sleeve_id": sid,
                "dropped": sid in self.dropped_sleeves,
                "drop_reason": self.drop_reasons.get(sid, ""),
                "pre_inception_rows": self.pre_inception_rows.get(sid, 0),
                "in_sample_missing_pct": round(self.in_sample_missing_pct.get(sid, 0.0) * 100, 2),
                "gap_fills": self.bdate_gap_fills.get(sid, 0),
                "outliers_capped": self.outliers_capped.get(sid, 0),
                "in_output": sid in self.output_sleeves,
            })
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Table saved: %s", path)
        return path


def drop_stale_columns(
    prices: pd.DataFrame,
    max_missing_pct: float = 0.20,
    report: Optional[CleaningReport] = None,
) -> pd.DataFrame:
    """Drop columns where in-sample missing values exceed *max_missing_pct*.

    Leading NaNs before a sleeve's first valid price (pre-inception period) are
    excluded from the missing fraction calculation.  Only gaps *within* the
    sleeve's own history count toward the threshold.  This prevents sleeves with
    a later inception date (e.g. BNO / Brent crude, inception 2010) from being
    wrongly dropped when the global backtest window starts earlier (e.g. 2007).

    Args:
        prices: Raw price DataFrame (may contain leading NaNs for late-inception
                sleeves).
        max_missing_pct: Maximum allowed in-sample missing fraction (0–1).
        report: Optional CleaningReport to record per-sleeve audit data.

    Returns:
        Cleaned DataFrame with problematic columns removed.
    """
    to_drop = []

    for col in prices.columns:
        series = prices[col]
        first_valid = series.first_valid_index()

        if first_valid is None:
            # Column is entirely empty — always drop.
            insample_frac = 1.0
            pre_inception = len(series)
        else:
            in_sample = series.loc[first_valid:]
            insample_frac = float(in_sample.isnull().mean())
            pre_inception = int(prices.index.get_loc(first_valid))

        if report is not None:
            report.pre_inception_rows[col] = pre_inception
            report.in_sample_missing_pct[col] = insample_frac

        if insample_frac > max_missing_pct:
            to_drop.append(col)
            if report is not None:
                report.dropped_sleeves.append(col)
                report.drop_reasons[col] = (
                    f"{insample_frac * 100:.1f}% in-sample missing "
                    f"(threshold: {max_missing_pct * 100:.0f}%)"
                )

    if to_drop:
        logger.warning(
            "Dropping %d column(s) with >%.0f%% in-sample missing values: %s",
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
    report: Optional[CleaningReport] = None,
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
        before = prices.isnull().sum()
        prices = prices.ffill(limit=ffill_limit)
        if report is not None:
            after = prices.isnull().sum()
            for col in prices.columns:
                n_filled = int(before[col] - after[col])
                if n_filled > 0:
                    report.bdate_gap_fills[col] = (
                        report.bdate_gap_fills.get(col, 0) + n_filled
                    )

    return prices


def trim_to_common_start(
    prices: pd.DataFrame,
    min_sleeves: Optional[int] = None,
    report: Optional[CleaningReport] = None,
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
    if report is not None:
        report.pre_trim_start = str(prices.index[0].date())
        report.common_start = str(common_start.date())
        report.rows_lost_to_trim = len(prices) - len(trimmed)
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
    report: Optional[CleaningReport] = None,
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
        if report is not None:
            for col in returns.columns:
                n = int(outliers[col].sum())
                if n > 0:
                    report.outliers_capped[col] = n
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


def clean_pipeline_with_report(
    raw_prices: pd.DataFrame,
    max_missing_pct: float = 0.20,
    ffill_limit: int = 5,
    min_sleeves: Optional[int] = None,
    return_method: str = "simple",
    winsorise: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return an audit report as a third element.

    Args: same as clean_pipeline().

    Returns:
        (clean_prices, daily_returns, CleaningReport)
    """
    report = CleaningReport(
        input_sleeves=list(raw_prices.columns),
        input_rows=len(raw_prices),
    )
    logger.info("Running data cleaning pipeline ...")

    prices = drop_stale_columns(raw_prices, max_missing_pct, report=report)
    prices = align_to_common_dates(prices, ffill_limit=ffill_limit, report=report)
    prices, _ = trim_to_common_start(prices, min_sleeves, report=report)
    returns = compute_returns(prices, method=return_method).iloc[1:]
    if winsorise:
        returns = handle_outliers(returns, report=report)

    report.output_sleeves = list(returns.columns)
    report.output_rows = len(returns)
    logger.info(
        "Cleaning complete: %d sleeves, %d daily return observations.",
        len(returns.columns),
        len(returns),
    )
    return prices.iloc[1:], returns, report


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
