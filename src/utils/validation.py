"""Input validation helpers used across the toolkit."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_weights(weights: Dict[str, float], tolerance: float = 1e-6) -> None:
    """Raise ValueError if weights do not sum to ~1 or contain negative values.

    Args:
        weights: Mapping of sleeve id -> weight.
        tolerance: Allowed deviation from 1.0.
    """
    total = sum(weights.values())
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"Portfolio weights must be non-negative. Got: {weights}")
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Portfolio weights must sum to 1.0 (got {total:.6f}). "
            "Re-normalise before calling this function."
        )


def validate_prices(prices: pd.DataFrame, min_rows: int = 20) -> None:
    """Raise ValueError for obviously bad price DataFrames.

    Args:
        prices: DataFrame of price series (rows = dates, cols = sleeve ids).
        min_rows: Minimum required number of rows.
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")
    if len(prices) < min_rows:
        raise ValueError(
            f"Price DataFrame has only {len(prices)} rows; "
            f"minimum required is {min_rows}."
        )
    if prices.isnull().all().any():
        all_null_cols = prices.columns[prices.isnull().all()].tolist()
        raise ValueError(
            f"Columns with all-null values: {all_null_cols}. "
            "Remove these sleeves before computing returns."
        )


def validate_returns(returns: pd.DataFrame) -> None:
    """Basic sanity check on a returns DataFrame."""
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns DataFrame index must be a DatetimeIndex.")


def check_date_range(
    start: str,
    end: Optional[str],
    available_start: pd.Timestamp,
    available_end: pd.Timestamp,
    label: str = "",
) -> None:
    """Log a warning when requested dates fall outside available history.

    Args:
        start: Requested start date string.
        end: Requested end date string (None = today).
        available_start: Earliest available date in the data.
        available_end: Latest available date in the data.
        label: Identifier string for logging context.
    """
    req_start = pd.Timestamp(start)
    req_end = pd.Timestamp(end) if end else pd.Timestamp.today()

    if req_start < available_start:
        logger.warning(
            "%sRequested start %s precedes available data start %s. "
            "Data will begin at %s.",
            f"[{label}] " if label else "",
            req_start.date(),
            available_start.date(),
            available_start.date(),
        )
    if req_end > available_end:
        logger.warning(
            "%sRequested end %s exceeds available data end %s.",
            f"[{label}] " if label else "",
            req_end.date(),
            available_end.date(),
        )


def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of weights normalised to sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Cannot normalise weights: total is zero or negative.")
    return {k: v / total for k, v in weights.items()}


def filter_available_sleeves(
    weights: Dict[str, float],
    available: List[str],
    renormalise: bool = True,
) -> Dict[str, float]:
    """Remove sleeves not in *available* from *weights*, optionally renormalising.

    Args:
        weights: Original weight dictionary.
        available: List of sleeve ids that have valid data.
        renormalise: If True, renormalise remaining weights to sum to 1.

    Returns:
        Filtered (and optionally renormalised) weight dictionary.
    """
    missing = [s for s in weights if s not in available]
    if missing:
        logger.warning(
            "Dropping sleeves absent from price data: %s. "
            "Remaining weights will be %s.",
            missing,
            "renormalised" if renormalise else "unchanged",
        )
    filtered = {k: v for k, v in weights.items() if k in available}
    if not filtered:
        raise ValueError("No valid sleeves remain after filtering. Check data loading.")
    return normalise_weights(filtered) if renormalise else filtered
