"""Portfolio weight loading and management.

Loads weight configurations from config/portfolio_weights.yaml and
provides helpers for equal-weight and custom weight dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.utils.logging_utils import get_logger
from src.utils.validation import normalise_weights, validate_weights

logger = get_logger(__name__)

_WEIGHTS_PATH = Path(__file__).parents[2] / "config" / "portfolio_weights.yaml"


def load_all_portfolios(path: Optional[Path] = None) -> Dict:
    """Load the full portfolio_weights.yaml as a dict."""
    p = path or _WEIGHTS_PATH
    with open(p, "r") as fh:
        return yaml.safe_load(fh)


def get_portfolio_weights(
    portfolio_id: str,
    available_sleeves: Optional[List[str]] = None,
    path: Optional[Path] = None,
) -> Dict[str, float]:
    """Return normalised weights for a named portfolio.

    For ``equal_weight`` portfolios, distributes 1/N across *available_sleeves*.

    Args:
        portfolio_id: Key in portfolio_weights.yaml (e.g. ``"strategic_diversified"``).
        available_sleeves: If provided, remove missing sleeves and renormalise.
        path: Override path to portfolio_weights.yaml.

    Returns:
        Dict mapping sleeve_id -> weight (sums to 1.0).
    """
    config = load_all_portfolios(path)
    portfolios = config.get("portfolios", {})

    if portfolio_id not in portfolios:
        raise KeyError(
            f"Portfolio '{portfolio_id}' not found. "
            f"Available: {list(portfolios.keys())}"
        )

    port = portfolios[portfolio_id]
    mode = port.get("mode", "static")

    if mode == "equal_weight":
        if not available_sleeves:
            raise ValueError(
                "available_sleeves must be provided for equal_weight portfolios."
            )
        n = len(available_sleeves)
        weights = {s: 1.0 / n for s in available_sleeves}
        logger.info("Equal-weight portfolio: %.4f per sleeve across %d sleeves.", 1.0 / n, n)
        return weights

    raw_weights = port.get("weights", {})
    if not raw_weights:
        raise ValueError(f"Portfolio '{portfolio_id}' has no weights defined.")

    # Filter to available sleeves if provided
    if available_sleeves is not None:
        missing = [s for s in raw_weights if s not in available_sleeves]
        if missing:
            logger.warning(
                "Portfolio '%s': dropping %d unavailable sleeves: %s",
                portfolio_id, len(missing), missing,
            )
        raw_weights = {k: v for k, v in raw_weights.items() if k in available_sleeves}
        if not raw_weights:
            raise ValueError(
                f"No weights remain after filtering to available sleeves for '{portfolio_id}'."
            )

    weights = normalise_weights(raw_weights)
    validate_weights(weights)
    return weights


def list_portfolio_ids(path: Optional[Path] = None) -> List[str]:
    """Return all portfolio IDs defined in the config."""
    config = load_all_portfolios(path)
    return list(config.get("portfolios", {}).keys())


def make_equal_weight(sleeves: List[str]) -> Dict[str, float]:
    """Return a simple 1/N equal-weight dict for the given sleeve list."""
    if not sleeves:
        raise ValueError("Cannot create equal-weight portfolio: sleeve list is empty.")
    n = len(sleeves)
    return {s: 1.0 / n for s in sleeves}
