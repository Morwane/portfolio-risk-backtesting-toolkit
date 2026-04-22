"""Custom shock scenario application.

Applies synthetic factor shocks (defined in stress_scenarios.yaml) to
a portfolio by multiplying each sleeve's shock by its weight, then
aggregates to a total portfolio P&L impact.

This is a first-order linear approximation — not a full revaluation —
appropriate for reporting and sensitivity analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SCENARIOS_PATH = Path(__file__).parents[2] / "config" / "stress_scenarios.yaml"


def load_custom_shocks(path: Optional[Path] = None) -> Dict:
    """Load custom shock scenario definitions from YAML."""
    p = path or _SCENARIOS_PATH
    with open(p, "r") as fh:
        config = yaml.safe_load(fh)
    return config.get("custom_shocks", {})


def apply_shock_to_portfolio(
    weights: Dict[str, float],
    shocks: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-sleeve and total portfolio impact of a shock vector.

    portfolio_impact = Σ  w_i × shock_i

    Sleeves not in *shocks* are assumed to have zero shock.

    Args:
        weights: Dict of sleeve_id -> weight (should sum to ~1.0).
        shocks: Dict of sleeve_id -> fractional shock (e.g. -0.20 = -20%).

    Returns:
        Dict with keys:
          - ``total_portfolio_impact``: weighted sum of shocks.
          - per-sleeve keys: ``{sleeve_id}_contribution``.
    """
    result = {"total_portfolio_impact": 0.0}
    for sleeve_id, weight in weights.items():
        shock = shocks.get(sleeve_id, 0.0)
        contribution = weight * shock
        result[f"{sleeve_id}_contribution"] = round(contribution, 6)
        result["total_portfolio_impact"] += contribution

    result["total_portfolio_impact"] = round(result["total_portfolio_impact"], 6)
    return result


def run_custom_shocks(
    weights: Dict[str, float],
    scenarios_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run all custom shock scenarios against a set of portfolio weights.

    Args:
        weights: Dict of sleeve_id -> weight.
        scenarios_path: Override path to stress_scenarios.yaml.

    Returns:
        DataFrame with one row per scenario. Columns:
        scenario_id, scenario_name, description,
        total_portfolio_impact, and per-sleeve _contribution columns.
    """
    shocks_config = load_custom_shocks(scenarios_path)
    rows = []

    for scenario_id, spec in shocks_config.items():
        name = spec["name"]
        description = spec.get("description", "")
        shocks = spec.get("shocks", {})

        impact = apply_shock_to_portfolio(weights, shocks)
        row = {
            "scenario_id": scenario_id,
            "scenario_name": name,
            "description": description,
            **impact,
        }
        rows.append(row)

        logger.info(
            "Shock '%s': total portfolio impact = %.2f%%",
            name,
            impact["total_portfolio_impact"] * 100,
        )

    return pd.DataFrame(rows)


def run_custom_shocks_multi(
    portfolio_weights_dict: Dict[str, Dict[str, float]],
    scenarios_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run all shock scenarios across multiple portfolio weight sets.

    Args:
        portfolio_weights_dict: Dict of {portfolio_id: weights_dict}.
        scenarios_path: Override path.

    Returns:
        Long-form DataFrame with columns:
        portfolio_id, scenario_id, scenario_name, total_portfolio_impact.
    """
    rows = []
    for portfolio_id, weights in portfolio_weights_dict.items():
        df = run_custom_shocks(weights, scenarios_path)
        df["portfolio_id"] = portfolio_id
        rows.append(df[["portfolio_id", "scenario_id", "scenario_name",
                          "description", "total_portfolio_impact"]])

    return pd.concat(rows, ignore_index=True)
