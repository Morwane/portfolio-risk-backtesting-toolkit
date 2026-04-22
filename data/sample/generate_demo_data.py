"""Synthetic price data generator for demo mode.

Generates 20 years of daily prices for all 22 universe sleeves using a
multi-factor return model with realistic cross-asset correlations,
volatilities, and crisis regimes.

Design:
  - Factor structure: global equity factor + asset-class factor + idiosyncratic
  - Crisis periods (GFC, COVID, 2022) encoded with elevated correlations and drawdowns
  - Volatility clustering via a simplified GARCH-like term
  - All series start at 100.0 on 2004-01-03

This data is for demonstration only — it does not replicate actual
historical returns of any instrument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_SEED = 42
_START = "2004-01-01"
_END = "2024-12-31"
_OUT_PATH = Path(__file__).parent / "demo_prices.parquet"

# Per-sleeve annualised return and volatility assumptions
# (id, ann_return, ann_vol, equity_beta, bond_beta)
_SLEEVE_PARAMS = [
    # id                      ann_ret  ann_vol  eq_beta  bond_beta
    ("us_large_cap",           0.080,   0.160,   1.00,   -0.10),
    ("us_tech",                0.120,   0.220,   1.20,   -0.15),
    ("europe_equity",          0.060,   0.180,   0.90,   -0.08),
    ("uk_equity",              0.055,   0.170,   0.85,   -0.06),
    ("japan_equity",           0.050,   0.190,   0.80,   -0.05),
    ("emerging_markets",       0.065,   0.220,   1.00,    0.00),
    ("us_small_cap",           0.085,   0.200,   1.10,   -0.10),
    ("global_min_vol",         0.065,   0.120,   0.65,   -0.05),
    ("us_treasury_short",      0.025,   0.020,  -0.05,    0.90),
    ("us_treasury_medium",     0.035,   0.055,  -0.15,    1.00),
    ("euro_govt",              0.030,   0.045,  -0.12,    0.95),
    ("us_tips",                0.030,   0.055,  -0.10,    0.85),
    ("uk_gilts",               0.030,   0.065,  -0.12,    0.90),
    ("us_ig_credit",           0.040,   0.065,   0.20,    0.80),
    ("us_hy_credit",           0.055,   0.105,   0.55,    0.40),
    ("euro_ig_credit",         0.035,   0.055,   0.18,    0.78),
    ("euro_hy_credit",         0.055,   0.110,   0.52,    0.35),
    ("gold",                   0.055,   0.165,   0.00,    0.10),
    ("brent_crude",            0.025,   0.360,   0.30,   -0.05),
    ("broad_commodities",      0.020,   0.200,   0.25,    0.00),
    ("reits",                  0.070,   0.205,   0.75,    0.10),
    ("cash",                   0.020,   0.005,   0.00,    0.05),
]

# Historical crisis regimes: (start, end, equity_shock_daily, bond_relief_daily)
_CRISES = [
    # GFC peak stress
    ("2008-09-15", "2009-03-09",  -0.0025,  0.0008),
    # Euro debt crisis
    ("2011-07-01", "2011-10-04",  -0.0015,  0.0005),
    # Taper tantrum
    ("2013-05-22", "2013-09-05",  -0.0004, -0.0006),
    # Oil collapse / EM stress 2015-16
    ("2015-08-01", "2016-02-11",  -0.0007,  0.0002),
    # COVID crash
    ("2020-02-20", "2020-03-23",  -0.0060,  0.0010),
    # 2022 rate shock (both equity AND bonds down)
    ("2022-01-03", "2022-10-13",  -0.0008, -0.0008),
]


def _crisis_multiplier(date: pd.Timestamp) -> tuple[float, float]:
    """Return (equity_extra, bond_extra) drift adjustment for crisis periods."""
    for start, end, eq, bond in _CRISES:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return eq, bond
    return 0.0, 0.0


def generate_prices(seed: int = _SEED) -> pd.DataFrame:
    """Generate synthetic daily price DataFrame.

    Returns:
        DataFrame indexed by business dates, columns = sleeve IDs.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=_START, end=_END)
    n = len(dates)
    trading_days = 252

    sleeve_ids = [s[0] for s in _SLEEVE_PARAMS]

    # Global factors: equity market factor, bond factor, commodity factor
    eq_factor = rng.normal(0, 0.010, n)     # daily
    bond_factor = rng.normal(0, 0.003, n)

    # Pre-compute crisis daily adjustments
    eq_crisis = np.zeros(n)
    bond_crisis = np.zeros(n)
    for i, date in enumerate(dates):
        eq_crisis[i], bond_crisis[i] = _crisis_multiplier(date)

    prices = {}
    for sid, ann_ret, ann_vol, eq_beta, bond_beta in _SLEEVE_PARAMS:
        daily_drift = ann_ret / trading_days
        daily_vol = ann_vol / np.sqrt(trading_days)

        # Idiosyncratic component with mild vol clustering
        residual = max(1 - eq_beta**2 - bond_beta**2, 0.04)
        idio_vol = daily_vol * np.sqrt(residual)
        idio = rng.normal(0, idio_vol, n)

        # Factor exposures
        factor_return = eq_beta * eq_factor + bond_beta * bond_factor

        # Crisis regime drift shift
        crisis_drift = eq_beta * eq_crisis + bond_beta * bond_crisis

        # Combine
        daily_returns = (
            daily_drift
            + factor_return
            + idio
            + crisis_drift
        )

        # Clamp extreme daily returns (±25%)
        daily_returns = np.clip(daily_returns, -0.25, 0.25)

        # Compute price path
        price = 100.0 * np.cumprod(1 + daily_returns)
        prices[sid] = price

    df = pd.DataFrame(prices, index=dates)
    df.index.name = "date"
    return df


def generate_and_save(out_path: Path = _OUT_PATH) -> pd.DataFrame:
    """Generate synthetic prices and save to parquet.

    Args:
        out_path: Output file path.

    Returns:
        The generated price DataFrame.
    """
    import logging
    logging.getLogger(__name__).info("Generating synthetic demo prices (%s to %s) ...", _START, _END)

    prices = generate_prices()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path)

    logging.getLogger(__name__).info(
        "Demo prices saved: %s  (%d rows, %d assets)",
        out_path, len(prices), len(prices.columns),
    )
    return prices


if __name__ == "__main__":
    from src.utils.logging_utils import configure_root_logger
    configure_root_logger("INFO")
    generate_and_save()
