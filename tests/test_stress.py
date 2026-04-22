"""Tests for stress testing modules."""

import numpy as np
import pandas as pd
import pytest

from src.stress.shocks import apply_shock_to_portfolio, run_custom_shocks
from src.stress.historical import run_historical_stress


@pytest.fixture
def simple_weights() -> dict:
    return {
        "us_large_cap": 0.40,
        "us_treasury_medium": 0.40,
        "gold": 0.20,
    }


@pytest.fixture
def sample_returns() -> pd.Series:
    """Synthetic portfolio returns spanning 2019–2024."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-02", "2024-12-31")
    r = pd.Series(rng.normal(0.0003, 0.008, len(dates)), index=dates, name="portfolio")
    # Inject COVID-like crash
    crash_start = pd.Timestamp("2020-02-20")
    crash_end = pd.Timestamp("2020-03-23")
    crash_mask = (r.index >= crash_start) & (r.index <= crash_end)
    r.loc[crash_mask] = -0.025
    return r


# ── Shock tests ───────────────────────────────────────────────────────────────

def test_shock_zero_impact(simple_weights):
    """Zero shocks → zero portfolio impact."""
    result = apply_shock_to_portfolio(simple_weights, {})
    assert result["total_portfolio_impact"] == 0.0


def test_shock_full_equity_down(simple_weights):
    shocks = {"us_large_cap": -0.20}
    result = apply_shock_to_portfolio(simple_weights, shocks)
    # 40% equity weight × -20% shock = -8% portfolio
    assert abs(result["total_portfolio_impact"] - (-0.08)) < 1e-9


def test_shock_diversification(simple_weights):
    """Bonds up, equities down — diversified portfolio is protected."""
    shocks = {
        "us_large_cap": -0.20,
        "us_treasury_medium": 0.05,
        "gold": 0.08,
    }
    result = apply_shock_to_portfolio(simple_weights, shocks)
    # 0.40*(-0.20) + 0.40*(0.05) + 0.20*(0.08) = -0.08 + 0.02 + 0.016 = -0.044
    expected = 0.40 * (-0.20) + 0.40 * 0.05 + 0.20 * 0.08
    assert abs(result["total_portfolio_impact"] - expected) < 1e-9


def test_shock_contribution_sum(simple_weights):
    shocks = {"us_large_cap": -0.15, "gold": 0.10}
    result = apply_shock_to_portfolio(simple_weights, shocks)
    contrib_sum = sum(
        v for k, v in result.items() if k.endswith("_contribution")
    )
    assert abs(contrib_sum - result["total_portfolio_impact"]) < 1e-9


# ── Historical stress tests ───────────────────────────────────────────────────

def test_historical_stress_covid_present(sample_returns):
    df = run_historical_stress(sample_returns)
    assert "scenario_name" in df.columns
    # COVID crash should be in the results
    names = df["scenario_name"].tolist()
    assert any("COVID" in n for n in names)


def test_historical_stress_covid_negative(sample_returns):
    df = run_historical_stress(sample_returns)
    covid = df[df["scenario_name"].str.contains("COVID-19 Crash")]
    if not covid.empty and covid["status"].iloc[0] == "ok":
        assert covid["portfolio_total_return"].iloc[0] < 0


def test_historical_stress_no_data_scenario():
    """Scenario outside data range should be flagged as no_data."""
    dates = pd.bdate_range("2022-01-03", "2023-12-31")
    r = pd.Series(np.zeros(len(dates)), index=dates)
    df = run_historical_stress(r)
    # GFC (2008) should have no_data status
    gfc = df[df["scenario_id"] == "gfc_2008"]
    if not gfc.empty:
        assert gfc["status"].iloc[0] == "no_data"


def test_historical_stress_output_columns(sample_returns):
    df = run_historical_stress(sample_returns)
    expected = {
        "scenario_name", "start", "end",
        "portfolio_total_return", "max_drawdown_in_window", "status",
    }
    assert expected.issubset(set(df.columns))
