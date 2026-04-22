"""Tests for VaR and Expected Shortfall computation."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.var_es import (
    historical_var,
    parametric_var,
    historical_es,
    compute_var_es_summary,
)


@pytest.fixture
def normal_returns() -> pd.Series:
    """Standard normal daily returns (µ=0, σ=1% daily)."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-04", periods=2520)
    r = pd.Series(rng.normal(0.0, 0.01, 2520), index=dates, name="test")
    return r


def test_historical_var_level(normal_returns):
    var_95 = historical_var(normal_returns, 0.95)
    # For N(0, 0.01): 5th percentile ≈ -1.645 * 0.01 → VaR ≈ 0.01645
    assert 0.010 < var_95 < 0.030
    assert var_95 > 0  # must be positive loss


def test_historical_var_ordering(normal_returns):
    var_95 = historical_var(normal_returns, 0.95)
    var_99 = historical_var(normal_returns, 0.99)
    assert var_99 > var_95, "99% VaR must exceed 95% VaR"


def test_parametric_var_normal(normal_returns):
    var_95 = parametric_var(normal_returns, 0.95)
    # For N(0, σ): parametric VaR = 1.645 * σ
    expected = 1.6449 * normal_returns.std()
    assert abs(var_95 - expected) < 0.001


def test_historical_es_exceeds_var(normal_returns):
    var_95 = historical_var(normal_returns, 0.95)
    es_95 = historical_es(normal_returns, 0.95)
    assert es_95 > var_95, "ES must exceed VaR at the same confidence level"


def test_historical_es_positive(normal_returns):
    es = historical_es(normal_returns, 0.95)
    assert es > 0


def test_insufficient_data():
    short = pd.Series([0.01, -0.02, 0.005] * 10)
    with pytest.raises(ValueError, match="Insufficient"):
        historical_var(short, 0.95)


def test_summary_shape(normal_returns):
    df = compute_var_es_summary(normal_returns, [0.95, 0.99], name="test")
    assert len(df) == 2
    assert "historical_var_1d" in df.columns
    assert "historical_es_1d" in df.columns
    assert "parametric_var_1d" in df.columns


def test_summary_values_positive(normal_returns):
    df = compute_var_es_summary(normal_returns, [0.95])
    assert (df["historical_var_1d"] > 0).all()
    assert (df["historical_es_1d"] > 0).all()
