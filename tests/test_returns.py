"""Tests for return computation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.returns import (
    to_simple_returns,
    to_log_returns,
    annualise_return,
    annualise_volatility,
    monthly_returns_table,
    cumulative_return_series,
    TRADING_DAYS,
)


@pytest.fixture
def flat_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=252)
    # All prices grow at exactly 10% per year
    daily_growth = (1.10) ** (1 / 252)
    prices = pd.DataFrame(
        {"A": 100.0 * daily_growth ** np.arange(252)},
        index=dates,
    )
    return prices


@pytest.fixture
def sample_returns() -> pd.Series:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-02", periods=2520)
    r = pd.Series(rng.normal(0.0003, 0.010, 2520), index=dates, name="port")
    return r


def test_simple_returns_shape(flat_prices):
    r = to_simple_returns(flat_prices)
    assert len(r) == len(flat_prices) - 1
    assert r.columns.tolist() == ["A"]


def test_simple_returns_constant_growth(flat_prices):
    r = to_simple_returns(flat_prices)
    daily_growth = (1.10) ** (1 / 252) - 1
    assert np.allclose(r["A"].values, daily_growth, atol=1e-10)


def test_log_returns_shape(flat_prices):
    r = to_log_returns(flat_prices)
    assert len(r) == len(flat_prices) - 1


def test_annualise_return_known():
    # 10% total return over 252 days → annualised = 10%
    r = annualise_return(0.10, 252)
    assert abs(r - 0.10) < 1e-8


def test_annualise_return_zero_days():
    assert np.isnan(annualise_return(0.10, 0))


def test_annualise_volatility(sample_returns):
    ann_vol = annualise_volatility(sample_returns)
    # Should be near 0.010 * sqrt(252) ≈ 0.159
    assert 0.12 < ann_vol < 0.20


def test_cumulative_return_series_base(sample_returns):
    cum = cumulative_return_series(sample_returns, base=100.0)
    assert cum.iloc[0] == pytest.approx(100.0 * (1 + sample_returns.iloc[0]))


def test_monthly_returns_table_shape(sample_returns):
    table = monthly_returns_table(sample_returns)
    # Should have 12 rows (months) and multiple year columns
    assert table.shape[0] <= 12
    assert table.shape[1] >= 1
