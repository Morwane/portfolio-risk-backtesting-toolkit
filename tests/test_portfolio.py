"""Tests for portfolio construction and rebalancing."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.construction import build_portfolio, compute_portfolio_returns
from src.portfolio.rebalancing import get_rebalancing_dates, compute_rebalancing_costs
from src.portfolio.weights import make_equal_weight


@pytest.fixture
def two_asset_returns() -> pd.DataFrame:
    """Two assets: one always up, one always flat."""
    dates = pd.bdate_range("2020-01-02", periods=504)
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "equity": rng.normal(0.0004, 0.010, 504),
            "bond": rng.normal(0.0001, 0.003, 504),
        },
        index=dates,
    )
    return df


def test_build_portfolio_nav_starts_at_100(two_asset_returns):
    weights = {"equity": 0.6, "bond": 0.4}
    nav, _ = build_portfolio(two_asset_returns, weights, "buy_and_hold")
    assert nav.iloc[0] == 100.0


def test_build_portfolio_nav_length(two_asset_returns):
    weights = {"equity": 0.6, "bond": 0.4}
    nav, _ = build_portfolio(two_asset_returns, weights, "quarterly")
    assert len(nav) == len(two_asset_returns)


def test_build_portfolio_nav_positive(two_asset_returns):
    weights = {"equity": 0.6, "bond": 0.4}
    nav, _ = build_portfolio(two_asset_returns, weights)
    assert (nav > 0).all()


def test_equal_weight_sums_to_one():
    w = make_equal_weight(["a", "b", "c", "d"])
    assert abs(sum(w.values()) - 1.0) < 1e-10
    assert all(abs(v - 0.25) < 1e-10 for v in w.values())


def test_portfolio_returns_length(two_asset_returns):
    weights = {"equity": 0.6, "bond": 0.4}
    nav, _ = build_portfolio(two_asset_returns, weights)
    r = compute_portfolio_returns(nav)
    assert len(r) == len(nav) - 1


def test_rebalancing_dates_quarterly(two_asset_returns):
    dates = get_rebalancing_dates(two_asset_returns.index, "quarterly")
    assert len(dates) >= 1
    assert all(d in two_asset_returns.index for d in dates)


def test_rebalancing_dates_buy_and_hold(two_asset_returns):
    dates = get_rebalancing_dates(two_asset_returns.index, "buy_and_hold")
    assert len(dates) == 1
    assert dates[0] == two_asset_returns.index[0]


def test_transaction_cost_zero_when_no_change():
    w = {"equity": 0.6, "bond": 0.4}
    cost = compute_rebalancing_costs(w, w, portfolio_value=1000.0, cost_bps=2.0)
    assert cost == 0.0


def test_transaction_cost_positive_when_change():
    w_before = {"equity": 0.7, "bond": 0.3}
    w_after = {"equity": 0.6, "bond": 0.4}
    cost = compute_rebalancing_costs(w_before, w_after, portfolio_value=10_000.0, cost_bps=2.0)
    assert cost > 0


def test_missing_sleeve_renormalised(two_asset_returns):
    """Weights with a sleeve not in returns should be dropped and renormalised."""
    weights = {"equity": 0.5, "bond": 0.3, "missing_sleeve": 0.2}
    nav, _ = build_portfolio(two_asset_returns, weights)
    assert len(nav) == len(two_asset_returns)
