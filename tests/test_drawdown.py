"""Tests for drawdown analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.drawdown import drawdown_series, max_drawdown, drawdown_table


@pytest.fixture
def flat_returns() -> pd.Series:
    """Returns that go up 1% every day — no drawdown."""
    dates = pd.bdate_range("2020-01-02", periods=100)
    return pd.Series([0.01] * 100, index=dates, name="flat")


@pytest.fixture
def crash_returns() -> pd.Series:
    """50% crash then recovery."""
    dates = pd.bdate_range("2020-01-02", periods=300)
    r = [0.005] * 100 + [-0.006] * 100 + [0.007] * 100
    return pd.Series(r, index=dates, name="crash")


def test_no_drawdown(flat_returns):
    dd = drawdown_series(flat_returns)
    assert (dd >= -1e-10).all(), "No drawdown when returns are always positive"


def test_max_drawdown_negative(crash_returns):
    mdd = max_drawdown(crash_returns)
    assert mdd < 0, "Max drawdown must be negative"


def test_max_drawdown_magnitude(crash_returns):
    mdd = max_drawdown(crash_returns)
    # -0.006 * 100 daily compounded ≈ -45% peak to trough
    assert mdd < -0.30, "Crash should produce significant drawdown"


def test_drawdown_series_range(crash_returns):
    dd = drawdown_series(crash_returns)
    assert dd.min() <= 0
    assert dd.max() <= 0.0001  # at peaks, drawdown ≤ 0


def test_drawdown_table_columns(crash_returns):
    table = drawdown_table(crash_returns, top_n=5)
    expected_cols = {
        "peak_date", "trough_date", "drawdown_pct",
        "duration_peak_to_trough_days",
    }
    assert expected_cols.issubset(set(table.columns))


def test_drawdown_table_not_empty(crash_returns):
    table = drawdown_table(crash_returns)
    assert len(table) >= 1


def test_drawdown_table_values_negative(crash_returns):
    table = drawdown_table(crash_returns)
    assert (table["drawdown_pct"] <= 0).all(), "Drawdown percentages must be ≤ 0"


def test_drawdown_table_top_n(crash_returns):
    table = drawdown_table(crash_returns, top_n=3)
    assert len(table) <= 3
