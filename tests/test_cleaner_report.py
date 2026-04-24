"""Tests for CleaningReport and clean_pipeline_with_report."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    clean_pipeline_with_report,
    drop_stale_columns,
    CleaningReport,
)


def _make_prices(n_days: int = 300, n_sleeves: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_sleeves)), axis=0),
        index=idx,
        columns=[f"s{i}" for i in range(n_sleeves)],
    )
    return prices


class TestCleaningReport:

    def test_pipeline_returns_3_tuple(self):
        prices = _make_prices()
        result = clean_pipeline_with_report(prices)
        assert len(result) == 3
        _, _, report = result
        assert isinstance(report, CleaningReport)

    def test_report_tracks_dropped_sleeve(self):
        prices = _make_prices(n_days=400)
        # Scatter >20% NaN through the middle of the series (in-sample gaps, not leading).
        prices.iloc[80:170, 0] = np.nan  # 90 / 400 = 22.5% in-sample missing
        _, _, report = clean_pipeline_with_report(prices)
        assert "s0" in report.dropped_sleeves
        assert report.drop_reasons["s0"] != ""

    def test_report_output_sleeves_after_drop(self):
        prices = _make_prices(n_days=400)
        prices.iloc[80:170, 0] = np.nan  # >20% in-sample → dropped
        _, returns, report = clean_pipeline_with_report(prices)
        assert set(report.output_sleeves) == set(returns.columns)
        assert "s0" not in report.output_sleeves

    def test_report_common_start_set(self):
        prices = _make_prices(n_days=300)
        _, _, report = clean_pipeline_with_report(prices)
        assert report.common_start is not None

    def test_report_output_rows_matches_returns(self):
        prices = _make_prices(n_days=300)
        _, returns, report = clean_pipeline_with_report(prices)
        assert report.output_rows == len(returns)

    def test_gap_fills_recorded(self):
        prices = _make_prices(n_days=400)
        # Introduce small gap (3 days in the middle of series)
        prices.iloc[150:153, 1] = np.nan
        _, _, report = clean_pipeline_with_report(prices)
        # s1 should have some fills recorded (gap ≤ 5 days)
        assert report.bdate_gap_fills.get("s1", 0) > 0

    def test_to_csv_produces_file(self, tmp_path):
        prices = _make_prices(n_days=300)
        _, _, report = clean_pipeline_with_report(prices)
        path = tmp_path / "dq.csv"
        report.to_csv(path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "sleeve_id" in df.columns
        assert "dropped" in df.columns

    def test_pre_inception_leading_nans_not_dropped(self):
        """A sleeve whose NaNs are entirely pre-inception (leading) must not be dropped."""
        prices = _make_prices(n_days=400)
        # 25% leading NaN — mimics a late-inception sleeve (e.g. BNO pattern).
        prices.iloc[:100, 0] = np.nan
        _, _, report = clean_pipeline_with_report(prices)
        assert "s0" not in report.dropped_sleeves
        assert "s0" in report.output_sleeves

    def test_pre_inception_rows_recorded(self):
        """pre_inception_rows in report equals the number of leading NaN rows."""
        prices = _make_prices(n_days=400)
        prices.iloc[:60, 0] = np.nan  # 60 leading NaN rows for s0
        _, _, report = clean_pipeline_with_report(prices)
        assert report.pre_inception_rows.get("s0", 0) == 60

    def test_brent_pattern_retained(self):
        """Sleeve with ~20% global missingness but clean in-sample data is retained."""
        n = 500
        idx = pd.bdate_range("2007-01-01", periods=n)
        # brent: NaN for first 30% of history (pre-inception), clean thereafter
        data = {"anchor": np.ones(n)}
        brent = np.full(n, np.nan)
        brent[int(n * 0.30):] = np.linspace(80, 100, n - int(n * 0.30))
        data["brent"] = brent
        prices = pd.DataFrame(data, index=idx)
        report = CleaningReport(input_sleeves=["anchor", "brent"])
        result = drop_stale_columns(prices, max_missing_pct=0.20, report=report)
        assert "brent" in result.columns
        assert "brent" not in report.dropped_sleeves
        assert report.pre_inception_rows["brent"] == int(n * 0.30)
        assert report.in_sample_missing_pct["brent"] == pytest.approx(0.0, abs=0.01)


class TestDropStaleColumnsWithReport:

    def test_report_updated(self):
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=200)
        df = pd.DataFrame(rng.normal(size=(200, 3)), index=idx, columns=["a", "b", "c"])
        # Mid-series NaN so it counts as in-sample missing (not pre-inception).
        df.iloc[50:100, 2] = np.nan  # 25% in-sample missing on "c"
        report = CleaningReport(input_sleeves=["a", "b", "c"])
        result = drop_stale_columns(df, max_missing_pct=0.20, report=report)
        assert "c" in report.dropped_sleeves
        assert "c" in report.drop_reasons
        assert "c" not in result.columns

    def test_leading_nan_not_dropped(self):
        """Pre-inception leading NaNs on their own do not trigger a column drop."""
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=200)
        df = pd.DataFrame(rng.normal(size=(200, 3)), index=idx, columns=["a", "b", "c"])
        df.iloc[:50, 2] = np.nan  # 25% LEADING NaN on "c" → pre-inception
        report = CleaningReport(input_sleeves=["a", "b", "c"])
        result = drop_stale_columns(df, max_missing_pct=0.20, report=report)
        assert "c" not in report.dropped_sleeves
        assert "c" in result.columns
        assert report.pre_inception_rows["c"] == 50
