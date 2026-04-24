"""Tests for benchmark-relative analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.benchmark import (
    compute_tracking_error,
    compute_information_ratio,
    compute_beta,
    build_benchmark_report,
)


def _random_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


class TestTrackingError:

    def test_identical_series_zero_te(self):
        r = _random_returns()
        te = compute_tracking_error(r, r)
        assert te < 1e-8

    def test_independent_series_positive_te(self):
        r1 = _random_returns(seed=1)
        r2 = _random_returns(seed=2)
        te = compute_tracking_error(r1, r2)
        assert te > 0.05  # at least 5% annualised TE

    def test_insufficient_overlap_returns_nan(self):
        r1 = _random_returns(n=10)
        r2 = _random_returns(n=10)
        te = compute_tracking_error(r1, r2)
        assert np.isnan(te)

    def test_annualisation(self):
        r1 = _random_returns(seed=1)
        r2 = _random_returns(seed=2)
        te_ann = compute_tracking_error(r1, r2, annualise=True)
        te_raw = compute_tracking_error(r1, r2, annualise=False)
        assert abs(te_ann / te_raw - np.sqrt(252)) < 0.01


class TestInformationRatio:

    def test_positive_active_return_positive_ir(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-01", periods=500)
        bm = pd.Series(rng.normal(0.0002, 0.01, 500), index=idx)
        port = bm + rng.normal(0.0004, 0.002, 500)  # consistent alpha
        ir = compute_information_ratio(port, bm)
        assert ir > 0

    def test_identical_series_nan_ir(self):
        r = _random_returns()
        ir = compute_information_ratio(r, r)
        assert np.isnan(ir)


class TestBeta:

    def test_beta_one_for_identical(self):
        r = _random_returns()
        beta = compute_beta(r, r)
        assert abs(beta - 1.0) < 0.001

    def test_beta_zero_for_independent(self):
        rng = np.random.default_rng(99)
        idx = pd.bdate_range("2020-01-01", periods=500)
        r1 = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        r2 = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        beta = compute_beta(r1, r2)
        assert abs(beta) < 0.15  # should be near 0 for truly independent series


class TestBuildBenchmarkReport:

    def test_report_structure(self):
        r1 = _random_returns(seed=1)
        r2 = _random_returns(seed=2)
        report = build_benchmark_report(
            {"port_a": r1, "port_b": r2},
            benchmark_map={"port_a": "port_b"},
        )
        assert "portfolio_id" in report.columns
        assert "tracking_error_ann" in report.columns
        assert "information_ratio" in report.columns

    def test_no_benchmark_assigned(self):
        r = _random_returns()
        report = build_benchmark_report({"solo": r}, benchmark_map={})
        row = report[report["portfolio_id"] == "solo"].iloc[0]
        assert row["note"] == "No benchmark assigned or benchmark not in run."

    def test_values_not_null_for_mapped_portfolios(self):
        r1 = _random_returns(seed=1)
        r2 = _random_returns(seed=2)
        report = build_benchmark_report(
            {"p1": r1, "p2": r2},
            benchmark_map={"p1": "p2"},
        )
        row = report[report["portfolio_id"] == "p1"].iloc[0]
        assert row["tracking_error_ann"] is not None
        assert row["information_ratio"] is not None
