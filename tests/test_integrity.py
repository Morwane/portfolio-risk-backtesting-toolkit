"""Tests for portfolio integrity controls."""

import pytest
from src.portfolio.integrity import (
    check_portfolio_integrity,
    STATUS_VALID,
    STATUS_DEGRADED,
    STATUS_INVALID,
)


def _weights(*ids):
    """Return equal weights for given sleeve IDs."""
    return {sid: 1.0 / len(ids) for sid in ids}


class TestCheckPortfolioIntegrity:

    def test_all_sleeves_present_valid(self):
        target = {"us_large_cap": 0.60, "us_treasury_medium": 0.40}
        executed = {"us_large_cap": 0.60, "us_treasury_medium": 0.40}
        report = check_portfolio_integrity(
            "test", target, executed,
            thresholds={"min_sleeves_remaining": 2, "min_equity_count": 1, "min_bond_count": 1},
        )
        assert report.run_status == STATUS_VALID
        assert report.n_dropped_sleeves == 0
        assert report.total_dropped_weight == 0.0

    def test_small_drop_degraded(self):
        # Drop 20% weight → DEGRADED (above 15% threshold)
        target = {
            "us_large_cap": 0.60,
            "us_treasury_short": 0.10,
            "us_treasury_medium": 0.10,
            "gold": 0.20,
        }
        executed = {"us_large_cap": 0.75, "gold": 0.25}  # renorm after 20% dropped
        report = check_portfolio_integrity(
            "test", target, executed,
            thresholds={"min_sleeves_remaining": 2, "min_equity_count": 1, "min_bond_count": 0},
        )
        assert report.run_status == STATUS_DEGRADED
        assert report.n_dropped_sleeves == 2

    def test_massive_drop_invalid(self):
        # Drop 50% → INVALID
        target = {f"sleeve_{i}": 0.1 for i in range(10)}
        # Keep only 5 sleeves
        executed = {f"sleeve_{i}": 0.2 for i in range(5)}
        report = check_portfolio_integrity("test", target, executed)
        assert report.run_status == STATUS_INVALID

    _t2 = {"min_sleeves_remaining": 2, "min_equity_count": 0, "min_bond_count": 0}

    def test_dropped_sleeve_ids_correct(self):
        target = {"a": 0.5, "b": 0.3, "c": 0.2}
        executed = {"a": 0.625, "b": 0.375}  # c dropped
        report = check_portfolio_integrity("test", target, executed, thresholds=self._t2)
        assert "c" in report.dropped_sleeve_ids
        assert len(report.dropped_sleeve_ids) == 1

    def test_sleeve_records_count(self):
        target = {"a": 0.5, "b": 0.5}
        executed = {"a": 0.5, "b": 0.5}
        report = check_portfolio_integrity("test", target, executed, thresholds=self._t2)
        assert len(report.sleeve_records) == 2

    def test_renorm_factor_correct(self):
        # Drop 20% weight → retained 80% → renorm factor = 1/0.8 = 1.25
        target = {"a": 0.40, "b": 0.40, "c": 0.20}
        executed = {"a": 0.50, "b": 0.50}
        report = check_portfolio_integrity("test", target, executed, thresholds=self._t2)
        assert abs(report.total_renorm_factor - 1.25) < 0.01

    def test_flags_populated_on_degraded(self):
        target = {"a": 0.50, "b": 0.30, "c": 0.20}
        executed = {"a": 0.625, "b": 0.375}
        report = check_portfolio_integrity("test", target, executed, thresholds=self._t2)
        assert len(report.flags) > 0

    def test_equal_weight_no_target(self):
        """Dynamic equal-weight portfolios have no fixed target."""
        from src.portfolio.integrity import check_all_portfolios
        all_configs = {"equal_weight": {"weights": None, "mode": "equal_weight"}}
        weights_dict = {"equal_weight": _weights("a", "b", "c")}
        reports = check_all_portfolios(["equal_weight"], all_configs, weights_dict)
        assert reports["equal_weight"].run_status == STATUS_VALID


class TestIntegrityOutputs:

    _t2 = {"min_sleeves_remaining": 2, "min_equity_count": 0, "min_bond_count": 0}

    def test_to_sleeve_df_columns(self):
        target = {"us_large_cap": 0.60, "us_treasury_medium": 0.40}
        executed = {"us_large_cap": 0.60, "us_treasury_medium": 0.40}
        report = check_portfolio_integrity("test", target, executed, thresholds=self._t2)
        df = report.to_sleeve_df()
        assert "sleeve_id" in df.columns
        assert "target_weight" in df.columns
        assert "executed_weight" in df.columns
        assert "status" in df.columns

    def test_ac_drift_df_columns(self):
        target = {"us_large_cap": 0.60, "us_treasury_medium": 0.40}
        executed = {"us_large_cap": 0.60, "us_treasury_medium": 0.40}
        report = check_portfolio_integrity("test", target, executed, thresholds=self._t2)
        df = report.to_ac_drift_df()
        assert "asset_class" in df.columns
        assert "target_weight" in df.columns
        assert "executed_weight" in df.columns
