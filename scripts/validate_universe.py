#!/usr/bin/env python
"""Universe validation script.

Probes every sleeve in config/universe.yaml against LSEG to determine:
  - Which RICs are accessible (primary vs fallback)
  - Which price fields work reliably
  - What date coverage is available

Run this BEFORE run_backtest.py to avoid silent data gaps.

Usage:
    python scripts/validate_universe.py [--start 2004-01-01] [--demo]

Outputs:
    data/processed/universe_validation.csv
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parents[1]))

import yaml

from src.utils.logging_utils import configure_root_logger, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LSEG universe accessibility.")
    parser.add_argument(
        "--start",
        default="2004-01-01",
        help="Earliest history to probe (default: 2004-01-01)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Skip LSEG and print a mock validation report (for testing)",
    )
    parser.add_argument(
        "--session-type",
        default="platform",
        choices=["platform", "deployed"],
        help="LSEG session type (default: platform)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _mock_validation_report():
    """Return a synthetic validation report for demo / CI usage."""
    import pandas as pd
    from src.data.mapping import get_sleeve_list

    sleeves = get_sleeve_list()
    rows = []
    for s in sleeves:
        rows.append({
            "sleeve_id": s["id"],
            "sleeve_name": s["name"],
            "asset_class": s["asset_class"],
            "primary_ric": s["primary_ric"],
            "fallback_ric": s["fallback_ric"],
            "recommended_ric": s["primary_ric"],
            "working_field": "TR.PriceClose",
            "data_start": "2007-01-02",
            "data_end": "2024-12-31",
            "inception_date": s.get("inception_date", ""),
            "status": "primary_ok",
            "note": "[DEMO MODE - no real LSEG probe]",
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    configure_root_logger(level=args.log_level)

    logger.info("=" * 60)
    logger.info("Portfolio Risk Toolkit — Universe Validation")
    logger.info("=" * 60)

    if args.demo:
        logger.info("Running in DEMO mode — skipping live LSEG probe.")
        report = _mock_validation_report()
        out_path = Path("data/processed/universe_validation.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        logger.info("Mock validation report saved to %s", out_path)
        _print_summary(report)
        return

    # Live LSEG validation
    from src.data.lseg_session import get_session, LSEGUnavailableError
    from src.data.discovery import run_universe_validation

    try:
        session = get_session(session_type=args.session_type)
    except LSEGUnavailableError as exc:
        logger.error(
            "Cannot open LSEG session: %s\n"
            "Tip: Run with --demo to generate a mock report, or check your\n"
            "     LSEG installation and credentials.",
            exc,
        )
        sys.exit(1)

    try:
        report = run_universe_validation(session, start_date=args.start)
    finally:
        try:
            session.close()
        except Exception:
            pass

    _print_summary(report)


def _print_summary(report):
    import pandas as pd
    logger.info("\n%s", "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("%s", "=" * 60)

    for _, row in report.iterrows():
        status_icon = "✓" if row["status"] == "primary_ok" else ("⚠" if row["status"] == "fallback_used" else "✗")
        logger.info(
            "  %s  %-30s  %-12s  %-10s  %s → %s",
            status_icon,
            row["sleeve_name"],
            row["recommended_ric"] or "N/A",
            row["status"],
            row.get("data_start", "?"),
            row.get("data_end", "?"),
        )

    n_ok = (report["status"] == "primary_ok").sum()
    n_fb = (report["status"] == "fallback_used").sum()
    n_fail = (report["status"] == "unavailable").sum()
    logger.info("\n  Total: %d ok | %d fallback | %d unavailable", n_ok, n_fb, n_fail)

    if n_fail > 0:
        logger.warning(
            "\n  %d sleeve(s) are unavailable. "
            "They will be excluded from the backtest.\n"
            "  Consider enabling demo_mode in config/settings.yaml.",
            n_fail,
        )


if __name__ == "__main__":
    main()
