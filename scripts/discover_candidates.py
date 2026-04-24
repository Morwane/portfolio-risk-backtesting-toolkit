#!/usr/bin/env python
"""Systematic candidate proxy discovery for unavailable universe sleeves.

Probes a curated list of candidate RICs per unavailable sleeve and saves a
full audit CSV.  Run this AFTER validate_universe.py has identified which
sleeves are unavailable.

This script does NOT modify universe.yaml automatically.  Inspect the output
CSV (data/processed/candidate_discovery_report.csv) and promote any confirmed-
working candidates manually.

Usage:
    python scripts/discover_candidates.py [--start 2004-01-01] [--out PATH]
    python scripts/discover_candidates.py --sleeve us_treasury_medium   # probe one sleeve only
    python scripts/discover_candidates.py --demo   # dry-run, no LSEG required

Outputs:
    data/processed/candidate_discovery_report.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from src.utils.logging_utils import configure_root_logger, get_logger

logger = get_logger(__name__)

# ── Candidate definitions ─────────────────────────────────────────────────────
# These are the 4 sleeves that returned "unavailable" in validate_universe.py.
# Each entry lists candidate RICs to probe in priority order.
# Rationale notes are included so results are self-documenting.

SLEEVE_CANDIDATES: Dict[str, Dict] = {
    "global_min_vol": {
        "sleeve_name": "Global Minimum Volatility",
        "asset_class": "Equity",
        "sub_class": "Defensive Equity",
        "current_primary": "MVOL.L",   # promoted from R1 discovery
        "current_fallback": "XDWD.DE", # promoted from R1 discovery
        "target_inception_ceiling": "2013-01-01",
        "candidates": [
            # ── Round 1 results ───────────────────────────────────────────
            {"ric": "ACWV",    "note": "[R1-FAILED] iShares MSCI ACWI Min Vol Factor ETF, inception 2011"},
            {"ric": "EFAV",    "note": "[R1-FAILED] iShares MSCI EAFE Min Vol Factor ETF, inception 2011"},
            {"ric": "SPHD",    "note": "[R1-FAILED] SPDR S&P 500 High Div Low Volatility ETF, inception 2012"},
            {"ric": "FDLO",    "note": "[R1-FAILED] Fidelity Low Volatility Factor ETF, inception 2016"},
            {"ric": "JMIN",    "note": "[R1-FAILED] JPMorgan US Minimum Volatility ETF, inception 2017"},
            {"ric": "LVHD",    "note": "[R1-FAILED] Franklin US Low Vol High Dividend ETF, inception 2015"},
            {"ric": "ONEV",    "note": "[R1-FAILED] SPDR Russell 1000 Low Volatility Focus ETF, inception 2015"},
            {"ric": "XDWD.DE", "note": "[R1-WORKS, PROMOTED-fallback] Xtrackers MSCI World Min Vol ETF (Xetra, EUR), 2014-08-15"},
            {"ric": "MVOL.L",  "note": "[R1-WORKS, PROMOTED-primary] iShares MSCI World Min Vol ETF (London, GBP), 2012-12-13"},
            {"ric": "MINV",    "note": "[R1-FAILED] Franklin LibertyQ Global Equity ETF, inception 2019"},
        ],
    },
    "us_treasury_short": {
        "sleeve_name": "US Treasuries 1-3Y",
        "asset_class": "Sovereign Bond",
        "sub_class": "US Rates Short",
        "current_primary": "BSV",   # promoted from R1 discovery (only confirmed working)
        "current_fallback": "BSV",  # no independent fallback: all other candidates failed LIVE
        "target_inception_ceiling": "2010-01-01",
        "candidates": [
            # ── Round 1 results ───────────────────────────────────────────
            {"ric": "SCHO",   "note": "[R1-FAILED] Schwab Short-Term US Treasury ETF, inception 2010"},
            {"ric": "SPTS",   "note": "[R1-FAILED] SPDR Portfolio Short Term Treasury ETF, inception 2007"},
            {"ric": "BSV",    "note": "[R1-WORKS, PROMOTED-primary] Vanguard Short-Term Bond ETF (1-5Y blend), inception 2007-04-10"},
            {"ric": "IEI",    "note": "[R1-FAILED] iShares 3-7Y Treasury Bond ETF, inception 2007"},
            {"ric": "SCHR",   "note": "[R1-FAILED] Schwab Intermediate US Treasury ETF (3-10Y), inception 2010"},
            {"ric": "FLOT",   "note": "[R1-FAILED] iShares Floating Rate Bond ETF, inception 2011"},
        ],
    },
    "us_treasury_medium": {
        "sleeve_name": "US Treasuries 7-10Y",
        "asset_class": "Sovereign Bond",
        "sub_class": "US Rates Medium",
        "current_primary": "SXRL.DE",   # promoted from R2 discovery
        "current_fallback": "IS04.DE",  # promoted from R2 discovery
        "target_inception_ceiling": "2010-01-01",
        "candidates": [
            # ── Round 1 (all failed) ──────────────────────────────────────
            {"ric": "ITE",    "note": "[R1-FAILED] SPDR Portfolio Intermediate Term Treasury ETF, inception 2007"},
            {"ric": "IEI",    "note": "[R1-FAILED] iShares 3-7Y Treasury Bond ETF, inception 2007"},
            {"ric": "SCHR",   "note": "[R1-FAILED] Schwab Intermediate US Treasury ETF (3-10Y), inception 2010"},
            {"ric": "GOVT",   "note": "[R1-FAILED] iShares US Treasury Bond ETF (all maturities), inception 2012"},
            {"ric": "TLT",    "note": "[R1-FAILED] iShares 20+ Year Treasury Bond ETF, inception 2002"},
            {"ric": "SCHZ",   "note": "[R1-FAILED] Schwab US Aggregate Bond ETF (broader), inception 2011"},
            # ── Round 2: UCITS-listed equivalents of IEF ─────────────────
            {"ric": "IBTM.L",  "note": "[R2-FAILED] iShares $ Treasury Bond 7-10yr UCITS ETF London (GBP)"},
            {"ric": "IBTM.AS", "note": "[R2-FAILED] iShares $ Treasury Bond 7-10yr UCITS ETF Amsterdam (EUR)"},
            {"ric": "IS04.DE", "note": "[R2-WORKS, PROMOTED-fallback] iShares $ Treasury Bond 7-10yr UCITS ETF Xetra (EUR), 2015-01-26"},
            {"ric": "SXRL.L",  "note": "[R2-FAILED] SPDR Bloomberg 7-10Y US Treasury Bond UCITS ETF London"},
            {"ric": "SXRL.DE", "note": "[R2-WORKS, PROMOTED-primary] SPDR Bloomberg 7-10Y US Treasury Bond UCITS ETF Xetra (EUR), 2009-12-01"},
            {"ric": "VDTY.L",  "note": "[R2-WORKS, not promoted] Vanguard U.S. Treasury 7-10Y Bond UCITS ETF London (GBP), 2016-02-25 — shorter history than SXRL.DE"},
            {"ric": "VUST.L",  "note": "[R2-FAILED] Vanguard USD Treasury Bond UCITS ETF London (broad duration)"},
            {"ric": "IBTM",    "note": "[R2-FAILED] iShares $ Treasury Bond 7-10yr UCITS ETF (no suffix)"},
        ],
    },
    "euro_hy_credit": {
        "sleeve_name": "Euro High Yield Credit",
        "asset_class": "Credit",
        "sub_class": "Euro HY Credit",
        "current_primary": "IHYG.L",  # promoted from R1 discovery (earliest history)
        "current_fallback": "SHYU.L", # promoted — USD-hedged, preferred for USD portfolio
        "target_inception_ceiling": "2015-01-01",
        "candidates": [
            # ── Round 1 results ───────────────────────────────────────────
            {"ric": "HYBB",    "note": "[R1-FAILED] iShares Euro HY Corp Bond ETF USD-listed, inception 2019"},
            {"ric": "IHYG.L",  "note": "[R1-WORKS, PROMOTED-primary] iShares Euro HY Corp Bond London (GBP), 2010-09-06"},
            {"ric": "IHYG.DE", "note": "[R1-FAILED] iShares Euro HY Corp Bond Xetra (EUR)"},
            {"ric": "HYLE.DE", "note": "[R1-WORKS] Xtrackers EUR HY Corp Bond Xetra (EUR), 2019-04-30 — too recent"},
            {"ric": "XHY9.DE", "note": "[R1-FAILED] iShares EUR HY Corp Bond UCITS ETF Xetra (EUR)"},
            {"ric": "SHYU.L",  "note": "[R1-WORKS, PROMOTED-fallback] iShares Euro HY Corp Bond London USD-hedged, 2011-09-14"},
            {"ric": "HWGU.L",  "note": "[R1-FAILED] SPDR Bloomberg Euro HY Bond London (USD-hedged)"},
        ],
    },
}


# ── Probe helpers ─────────────────────────────────────────────────────────────

def _probe_rd_recent(
    lib: Any,
    ric: str,
    field: str,
    window_days: int = 30,
    delay: float = 0.4,
) -> Tuple[bool, Optional[str], Optional[str], int]:
    """Quick existence check: probe recent 30-day window.

    Returns (accessible, first_date, last_date, row_count).
    """
    end = datetime.today()
    start = end - timedelta(days=window_days)
    try:
        df = lib.get_history(
            universe=[ric],
            fields=[field],
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1D",
        )
        time.sleep(delay)
        if df is None or df.empty:
            return False, None, None, 0
        df = df.dropna()
        if df.empty:
            return False, None, None, 0
        return True, str(df.index.min().date()), str(df.index.max().date()), len(df)
    except Exception as exc:
        logger.debug("Probe failed %s / %s: %s", ric, field, exc)
        return False, None, None, 0


def _probe_rd_full(
    lib: Any,
    ric: str,
    field: str,
    start_date: str,
    delay: float = 0.5,
) -> Tuple[Optional[str], Optional[str], int]:
    """Full history probe from start_date to today.

    Returns (first_date, last_date, row_count).
    """
    try:
        df = lib.get_history(
            universe=[ric],
            fields=[field],
            start=start_date,
            end=datetime.today().strftime("%Y-%m-%d"),
            interval="1D",
        )
        time.sleep(delay)
        if df is None or df.empty:
            return None, None, 0
        df = df.dropna()
        if df.empty:
            return None, None, 0
        return str(df.index.min().date()), str(df.index.max().date()), len(df)
    except Exception as exc:
        logger.debug("Full range probe failed %s: %s", ric, exc)
        return None, None, 0


def _probe_candidate(
    sleeve_id: str,
    candidate: Dict,
    session: Any,
    start_date: str,
    field: str = "TR.PriceClose",
) -> Dict:
    """Probe a single candidate RIC and return a result row."""
    ric = candidate["ric"]
    lib = session.lib

    row = {
        "sleeve_id": sleeve_id,
        "candidate_ric": ric,
        "candidate_note": candidate["note"],
        "field_tested": field,
        "status": "untested",
        "data_start": None,
        "data_end": None,
        "row_count": 0,
        "error": "",
    }

    logger.info("    Testing %s (%s) ...", ric, candidate["note"][:50])

    # Step 1: quick existence check
    ok, d_start, d_end, n = _probe_rd_recent(lib, ric, field)
    if not ok:
        # Try CF_CLOSE fallback field
        ok, d_start, d_end, n = _probe_rd_recent(lib, ric, "CF_CLOSE")
        if ok:
            row["field_tested"] = "CF_CLOSE"

    if not ok:
        row["status"] = "0_rows_or_error"
        row["error"] = "No data in recent 30-day window for TR.PriceClose and CF_CLOSE"
        logger.warning("    [%s] %s → 0 rows / inaccessible", sleeve_id, ric)
        return row

    # Step 2: full history depth probe
    f_start, f_end, f_count = _probe_rd_full(lib, ric, row["field_tested"], start_date)
    if f_count > 0:
        row["data_start"] = f_start
        row["data_end"] = f_end
        row["row_count"] = f_count
    else:
        row["data_start"] = d_start
        row["data_end"] = d_end
        row["row_count"] = n

    row["status"] = "works"
    logger.info(
        "    [%s] %s → WORKS  %s → %s  (%d rows)",
        sleeve_id, ric, row["data_start"], row["data_end"], row["row_count"],
    )
    return row


# ── Main discovery loop ───────────────────────────────────────────────────────

def run_candidate_discovery(
    session: Any,
    start_date: str = "2004-01-01",
    output_path: Optional[Path] = None,
    sleeve_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Probe candidates for unavailable sleeves.

    Args:
        sleeve_filter: If provided, only probe these sleeve IDs. None = all sleeves.

    Returns a DataFrame with one row per (sleeve, candidate) pair.
    """
    candidates_to_probe = {
        k: v for k, v in SLEEVE_CANDIDATES.items()
        if sleeve_filter is None or k in sleeve_filter
    }

    if sleeve_filter:
        unknown = [s for s in sleeve_filter if s not in SLEEVE_CANDIDATES]
        if unknown:
            logger.warning("Unknown sleeve IDs ignored: %s", unknown)
            logger.info("Known sleeves: %s", list(SLEEVE_CANDIDATES.keys()))

    rows = []
    total_candidates = sum(len(v["candidates"]) for v in candidates_to_probe.values())
    logger.info(
        "Candidate discovery: %d sleeve(s), %d candidates total",
        len(candidates_to_probe),
        total_candidates,
    )

    for sleeve_id, sleeve_info in candidates_to_probe.items():
        logger.info(
            "\n[%s] %s  (currently unavailable: %s / %s)",
            sleeve_id,
            sleeve_info["sleeve_name"],
            sleeve_info["current_primary"],
            sleeve_info["current_fallback"],
        )
        for candidate in sleeve_info["candidates"]:
            result = _probe_candidate(sleeve_id, candidate, session, start_date)
            result["sleeve_name"] = sleeve_info["sleeve_name"]
            result["asset_class"] = sleeve_info["asset_class"]
            result["current_primary"] = sleeve_info["current_primary"]
            result["current_fallback"] = sleeve_info["current_fallback"]
            result["target_inception_ceiling"] = sleeve_info["target_inception_ceiling"]
            rows.append(result)

    report = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "sleeve_id", "sleeve_name", "asset_class",
        "current_primary", "current_fallback",
        "candidate_ric", "candidate_note",
        "status", "field_tested",
        "data_start", "data_end", "row_count",
        "target_inception_ceiling", "error",
    ]
    report = report[[c for c in col_order if c in report.columns]]

    save_path = output_path or Path("data/processed/candidate_discovery_report.csv")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(save_path, index=False)
    logger.info("\nCandidate discovery report saved: %s", save_path)

    return report


def _print_discovery_summary(report: pd.DataFrame) -> None:
    """Print a human-readable summary of discovery results."""
    logger.info("\n%s", "=" * 70)
    logger.info("CANDIDATE DISCOVERY SUMMARY")
    logger.info("=" * 70)

    working = report[report["status"] == "works"]

    for sleeve_id in report["sleeve_id"].unique():
        sleeve_rows = report[report["sleeve_id"] == sleeve_id]
        sleeve_name = sleeve_rows["sleeve_name"].iloc[0]
        works = sleeve_rows[sleeve_rows["status"] == "works"]

        logger.info("\n  [%s] %s", sleeve_id, sleeve_name)
        if works.empty:
            logger.warning("    ✗  No working candidates found.")
        else:
            for _, row in works.iterrows():
                logger.info(
                    "    ✓  %-10s  %s → %s  (%d rows)",
                    row["candidate_ric"],
                    row["data_start"] or "?",
                    row["data_end"] or "?",
                    row["row_count"],
                )
            # Recommend first working candidate with earliest data
            best = works.sort_values("data_start").iloc[0]
            logger.info(
                "    → Recommended: %s  (earliest data: %s)",
                best["candidate_ric"],
                best["data_start"],
            )

        failed = sleeve_rows[sleeve_rows["status"] != "works"]
        if not failed.empty:
            logger.info(
                "    Failed: %s",
                ", ".join(failed["candidate_ric"].tolist()),
            )

    n_sleeves_resolved = len(working["sleeve_id"].unique())
    n_sleeves_total = len(report["sleeve_id"].unique())
    logger.info(
        "\n  Resolved: %d / %d unavailable sleeves have at least one working candidate.",
        n_sleeves_resolved,
        n_sleeves_total,
    )
    if n_sleeves_resolved < n_sleeves_total:
        still_missing = [
            sid for sid in report["sleeve_id"].unique()
            if sid not in working["sleeve_id"].values
        ]
        logger.warning(
            "  Still unavailable: %s\n"
            "  These sleeves will continue to be excluded from live runs.\n"
            "  Portfolio integrity will remain DEGRADED if their target weight is significant.",
            still_missing,
        )


def _mock_discovery_report() -> pd.DataFrame:
    """Return a synthetic discovery report for demo / CI usage."""
    rows = []
    for sleeve_id, sleeve_info in SLEEVE_CANDIDATES.items():
        for i, candidate in enumerate(sleeve_info["candidates"]):
            rows.append({
                "sleeve_id": sleeve_id,
                "sleeve_name": sleeve_info["sleeve_name"],
                "asset_class": sleeve_info["asset_class"],
                "current_primary": sleeve_info["current_primary"],
                "current_fallback": sleeve_info["current_fallback"],
                "candidate_ric": candidate["ric"],
                "candidate_note": candidate["note"],
                "status": "works" if i == 0 else "0_rows_or_error",
                "field_tested": "TR.PriceClose",
                "data_start": "2011-01-03" if i == 0 else None,
                "data_end": "2026-04-22" if i == 0 else None,
                "row_count": 3850 if i == 0 else 0,
                "target_inception_ceiling": sleeve_info["target_inception_ceiling"],
                "error": "" if i == 0 else "No data in recent 30-day window [DEMO]",
            })
    return pd.DataFrame(rows)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover candidate proxies for unavailable universe sleeves."
    )
    parser.add_argument(
        "--start",
        default="2004-01-01",
        help="Earliest history to probe (default: 2004-01-01)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: data/processed/candidate_discovery_report.csv)",
    )
    parser.add_argument(
        "--sleeve",
        nargs="+",
        metavar="SLEEVE_ID",
        default=None,
        help=(
            "Probe only these sleeve IDs (space-separated). "
            f"Known IDs: {', '.join(SLEEVE_CANDIDATES.keys())}. "
            "Default: probe all sleeves."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Skip LSEG and generate a synthetic discovery report (for testing)",
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


def main():
    args = parse_args()
    configure_root_logger(level=args.log_level)

    logger.info("=" * 70)
    logger.info("Portfolio Risk Toolkit — Candidate Proxy Discovery")
    logger.info("=" * 70)

    sleeve_filter = args.sleeve or None
    sleeves_to_log = sleeve_filter if sleeve_filter else list(SLEEVE_CANDIDATES.keys())
    logger.info("Probing %d sleeve(s): %s", len(sleeves_to_log), sleeves_to_log)

    out_path = Path(args.out) if args.out else None

    if args.demo:
        logger.info("Running in DEMO mode — no live LSEG probe.")
        report = _mock_discovery_report()
        if sleeve_filter:
            report = report[report["sleeve_id"].isin(sleeve_filter)].reset_index(drop=True)
        save_path = out_path or Path("data/processed/candidate_discovery_report.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(save_path, index=False)
        logger.info("Mock discovery report saved: %s", save_path)
        _print_discovery_summary(report)
        return

    from src.data.lseg_session import get_session, LSEGUnavailableError

    try:
        session = get_session(session_type=args.session_type)
    except LSEGUnavailableError as exc:
        logger.error(
            "Cannot open LSEG session: %s\n"
            "Tip: Run with --demo to test the script without LSEG.",
            exc,
        )
        sys.exit(1)

    try:
        report = run_candidate_discovery(
            session,
            start_date=args.start,
            output_path=out_path,
            sleeve_filter=sleeve_filter,
        )
    finally:
        try:
            session.close()
        except Exception:
            pass

    _print_discovery_summary(report)

    working = report[report["status"] == "works"]
    if not working.empty:
        logger.info(
            "\nNext step: review data/processed/candidate_discovery_report.csv "
            "and promote confirmed-working RICs into config/universe.yaml manually."
        )


if __name__ == "__main__":
    main()
