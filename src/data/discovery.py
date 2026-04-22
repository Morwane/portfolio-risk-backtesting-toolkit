"""LSEG universe discovery and validation layer.

This module answers the questions the brief mandates before any analysis:
  - Which instruments are actually accessible?
  - Which fields work reliably?
  - What date coverage is available?
  - Which proxies should be substituted?

Results are written to data/processed/universe_validation.csv for auditing.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.mapping import get_sleeve_list
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_VALIDATION_PATH = Path("data/processed/universe_validation.csv")

# Short test window for field existence checks (avoids pulling years of data)
_TEST_WINDOW_DAYS = 30
_FIELD_PRIORITY = ["TR.PriceClose", "CF_CLOSE", "CLOSE"]


def _probe_rd(lib: Any, ric: str, field: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Probe using the refinitiv.data / lseg.data API.

    Returns:
        (success, first_date_str, last_date_str)
    """
    end = datetime.today()
    start = end - timedelta(days=_TEST_WINDOW_DAYS)
    try:
        df = lib.get_history(
            universe=[ric],
            fields=[field],
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1D",
        )
        if df is None or df.empty:
            return False, None, None
        df = df.dropna()
        if df.empty:
            return False, None, None
        return True, str(df.index.min().date()), str(df.index.max().date())
    except Exception as exc:
        logger.debug("RD probe failed for %s / %s: %s", ric, field, exc)
        return False, None, None


def _probe_rd_full_range(lib: Any, ric: str, field: str, start_date: str) -> Tuple[Optional[str], Optional[str]]:
    """Probe full date range for a confirmed accessible RIC."""
    try:
        df = lib.get_history(
            universe=[ric],
            fields=[field],
            start=start_date,
            end=datetime.today().strftime("%Y-%m-%d"),
            interval="1D",
        )
        if df is None or df.empty:
            return None, None
        df = df.dropna()
        if df.empty:
            return None, None
        return str(df.index.min().date()), str(df.index.max().date())
    except Exception as exc:
        logger.debug("RD full range probe failed for %s: %s", ric, exc)
        return None, None


def _probe_eikon(ek: Any, ric: str, field: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Probe using the Eikon Python API."""
    end = datetime.today()
    start = end - timedelta(days=_TEST_WINDOW_DAYS)
    try:
        df, err = ek.get_timeseries(
            rics=[ric],
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            fields=["CLOSE"],
            interval="daily",
        )
        if err or df is None or df.empty:
            return False, None, None
        df = df.dropna()
        if df.empty:
            return False, None, None
        return True, str(df.index.min().date()), str(df.index.max().date())
    except Exception as exc:
        logger.debug("Eikon probe failed for %s: %s", ric, exc)
        return False, None, None


def validate_single_ric(
    ric: str,
    session: Any,
    start_date: str = "2004-01-01",
    delay: float = 0.5,
) -> Dict:
    """Validate a single RIC and return a status dict.

    Args:
        ric: The RIC string to probe.
        session: Active LSEG session adapter (from lseg_session.get_session).
        start_date: Earliest desired history start.
        delay: Seconds to sleep between API calls (rate limiting).

    Returns:
        Dict with keys: ric, accessible, working_field, data_start, data_end,
        row_count_test_window, note.
    """
    result = {
        "ric": ric,
        "accessible": False,
        "working_field": None,
        "data_start": None,
        "data_end": None,
        "note": "",
    }

    backend = getattr(session, "backend", "unknown")
    lib = session.lib

    if backend == "rd":
        fields_to_try = _FIELD_PRIORITY
        for field in fields_to_try:
            success, d_start, d_end = _probe_rd(lib, ric, field)
            time.sleep(delay)
            if success:
                result["accessible"] = True
                result["working_field"] = field
                full_start, full_end = _probe_rd_full_range(lib, ric, field, start_date)
                result["data_start"] = full_start or d_start
                result["data_end"] = full_end or d_end
                break
        if not result["accessible"]:
            result["note"] = "No usable field found in RD API"

    elif backend == "eikon":
        success, d_start, d_end = _probe_eikon(lib, ric, "CLOSE")
        time.sleep(delay)
        if success:
            result["accessible"] = True
            result["working_field"] = "CLOSE"
            result["data_start"] = d_start
            result["data_end"] = d_end
        else:
            result["note"] = "RIC not accessible via Eikon API"

    else:
        result["note"] = f"Unknown backend: {backend}"

    return result


def run_universe_validation(
    session: Any,
    start_date: str = "2004-01-01",
    output_path: Optional[Path] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Validate every sleeve in the universe config and produce a report.

    For each sleeve, probes both primary and fallback RICs.
    Recommends which RIC to use based on accessibility and history length.

    Args:
        session: Active LSEG session adapter.
        start_date: Earliest desired history.
        output_path: Where to save the CSV report. Defaults to
                     ``data/processed/universe_validation.csv``.
        delay: Rate-limiting delay between API calls.

    Returns:
        DataFrame with one row per sleeve, columns: sleeve_id, sleeve_name,
        primary_ric, fallback_ric, recommended_ric, working_field,
        data_start, data_end, status, note.
    """
    sleeves = get_sleeve_list()
    rows = []

    logger.info("Starting universe validation for %d sleeves ...", len(sleeves))

    for sleeve in sleeves:
        sid = sleeve["id"]
        name = sleeve["name"]
        primary = sleeve["primary_ric"]
        fallback = sleeve["fallback_ric"]

        logger.info("  Probing [%s]  primary=%s  fallback=%s", sid, primary, fallback)

        primary_result = validate_single_ric(primary, session, start_date, delay)
        fallback_result = validate_single_ric(fallback, session, start_date, delay)

        # Determine recommendation
        if primary_result["accessible"]:
            recommended = primary
            working_field = primary_result["working_field"]
            data_start = primary_result["data_start"]
            data_end = primary_result["data_end"]
            status = "primary_ok"
            note = ""
        elif fallback_result["accessible"]:
            recommended = fallback
            working_field = fallback_result["working_field"]
            data_start = fallback_result["data_start"]
            data_end = fallback_result["data_end"]
            status = "fallback_used"
            note = f"Primary {primary} inaccessible; using fallback {fallback}"
            logger.warning("  [%s] primary %s failed, using fallback %s", sid, primary, fallback)
        else:
            recommended = None
            working_field = None
            data_start = None
            data_end = None
            status = "unavailable"
            note = f"Both {primary} and {fallback} inaccessible. Use demo data."
            logger.error("  [%s] BOTH primary and fallback inaccessible.", sid)

        rows.append(
            {
                "sleeve_id": sid,
                "sleeve_name": name,
                "asset_class": sleeve.get("asset_class", ""),
                "primary_ric": primary,
                "fallback_ric": fallback,
                "recommended_ric": recommended,
                "working_field": working_field,
                "data_start": data_start,
                "data_end": data_end,
                "inception_date": sleeve.get("inception_date", ""),
                "status": status,
                "note": note,
            }
        )

    report = pd.DataFrame(rows)

    # Persist
    save_path = output_path or _VALIDATION_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(save_path, index=False)
    logger.info("Universe validation report saved to %s", save_path)

    n_ok = (report["status"] == "primary_ok").sum()
    n_fallback = (report["status"] == "fallback_used").sum()
    n_unavail = (report["status"] == "unavailable").sum()
    logger.info(
        "Validation summary: %d primary_ok | %d fallback_used | %d unavailable",
        n_ok, n_fallback, n_unavail,
    )

    return report


def load_validation_report(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load a previously saved validation report if it exists."""
    p = Path(path or _VALIDATION_PATH)
    if not p.exists():
        return None
    return pd.read_csv(p)


def get_recommended_ric_map(report: pd.DataFrame) -> Dict[str, str]:
    """Return {sleeve_id: recommended_ric} for available sleeves only."""
    available = report[report["status"] != "unavailable"]
    return dict(zip(available["sleeve_id"], available["recommended_ric"]))


def get_recommended_field_map(report: pd.DataFrame) -> Dict[str, str]:
    """Return {sleeve_id: working_field} for available sleeves only."""
    available = report[report["status"] != "unavailable"]
    return dict(zip(available["sleeve_id"], available["working_field"]))
