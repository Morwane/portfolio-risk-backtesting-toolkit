"""Data loading layer: LSEG fetch + local cache + demo fallback.

Entry point: load_prices(). It returns a tidy DataFrame of adjusted close
prices for all sleeves in the universe, handling:
  - LSEG live fetch (via rd or eikon backend)
  - local parquet cache (skips API if cache is fresh)
  - demo mode (synthetic data from data/sample/)
  - per-sleeve fallback RIC logic from discovery validation report
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.discovery import load_validation_report, get_recommended_ric_map, get_recommended_field_map
from src.data.mapping import get_sleeve_list, get_name_map
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path("data/processed")
_DEMO_PATH = Path("data/sample/demo_prices.parquet")


# ── Internal: LSEG fetch helpers ─────────────────────────────────────────────

def _fetch_rd(
    lib: Any,
    rics: List[str],
    field: str,
    start: str,
    end: str,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch a batch of RICs using refinitiv.data / lseg.data.

    Returns a DataFrame indexed by date, columns = RICs.
    """
    try:
        df = lib.get_history(
            universe=rics,
            fields=[field],
            start=start,
            end=end,
            interval="1D",
        )
        time.sleep(delay)
        if df is None or df.empty:
            return pd.DataFrame()

        # get_history can return MultiIndex columns when multiple fields
        if isinstance(df.columns, pd.MultiIndex):
            # Shape: (date, (ric, field)) -> keep only the field level
            df = df.xs(field, level=1, axis=1)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    except Exception as exc:
        logger.warning("RD batch fetch failed for RICs %s: %s", rics, exc)
        return pd.DataFrame()


def _fetch_eikon(
    ek: Any,
    rics: List[str],
    start: str,
    end: str,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch using the Eikon Python API.

    Returns a DataFrame indexed by date, columns = RICs.
    """
    try:
        df, err = ek.get_timeseries(
            rics=rics,
            start_date=start,
            end_date=end,
            fields=["CLOSE"],
            interval="daily",
        )
        time.sleep(delay)
        if err or df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        # Eikon returns a MultiIndex (ric, field) or flat if single ric
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs("CLOSE", level=1, axis=1)
        else:
            df.columns = rics[:1]
        return df
    except Exception as exc:
        logger.warning("Eikon batch fetch failed for RICs %s: %s", rics, exc)
        return pd.DataFrame()


def _fetch_single_sleeve(
    sleeve: Dict,
    ric_map: Dict[str, str],
    field_map: Dict[str, str],
    session: Any,
    start: str,
    end: str,
) -> Tuple[str, pd.Series]:
    """Fetch one sleeve's price series.

    Falls back to fallback_ric if the validation report recommends it.

    Returns:
        (sleeve_id, price_series)
    """
    sid = sleeve["id"]
    ric = ric_map.get(sid, sleeve["primary_ric"])
    field = field_map.get(sid, "TR.PriceClose")
    backend = getattr(session, "backend", "unknown")
    lib = session.lib

    if backend == "rd":
        df = _fetch_rd(lib, [ric], field, start, end)
        if df.empty:
            # Try fallback RIC
            fallback = sleeve["fallback_ric"]
            if fallback != ric:
                logger.warning("[%s] Primary RIC %s empty, trying fallback %s", sid, ric, fallback)
                df = _fetch_rd(lib, [fallback], field, start, end)
    elif backend == "eikon":
        df = _fetch_eikon(lib, [ric], start, end)
        if df.empty:
            fallback = sleeve["fallback_ric"]
            if fallback != ric:
                logger.warning("[%s] Primary RIC %s empty, trying fallback %s", sid, ric, fallback)
                df = _fetch_eikon(lib, [fallback], start, end)
    else:
        logger.error("[%s] Unknown backend: %s", sid, backend)
        return sid, pd.Series(dtype=float, name=sid)

    if df.empty or ric not in df.columns and len(df.columns) == 0:
        logger.warning("[%s] No data returned.", sid)
        return sid, pd.Series(dtype=float, name=sid)

    series = df.iloc[:, 0].rename(sid)
    return sid, series


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(start: str, end: str) -> Path:
    end_str = end or "today"
    return _CACHE_DIR / f"prices_{start}_{end_str}.parquet"


def _cache_is_fresh(path: Path, max_age_days: int = 1) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    return age.days < max_age_days


def _save_cache(prices: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    logger.debug("Price cache saved to %s", path)


def _load_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    logger.info("Loaded prices from cache: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_prices(
    session: Optional[Any] = None,
    start: str = "2007-01-01",
    end: Optional[str] = None,
    demo_mode: bool = False,
    use_cache: bool = True,
    cache_max_age_days: int = 1,
    validation_report: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Main entry point: return adjusted close prices for all universe sleeves.

    Data source priority:
      1. Local parquet cache (if fresh and use_cache=True)
      2. Demo synthetic data (if demo_mode=True)
      3. LSEG live fetch via session

    Args:
        session: Active LSEG session adapter. Required unless demo_mode=True
                 or cache is valid.
        start: Start date string (YYYY-MM-DD).
        end: End date string. None defaults to today.
        demo_mode: If True, load synthetic demo data.
        use_cache: If True, read from/write to local parquet cache.
        cache_max_age_days: Cache freshness threshold.
        validation_report: Optional pre-loaded validation report DataFrame.
                           If None and not demo_mode, attempts to load from
                           data/processed/universe_validation.csv.

    Returns:
        DataFrame indexed by date (DatetimeIndex), columns = sleeve_ids,
        values = adjusted close prices.
    """
    end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    # ── Demo mode ─────────────────────────────────────────────────────────
    if demo_mode:
        if not _DEMO_PATH.exists():
            raise FileNotFoundError(
                f"Demo data not found at {_DEMO_PATH}. "
                "Run scripts/build_demo_outputs.py first."
            )
        logger.info("Loading synthetic demo prices from %s", _DEMO_PATH)
        prices = pd.read_parquet(_DEMO_PATH)
        prices.index = pd.to_datetime(prices.index)
        return prices.loc[start:end_str]

    # ── Cache check ────────────────────────────────────────────────────────
    cache = _cache_path(start, end_str)
    if use_cache and _cache_is_fresh(cache, cache_max_age_days):
        return _load_cache(cache)

    # ── Live LSEG fetch ────────────────────────────────────────────────────
    if session is None:
        raise ValueError(
            "session is required for live LSEG fetch. "
            "Pass demo_mode=True or ensure cache is valid."
        )

    # Load validation report to get recommended RICs and fields
    report = validation_report or load_validation_report()
    ric_map: Dict[str, str] = get_recommended_ric_map(report) if report is not None else {}
    field_map: Dict[str, str] = get_recommended_field_map(report) if report is not None else {}

    sleeves = get_sleeve_list()
    all_series: Dict[str, pd.Series] = {}

    logger.info("Fetching %d sleeves from LSEG (%s to %s) ...", len(sleeves), start, end_str)
    for sleeve in sleeves:
        sid, series = _fetch_single_sleeve(sleeve, ric_map, field_map, session, start, end_str)
        if not series.empty:
            all_series[sid] = series
        else:
            logger.warning("[%s] Excluded: no data returned.", sleeve["id"])

    if not all_series:
        raise RuntimeError(
            "No price data returned for any sleeve. "
            "Check LSEG connection or enable demo_mode."
        )

    prices = pd.DataFrame(all_series)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices.sort_index(inplace=True)

    logger.info("Loaded %d sleeves, %d rows (%s to %s).",
                len(prices.columns), len(prices),
                prices.index.min().date(), prices.index.max().date())

    if use_cache:
        _save_cache(prices, cache)

    return prices
