"""Run manifest and warnings collector.

Every backtest run emits two JSON files:
  outputs/run_manifest.json  — what ran, when, settings, data sources, integrity status
  outputs/run_warnings.json  — data quality, fallback, and integrity warnings
"""

from __future__ import annotations

import datetime
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MANIFEST_PATH = Path("outputs/run_manifest.json")
_WARNINGS_PATH = Path("outputs/run_warnings.json")


class RunManifest:
    """Accumulate run metadata and persist as a JSON manifest."""

    def __init__(self) -> None:
        self._start = datetime.datetime.now()
        self.run_id: str = self._start.strftime("%Y%m%d_%H%M%S")
        self.run_datetime: str = self._start.isoformat()
        self.python_version: str = platform.python_version()
        self.mode: str = "unknown"
        self.data_start: Optional[str] = None
        self.data_end: Optional[str] = None
        self.common_history_start: Optional[str] = None
        self.universe_total: int = 0
        self.available_sleeves: List[str] = []
        self.dropped_sleeves: List[str] = []
        self.portfolios: List[str] = []
        self.settings: Dict[str, Any] = {}
        self.ric_mapping: Dict[str, Dict] = {}
        self.integrity_status: Dict[str, str] = {}
        self.runtime_seconds: Optional[float] = None

    # ── Setters ───────────────────────────────────────────────────────────

    def set_mode(self, demo: bool) -> None:
        self.mode = "DEMO" if demo else "LIVE"

    def set_data_range(
        self,
        start: Any,
        end: Any,
        common_start: Any,
    ) -> None:
        self.data_start = str(start)
        self.data_end = str(end) if end else "today"
        self.common_history_start = str(common_start) if common_start else None

    def set_universe(self, available: List[str], total: int) -> None:
        self.available_sleeves = list(available)
        self.universe_total = total

    def set_dropped_sleeves(self, dropped: List[str]) -> None:
        self.dropped_sleeves = list(dropped)

    def set_portfolios(self, portfolio_ids: List[str]) -> None:
        self.portfolios = list(portfolio_ids)

    def set_settings(self, raw_settings: dict) -> None:
        bt = raw_settings.get("backtesting", {})
        an = raw_settings.get("analytics", {})
        self.settings = {
            "rebalancing_frequency": bt.get("rebalancing_frequency"),
            "transaction_costs_bps": bt.get("transaction_costs_bps"),
            "base_currency": bt.get("base_currency"),
            "risk_free_rate_annual": an.get("risk_free_rate_annual"),
            "var_confidence_levels": an.get("var_confidence_levels"),
            "rolling_window_days": an.get("rolling_window_days"),
        }

    def add_ric_entry(
        self,
        sleeve_id: str,
        ric: str,
        field: str,
        status: str,
        fallback_level: int = 0,
    ) -> None:
        self.ric_mapping[sleeve_id] = {
            "ric": ric,
            "field": field,
            "status": status,
            "fallback_level": fallback_level,
        }

    def set_integrity_status(self, portfolio_id: str, status: str) -> None:
        self.integrity_status[portfolio_id] = status

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        elapsed = (datetime.datetime.now() - self._start).total_seconds()
        self.runtime_seconds = round(elapsed, 1)
        return {
            "run_id": self.run_id,
            "run_datetime": self.run_datetime,
            "python_version": self.python_version,
            "mode": self.mode,
            "data_range": {
                "requested_start": self.data_start,
                "requested_end": self.data_end,
                "common_history_start": self.common_history_start,
            },
            "universe": {
                "total_defined": self.universe_total,
                "available_after_cleaning": len(self.available_sleeves),
                "dropped": len(self.dropped_sleeves),
                "available_ids": self.available_sleeves,
                "dropped_ids": self.dropped_sleeves,
            },
            "portfolios_run": self.portfolios,
            "settings": self.settings,
            "ric_mapping": self.ric_mapping,
            "integrity_status": self.integrity_status,
            "runtime_seconds": self.runtime_seconds,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        out = path or _MANIFEST_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Run manifest saved: %s", out)
        return out


class WarningsCollector:
    """Collect warnings emitted during a run and persist as JSON."""

    def __init__(self) -> None:
        self._warnings: List[Dict] = []

    def add(
        self,
        category: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> None:
        self._warnings.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "category": category,
                "message": message,
                "details": details or {},
            }
        )
        logger.warning("[%s] %s", category, message)

    def count(self) -> int:
        return len(self._warnings)

    def get_by_category(self, category: str) -> List[Dict]:
        return [w for w in self._warnings if w["category"] == category]

    def save(self, path: Optional[Path] = None) -> Path:
        out = path or _WARNINGS_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_warnings": len(self._warnings),
            "warnings_by_category": _group_by(self._warnings, "category"),
            "warnings": self._warnings,
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Warnings file saved: %s  (%d warnings)", out, len(self._warnings))
        return out


def _group_by(items: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[item.get(key, "unknown")] = counts.get(item.get(key, "unknown"), 0) + 1
    return counts
