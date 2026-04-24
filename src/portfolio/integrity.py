"""Portfolio integrity controls.

Makes the hard distinction between:
  - target allocation      (YAML-defined weights)
  - executed allocation    (weights after unavailable sleeves are dropped + renormalised)
  - distortion             (quantified drift between the two)
  - run status             (VALID | DEGRADED | INVALID_PORTFOLIO_CONFIGURATION)

Outputs:
  outputs/tables/target_vs_executed_weights.csv
  outputs/tables/asset_class_drift_report.csv
  outputs/tables/portfolio_validity_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.mapping import get_asset_class_map, get_name_map
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Run-status constants ──────────────────────────────────────────────────────
STATUS_VALID = "VALID"
STATUS_DEGRADED = "DEGRADED"
STATUS_INVALID = "INVALID_PORTFOLIO_CONFIGURATION"

# ── Default thresholds (can be overridden from settings.yaml) ─────────────────
DEFAULT_THRESHOLDS = {
    "max_total_weight_dropped": 0.15,        # >15% total weight dropped → DEGRADED
    "max_total_weight_dropped_invalid": 0.40, # >40% → INVALID
    "max_ac_weight_dropped": 0.10,           # >10% drop in any single asset class → DEGRADED
    "min_sleeves_remaining": 5,              # fewer than 5 sleeves → INVALID
    "min_equity_count": 2,                   # < 2 equity sleeves → DEGRADED
    "min_bond_count": 1,                     # < 1 bond/credit sleeve → DEGRADED
}


@dataclass
class SleeveIntegrityRecord:
    sleeve_id: str
    name: str
    asset_class: str
    target_weight: float
    executed_weight: float
    weight_delta: float           # executed - target
    status: str                   # "OK" | "DROPPED" | "RENORMED"


@dataclass
class AssetClassDrift:
    asset_class: str
    target_weight: float
    executed_weight: float
    weight_delta: float
    target_sleeve_count: int
    executed_sleeve_count: int
    dropped_sleeve_count: int
    status: str                   # "OK" | "REDUCED" | "MISSING"


@dataclass
class IntegrityReport:
    portfolio_id: str
    run_status: str               # VALID | DEGRADED | INVALID_...
    total_dropped_weight: float
    total_renorm_factor: float    # executed weights / target weights (for retained sleeves)
    n_target_sleeves: int
    n_executed_sleeves: int
    n_dropped_sleeves: int
    dropped_sleeve_ids: List[str]
    sleeve_records: List[SleeveIntegrityRecord] = field(default_factory=list)
    asset_class_drifts: List[AssetClassDrift] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_sleeve_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.sleeve_records])

    def to_ac_drift_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(d) for d in self.asset_class_drifts])


def check_portfolio_integrity(
    portfolio_id: str,
    target_weights: Dict[str, float],
    executed_weights: Dict[str, float],
    thresholds: Optional[Dict] = None,
) -> IntegrityReport:
    """Compare target vs executed weights and assess portfolio validity.

    Args:
        portfolio_id: Portfolio identifier (for reporting).
        target_weights: Intended weights from YAML (sleeve_id -> weight).
        executed_weights: Actual weights used after drops + renormalisation.
        thresholds: Override default integrity thresholds.

    Returns:
        IntegrityReport with per-sleeve records, asset-class drift, and run status.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    name_map = get_name_map()
    ac_map = get_asset_class_map()

    dropped_ids = [s for s in target_weights if s not in executed_weights]
    total_dropped_weight = sum(target_weights[s] for s in dropped_ids)
    total_retained_target = 1.0 - total_dropped_weight
    renorm_factor = (1.0 / total_retained_target) if total_retained_target > 0 else 0.0

    # ── Per-sleeve records ────────────────────────────────────────────────
    sleeve_records: List[SleeveIntegrityRecord] = []
    for sid, tw in target_weights.items():
        ew = executed_weights.get(sid, 0.0)
        if sid in dropped_ids:
            status = "DROPPED"
        elif abs(ew - tw) > 0.001:
            status = "RENORMED"
        else:
            status = "OK"
        sleeve_records.append(
            SleeveIntegrityRecord(
                sleeve_id=sid,
                name=name_map.get(sid, sid),
                asset_class=ac_map.get(sid, "Unknown"),
                target_weight=round(tw, 4),
                executed_weight=round(ew, 4),
                weight_delta=round(ew - tw, 4),
                status=status,
            )
        )
    # Add sleeves that are in executed but not in target (e.g. equal_weight)
    for sid, ew in executed_weights.items():
        if sid not in target_weights:
            sleeve_records.append(
                SleeveIntegrityRecord(
                    sleeve_id=sid,
                    name=name_map.get(sid, sid),
                    asset_class=ac_map.get(sid, "Unknown"),
                    target_weight=0.0,
                    executed_weight=round(ew, 4),
                    weight_delta=round(ew, 4),
                    status="ADDED",
                )
            )

    # ── Asset-class drift ─────────────────────────────────────────────────
    all_acs = sorted({ac_map.get(s, "Unknown") for s in {**target_weights, **executed_weights}})
    ac_drifts: List[AssetClassDrift] = []

    for ac in all_acs:
        target_ac_ids = [s for s in target_weights if ac_map.get(s) == ac]
        exec_ac_ids = [s for s in executed_weights if ac_map.get(s) == ac]
        dropped_ac_ids = [s for s in target_ac_ids if s not in executed_weights]

        t_weight = sum(target_weights.get(s, 0.0) for s in target_ac_ids)
        e_weight = sum(executed_weights.get(s, 0.0) for s in exec_ac_ids)

        if len(target_ac_ids) == 0:
            ac_status = "OK"
        elif len(exec_ac_ids) == 0:
            ac_status = "MISSING"
        elif abs(e_weight - t_weight) > t["max_ac_weight_dropped"]:
            ac_status = "REDUCED"
        else:
            ac_status = "OK"

        ac_drifts.append(
            AssetClassDrift(
                asset_class=ac,
                target_weight=round(t_weight, 4),
                executed_weight=round(e_weight, 4),
                weight_delta=round(e_weight - t_weight, 4),
                target_sleeve_count=len(target_ac_ids),
                executed_sleeve_count=len(exec_ac_ids),
                dropped_sleeve_count=len(dropped_ac_ids),
                status=ac_status,
            )
        )

    # ── Determine run status + flags ──────────────────────────────────────
    flags: List[str] = []
    run_status = STATUS_VALID

    # Count remaining sleeves by relevant classes
    equity_count = sum(
        1 for s in executed_weights if ac_map.get(s) == "Equity"
    )
    bond_count = sum(
        1 for s in executed_weights
        if ac_map.get(s) in ("Sovereign Bond", "Credit")
    )
    n_exec = len(executed_weights)

    if total_dropped_weight > t["max_total_weight_dropped_invalid"]:
        run_status = STATUS_INVALID
        flags.append(
            f"Total dropped weight {total_dropped_weight:.1%} exceeds "
            f"invalid threshold {t['max_total_weight_dropped_invalid']:.1%}."
        )
    elif n_exec < t["min_sleeves_remaining"]:
        run_status = STATUS_INVALID
        flags.append(
            f"Only {n_exec} sleeves remain (minimum required: "
            f"{t['min_sleeves_remaining']})."
        )
    elif total_dropped_weight > t["max_total_weight_dropped"]:
        run_status = STATUS_DEGRADED
        flags.append(
            f"Total dropped weight {total_dropped_weight:.1%} exceeds "
            f"degraded threshold {t['max_total_weight_dropped']:.1%}."
        )

    for acd in ac_drifts:
        if acd.status == "REDUCED" and run_status == STATUS_VALID:
            run_status = STATUS_DEGRADED
            flags.append(
                f"Asset class '{acd.asset_class}' weight dropped "
                f"from {acd.target_weight:.1%} to {acd.executed_weight:.1%}."
            )
        if acd.status == "MISSING":
            run_status = STATUS_DEGRADED if run_status == STATUS_VALID else run_status
            flags.append(f"Asset class '{acd.asset_class}' entirely missing from executed portfolio.")

    if equity_count < t["min_equity_count"] and run_status == STATUS_VALID:
        run_status = STATUS_DEGRADED
        flags.append(f"Only {equity_count} equity sleeves remain (minimum: {t['min_equity_count']}).")

    if bond_count < t["min_bond_count"] and run_status == STATUS_VALID:
        run_status = STATUS_DEGRADED
        flags.append(f"Only {bond_count} bond/credit sleeves remain (minimum: {t['min_bond_count']}).")

    if dropped_ids:
        flags.append(f"Dropped sleeves: {dropped_ids}.")

    # ── Log result ────────────────────────────────────────────────────────
    log_fn = logger.warning if run_status != STATUS_VALID else logger.info
    log_fn(
        "Portfolio '%s' integrity: %s  (dropped=%d sleeves / %.1f%% weight)",
        portfolio_id, run_status,
        len(dropped_ids), total_dropped_weight * 100,
    )
    for flag in flags:
        logger.warning("  [INTEGRITY] %s", flag)

    return IntegrityReport(
        portfolio_id=portfolio_id,
        run_status=run_status,
        total_dropped_weight=round(total_dropped_weight, 4),
        total_renorm_factor=round(renorm_factor, 4),
        n_target_sleeves=len(target_weights),
        n_executed_sleeves=n_exec,
        n_dropped_sleeves=len(dropped_ids),
        dropped_sleeve_ids=dropped_ids,
        sleeve_records=sleeve_records,
        asset_class_drifts=ac_drifts,
        flags=flags,
    )


def check_all_portfolios(
    portfolio_ids: List[str],
    all_port_configs: Dict,
    weights_dict: Dict[str, Dict[str, float]],
    thresholds: Optional[Dict] = None,
) -> Dict[str, IntegrityReport]:
    """Run integrity checks across all portfolios in a run.

    Args:
        portfolio_ids: List of portfolio IDs in this run.
        all_port_configs: Full dict from load_all_portfolios().get("portfolios").
        weights_dict: {portfolio_id: executed_weights} built by run_backtest.py.
        thresholds: Optional override thresholds.

    Returns:
        {portfolio_id: IntegrityReport}
    """
    reports: Dict[str, IntegrityReport] = {}
    for pid in portfolio_ids:
        raw_w = all_port_configs.get(pid, {}).get("weights") or {}
        used_w = weights_dict.get(pid, {})
        if not raw_w:
            # Equal-weight or dynamic portfolio: target == executed by design
            reports[pid] = IntegrityReport(
                portfolio_id=pid,
                run_status=STATUS_VALID,
                total_dropped_weight=0.0,
                total_renorm_factor=1.0,
                n_target_sleeves=len(used_w),
                n_executed_sleeves=len(used_w),
                n_dropped_sleeves=0,
                dropped_sleeve_ids=[],
                flags=["Dynamic equal-weight portfolio — no fixed target allocation."],
            )
        else:
            reports[pid] = check_portfolio_integrity(pid, raw_w, used_w, thresholds)
    return reports


def build_validity_summary(reports: Dict[str, IntegrityReport]) -> Dict:
    """Build a JSON-serialisable validity summary across all portfolios."""
    overall_statuses = [r.run_status for r in reports.values()]
    if STATUS_INVALID in overall_statuses:
        overall = STATUS_INVALID
    elif STATUS_DEGRADED in overall_statuses:
        overall = STATUS_DEGRADED
    else:
        overall = STATUS_VALID

    return {
        "overall_run_status": overall,
        "portfolios": {
            pid: {
                "status": r.run_status,
                "dropped_sleeves": r.dropped_sleeve_ids,
                "total_dropped_weight_pct": round(r.total_dropped_weight * 100, 1),
                "n_executed_sleeves": r.n_executed_sleeves,
                "flags": r.flags,
            }
            for pid, r in reports.items()
        },
    }


def export_integrity_reports(
    reports: Dict[str, IntegrityReport],
    output_dir: Path = Path("outputs/tables"),
) -> Dict[str, Path]:
    """Save all integrity report artefacts to disk.

    Returns:
        Dict of {artefact_name: path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # target_vs_executed_weights.csv  (all portfolios, long form)
    sleeve_rows: List[pd.DataFrame] = []
    for pid, rep in reports.items():
        df = rep.to_sleeve_df()
        df.insert(0, "portfolio_id", pid)
        sleeve_rows.append(df)
    if sleeve_rows:
        tve_path = output_dir / "target_vs_executed_weights.csv"
        pd.concat(sleeve_rows, ignore_index=True).to_csv(tve_path, index=False)
        logger.info("Table saved: %s", tve_path)
        paths["target_vs_executed_weights"] = tve_path

    # asset_class_drift_report.csv
    ac_rows: List[pd.DataFrame] = []
    for pid, rep in reports.items():
        df = rep.to_ac_drift_df()
        df.insert(0, "portfolio_id", pid)
        ac_rows.append(df)
    if ac_rows:
        ac_path = output_dir / "asset_class_drift_report.csv"
        pd.concat(ac_rows, ignore_index=True).to_csv(ac_path, index=False)
        logger.info("Table saved: %s", ac_path)
        paths["asset_class_drift_report"] = ac_path

    # portfolio_validity_summary.json
    summary = build_validity_summary(reports)
    json_path = output_dir / "portfolio_validity_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Portfolio validity summary saved: %s", json_path)
    paths["portfolio_validity_summary"] = json_path

    return paths
