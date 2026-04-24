"""One-page markdown tearsheet generator.

Produces outputs/tearsheet.md — a compact, PM/risk-review-ready summary of a
backtest run. Includes portfolio identity, run status, executed allocation,
return/risk summary, drawdown, tail risk, scenario losses, benchmark context,
data warnings, and integrity status.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TEARSHEET_PATH = Path("outputs/tearsheet.md")


def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # NaN check
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _fmt(v: Optional[float], decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def build_tearsheet(
    portfolio_id: str,
    portfolio_name: str,
    run_date: str,
    run_status: str,
    mode: str,
    common_history_start: str,
    executed_weights: Dict[str, float],
    risk_report: Dict,
    stress_historical: pd.DataFrame,
    integrity_flags: List[str],
    warnings_count: int,
    benchmark_row: Optional[pd.Series] = None,
    dropped_sleeves: Optional[List[str]] = None,
    ac_map: Optional[Dict[str, str]] = None,
    name_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save the markdown tearsheet.

    Args:
        portfolio_id: Portfolio key (e.g. 'strategic_diversified').
        portfolio_name: Human-readable name.
        run_date: ISO date string of the run.
        run_status: 'VALID', 'DEGRADED', or 'INVALID_PORTFOLIO_CONFIGURATION'.
        mode: 'DEMO' or 'LIVE'.
        common_history_start: Start date of the backtested history.
        executed_weights: {sleeve_id: weight} actually used in this run.
        risk_report: Output dict from compute_risk_report().
        stress_historical: Historical stress results DataFrame.
        integrity_flags: List of integrity warning strings.
        warnings_count: Total warning count from WarningsCollector.
        benchmark_row: Optional Series from benchmark_relative_report for this portfolio.
        dropped_sleeves: Sleeves excluded due to data unavailability.
        ac_map: {sleeve_id: asset_class}.
        name_map: {sleeve_id: display_name}.
        output_path: Override output path.

    Returns:
        Path to the saved tearsheet.
    """
    out = output_path or _TEARSHEET_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = risk_report.get("summary", {})
    var_es = risk_report.get("var_es", {})

    status_badge = {
        "VALID": "✅ VALID",
        "DEGRADED": "⚠️ DEGRADED",
        "INVALID_PORTFOLIO_CONFIGURATION": "❌ INVALID",
    }.get(run_status, run_status)

    mode_badge = "📊 DEMO (synthetic data)" if mode == "DEMO" else "📡 LIVE (LSEG data)"

    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        f"# Portfolio Risk Tearsheet — {portfolio_name}",
        "",
        f"**Run date:** {run_date}  |  **Mode:** {mode_badge}  |  **Status:** {status_badge}",
        f"**History:** {common_history_start} to {run_date}",
        "",
        "---",
        "",
    ]

    # ── Integrity / warnings banner ───────────────────────────────────────
    if run_status != "VALID" or warnings_count > 0:
        lines += ["## ⚠️ Run Integrity", ""]
        lines.append(f"Run status: **{run_status}**")
        if dropped_sleeves:
            lines.append(f"Dropped sleeves: `{'`, `'.join(dropped_sleeves)}`")
        for flag in integrity_flags:
            lines.append(f"- {flag}")
        if warnings_count > 0:
            lines.append(f"- {warnings_count} data/processing warning(s) recorded in `outputs/run_warnings.json`.")
        lines += ["", "---", ""]

    # ── Executed allocation ───────────────────────────────────────────────
    lines += ["## Executed Allocation", ""]
    if ac_map and name_map:
        by_ac: Dict[str, List] = {}
        for sid, w in sorted(executed_weights.items(), key=lambda x: -x[1]):
            ac = ac_map.get(sid, "Other")
            by_ac.setdefault(ac, []).append((name_map.get(sid, sid), w))

        lines.append("| Asset Class | Sleeve | Weight |")
        lines.append("|-------------|--------|--------|")
        for ac in sorted(by_ac):
            for name, w in by_ac[ac]:
                lines.append(f"| {ac} | {name} | {_pct(w)} |")
    else:
        lines.append("| Sleeve | Weight |")
        lines.append("|--------|--------|")
        for sid, w in sorted(executed_weights.items(), key=lambda x: -x[1]):
            lines.append(f"| {sid} | {_pct(w)} |")

    lines += ["", "---", ""]

    # ── Return and risk summary ───────────────────────────────────────────
    ann_ret = summary.get("annualised_return")
    ann_vol = summary.get("annualised_volatility")
    sharpe = summary.get("sharpe_ratio")
    sortino = summary.get("sortino_ratio")
    calmar = summary.get("calmar_ratio")
    max_dd = summary.get("max_drawdown")
    total_ret = summary.get("total_return")

    lines += [
        "## Return & Risk Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total return | {_pct(total_ret)} |",
        f"| Annualised return | {_pct(ann_ret)} |",
        f"| Annualised volatility | {_pct(ann_vol)} |",
        f"| Sharpe ratio | {_fmt(sharpe)} |",
        f"| Sortino ratio | {_fmt(sortino)} |",
        f"| Calmar ratio | {_fmt(calmar)} |",
        f"| Max drawdown | {_pct(max_dd)} |",
        "",
        "---",
        "",
    ]

    # ── Tail risk ─────────────────────────────────────────────────────────
    lines += ["## Tail Risk (Historical)", ""]
    lines.append("| Metric | 95% | 99% |")
    lines.append("|--------|-----|-----|")

    var_95 = var_es.get("historical_var_95") or var_es.get(0.95, {}).get("historical_var")
    var_99 = var_es.get("historical_var_99") or var_es.get(0.99, {}).get("historical_var")
    es_95 = var_es.get("historical_es_95") or var_es.get(0.95, {}).get("historical_es")
    es_99 = var_es.get("historical_es_99") or var_es.get(0.99, {}).get("historical_es")

    lines.append(f"| Historical VaR (1-day) | {_pct(var_95)} | {_pct(var_99)} |")
    lines.append(f"| Historical ES (CVaR) | {_pct(es_95)} | {_pct(es_99)} |")
    lines += ["", "> *VaR reported as a positive loss fraction. Historical method, no distributional assumption.*", "", "---", ""]

    # ── Scenario losses ───────────────────────────────────────────────────
    if not stress_historical.empty:
        ok_rows = (
            stress_historical[stress_historical["status"] == "ok"]
            if "status" in stress_historical.columns
            else stress_historical.dropna(subset=["portfolio_total_return"])
        )
        if not ok_rows.empty:
            lines += ["## Historical Stress Scenarios", ""]
            lines.append("| Scenario | Total Return | Max Drawdown |")
            lines.append("|----------|-------------|-------------|")
            for _, row in ok_rows.iterrows():
                tr = row.get("portfolio_total_return")
                dd = row.get("max_drawdown_in_window")
                sname = row.get("scenario_name", "")
                lines.append(f"| {sname} | {_pct(tr)} | {_pct(dd)} |")
            lines += ["", "---", ""]

    # ── Benchmark context ─────────────────────────────────────────────────
    if benchmark_row is not None and benchmark_row.get("note") == "OK":
        bm_id = benchmark_row.get("benchmark_id", "")
        te = benchmark_row.get("tracking_error_ann")
        ir = benchmark_row.get("information_ratio")
        ar = benchmark_row.get("active_return_ann")
        lines += [
            "## Benchmark-Relative",
            "",
            f"Benchmark: `{bm_id}`",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Active return (ann.) | {_pct(ar)} |",
            f"| Tracking error (ann.) | {_pct(te)} |",
            f"| Information ratio | {_fmt(ir)} |",
            "",
            "---",
            "",
        ]

    # ── Data and methodology notes ────────────────────────────────────────
    lines += [
        "## Data & Methodology Notes",
        "",
        "- Returns are **price returns** (not total returns). Dividends and income are excluded.",
        "- All sleeves are USD-denominated ETF proxies. Non-USD sleeves embed unhedged FX risk.",
        "- Transaction costs: 2 bps one-way per rebalancing trade (linear model, no market impact).",
        "- VaR and ES are **historical** (empirical), not parametric. Parametric VaR assumes normality and is reported for comparison only.",
        "- Stress test windows are applied to the **executed** portfolio allocation, not the target.",
        "- Custom shock scenarios use a linear first-order approximation (weight × shock). Convexity and cross-asset dynamics are not modelled.",
    ]
    if mode == "DEMO":
        lines += [
            "- **DEMO MODE**: All prices are synthetic. Results are for demonstration purposes only.",
            "  Synthetic data is calibrated to realistic volatilities and correlations but does not replicate actual historical returns.",
        ]

    lines += ["", "---", ""]
    lines.append(f"*Generated by Portfolio Risk Backtesting Toolkit on {run_date}*")

    text = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)

    logger.info("Tearsheet saved: %s", out)
    return out
