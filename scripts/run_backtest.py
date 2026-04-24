#!/usr/bin/env python
"""Main backtest runner.

Orchestrates the full pipeline:
  data load → clean → portfolio construction → analytics → stress → export

Usage:
    # Live LSEG data, strategic portfolio:
    python scripts/run_backtest.py --portfolio strategic_diversified

    # Demo mode (no LSEG required):
    python scripts/run_backtest.py --demo --portfolio all

    # Custom date range:
    python scripts/run_backtest.py --demo --start 2010-01-01 --end 2023-12-31

Outputs (in outputs/):
    run_manifest.json
    run_warnings.json
    tearsheet.md
    tables/portfolio_summary.csv
    tables/monthly_performance.csv
    tables/drawdown_table.csv
    tables/var_es_summary.csv
    tables/stress_test_results.csv
    tables/asset_class_contributions.csv
    tables/rolling_metrics.csv
    tables/target_vs_executed_weights.csv
    tables/asset_class_drift_report.csv
    tables/portfolio_validity_summary.json
    tables/data_quality_report.csv
    tables/correlation_report.csv
    tables/benchmark_relative_report.csv
    tables/turnover_report.csv
    charts/cumulative_performance.png
    charts/drawdown.png
    charts/rolling_volatility.png
    charts/rolling_sharpe.png
    charts/monthly_returns_heatmap.png
    charts/asset_allocation.png
    charts/contribution_bar.png
    charts/stress_comparison.png
    charts/var_distribution.png
    charts/correlation_heatmap.png
    docs/images/  (subset for README)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import yaml
import pandas as pd

from src.utils.logging_utils import configure_root_logger, get_logger
from src.reporting.manifest import RunManifest, WarningsCollector

logger = get_logger(__name__)

_SETTINGS_PATH = Path("config/settings.yaml")
_DOCS_FIGURES = [
    "cumulative_performance",
    "drawdown",
    "rolling_volatility",
    "monthly_returns_heatmap",
    "stress_comparison",
    "correlation_heatmap",
]


def load_settings() -> dict:
    with open(_SETTINGS_PATH) as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run portfolio risk backtest.")
    parser.add_argument("--portfolio", default="strategic_diversified",
                        help="Portfolio ID from portfolio_weights.yaml, or 'all'")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data (no LSEG required)")
    parser.add_argument("--rebalance", default=None,
                        choices=["monthly", "quarterly", "annual", "buy_and_hold"],
                        help="Override rebalancing frequency from settings")
    parser.add_argument("--session-type", default="platform",
                        choices=["platform", "deployed"])
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings()
    configure_root_logger(level=args.log_level)

    manifest = RunManifest()
    warnings = WarningsCollector()

    logger.info("=" * 60)
    logger.info("Portfolio Risk Backtesting Toolkit")
    logger.info("=" * 60)

    # ── Resolve settings ──────────────────────────────────────────────────
    demo_mode = args.demo or settings["demo_mode"]["enabled"]
    start_date = args.start or settings["backtesting"]["start_date"]
    end_date = args.end or settings["backtesting"]["end_date"]
    rebal_freq = args.rebalance or settings["backtesting"]["rebalancing_frequency"]
    cost_bps = settings["backtesting"]["transaction_costs_bps"]
    rf_rate = settings["analytics"]["risk_free_rate_annual"]
    var_levels = settings["analytics"]["var_confidence_levels"]
    roll_window = settings["analytics"]["rolling_window_days"]

    manifest.set_mode(demo_mode)
    manifest.set_settings(settings)

    logger.info("Mode: %s | Start: %s | End: %s | Rebalance: %s",
                "DEMO" if demo_mode else "LIVE", start_date, end_date or "today", rebal_freq)

    # ── Load data ─────────────────────────────────────────────────────────
    session = None
    if not demo_mode:
        from src.data.lseg_session import get_session, LSEGUnavailableError
        try:
            session = get_session(session_type=args.session_type)
        except LSEGUnavailableError as exc:
            logger.error("LSEG unavailable: %s\nRe-run with --demo flag.", exc)
            sys.exit(1)

    from src.data.loader import load_prices
    from src.data.cleaner import clean_pipeline_with_report, resample_to_monthly
    from src.data.mapping import get_sleeve_list

    logger.info("Loading prices ...")
    raw_prices = load_prices(
        session=session,
        start=start_date,
        end=end_date,
        demo_mode=demo_mode,
    )

    universe_total = len(get_sleeve_list())
    prices, daily_returns, cleaning_report = clean_pipeline_with_report(raw_prices)
    available_sleeves = list(daily_returns.columns)
    common_start = cleaning_report.common_start

    manifest.set_universe(available_sleeves, universe_total)
    manifest.set_dropped_sleeves(cleaning_report.dropped_sleeves)
    manifest.set_data_range(start_date, end_date, common_start)

    logger.info("Available sleeves after cleaning: %d", len(available_sleeves))

    # Emit warnings for cleaning outcomes
    if cleaning_report.dropped_sleeves:
        warnings.add(
            "DATA_QUALITY",
            f"{len(cleaning_report.dropped_sleeves)} sleeve(s) dropped during cleaning.",
            {"dropped": cleaning_report.dropped_sleeves,
             "reasons": cleaning_report.drop_reasons},
        )
    total_fills = sum(cleaning_report.bdate_gap_fills.values())
    if total_fills > 0:
        warnings.add(
            "DATA_QUALITY",
            f"{total_fills} gap-fills applied across sleeves (ffill ≤ 5 days).",
            {"fills_by_sleeve": cleaning_report.bdate_gap_fills},
        )
    if cleaning_report.outliers_capped:
        warnings.add(
            "DATA_QUALITY",
            f"Outliers capped in {len(cleaning_report.outliers_capped)} sleeve(s).",
            {"capped_by_sleeve": cleaning_report.outliers_capped},
        )

    # Save data quality report
    cleaning_report.to_csv(Path("outputs/tables/data_quality_report.csv"))

    # ── Build portfolios ──────────────────────────────────────────────────
    from src.portfolio.weights import get_portfolio_weights, list_portfolio_ids, load_all_portfolios
    from src.portfolio.construction import build_portfolio, compute_portfolio_returns

    portfolio_ids = (
        list_portfolio_ids() if args.portfolio == "all"
        else [args.portfolio]
    )

    nav_dict: dict = {}
    weights_dict: dict = {}
    port_returns_dict: dict = {}
    weights_history_dict: dict = {}

    for pid in portfolio_ids:
        try:
            weights = get_portfolio_weights(pid, available_sleeves=available_sleeves)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping portfolio '%s': %s", pid, exc)
            warnings.add("PORTFOLIO", f"Portfolio '{pid}' skipped: {exc}")
            continue

        logger.info("Building portfolio: %s", pid)
        nav, wh = build_portfolio(
            daily_returns, weights, rebal_freq,
            initial_value=100.0, cost_bps=cost_bps,
            track_rebalances=True,
        )
        nav.name = pid
        nav_dict[pid] = nav
        weights_dict[pid] = weights
        port_returns_dict[pid] = compute_portfolio_returns(nav)
        weights_history_dict[pid] = wh

    if not nav_dict:
        logger.error("No portfolios built. Exiting.")
        sys.exit(1)

    nav_df = pd.DataFrame(nav_dict)
    primary_pid = portfolio_ids[0]
    primary_returns = port_returns_dict[primary_pid]
    primary_weights = weights_dict[primary_pid]

    # ── Portfolio display names + LIVE disclosure ─────────────────────────
    all_port_configs = load_all_portfolios().get("portfolios", {})
    portfolio_labels = {
        pid: all_port_configs.get(pid, {}).get("name", pid)
        for pid in nav_dict
    }

    manifest.set_portfolios(list(nav_dict.keys()))

    # Load name map early — used both for subtitles and later for analytics.
    from src.data.mapping import get_name_map as _get_name_map_early
    _sleeve_name_map_early = _get_name_map_early()

    # In live mode, detect renormalised weights and build disclosure subtitles.
    # Subtitles name specific dropped sleeves (not just a count) and report the
    # total dropped target weight so a PM can assess distortion at a glance.
    port_subtitles: dict = {}
    if not demo_mode:
        from src.data.mapping import get_asset_class_map as _get_ac_map
        _ac_map_live = _get_ac_map()
        for _pid in nav_dict:
            _raw_w = all_port_configs.get(_pid, {}).get("weights") or {}
            _used_w = weights_dict[_pid]
            _dropped_ids = [s for s in _raw_w if s not in _used_w]
            if _dropped_ids:
                _dropped_pct = sum(_raw_w.get(s, 0) for s in _dropped_ids) * 100
                _eq = sum(w for s, w in _used_w.items()
                          if _ac_map_live.get(s) == "Equity") * 100
                _fi = sum(w for s, w in _used_w.items()
                          if _ac_map_live.get(s) in ("Sovereign Bond", "Credit")) * 100
                # Show up to 3 human-readable dropped sleeve names
                _dropped_names = [_sleeve_name_map_early.get(s, s) for s in _dropped_ids]
                _names_str = ", ".join(_dropped_names[:3])
                if len(_dropped_names) > 3:
                    _names_str += f" +{len(_dropped_names) - 3} more"
                port_subtitles[_pid] = (
                    f"LIVE · Unavailable: {_names_str} · "
                    f"{_dropped_pct:.0f}% target weight dropped · "
                    f"executed equity {_eq:.0f}%  fixed income {_fi:.0f}%"
                )

    if port_subtitles:
        # For multi-portfolio charts use a concise combined disclosure line.
        if len(port_subtitles) == 1:
            multi_subtitle: str | None = next(iter(port_subtitles.values()))
        else:
            _parts = []
            for _pid, _note in port_subtitles.items():
                _label = " ".join(portfolio_labels.get(_pid, _pid).split()[:2])
                # Extract just the weight-drop figure for the compact line
                try:
                    _drop_part = [p for p in _note.split("·") if "%" in p and "weight" in p][0].strip()
                except IndexError:
                    _drop_part = "weights renormalised"
                _parts.append(f"{_label}: {_drop_part}")
            multi_subtitle = "LIVE · weights renormalised — " + "  ·  ".join(_parts)
    else:
        multi_subtitle = None

    # ── Portfolio integrity checks ─────────────────────────────────────────
    from src.portfolio.integrity import check_all_portfolios, export_integrity_reports

    logger.info("Running portfolio integrity checks ...")
    integrity_reports = check_all_portfolios(
        list(nav_dict.keys()), all_port_configs, weights_dict
    )
    export_integrity_reports(integrity_reports)

    for pid, rep in integrity_reports.items():
        manifest.set_integrity_status(pid, rep.run_status)
        if rep.run_status != "VALID":
            warnings.add(
                "PORTFOLIO_INTEGRITY",
                f"Portfolio '{pid}' integrity status: {rep.run_status}",
                {"flags": rep.flags, "dropped": rep.dropped_sleeve_ids},
            )

    primary_integrity = integrity_reports.get(primary_pid)

    # ── Analytics ─────────────────────────────────────────────────────────
    from src.analytics.risk import compute_risk_report
    from src.analytics.contributions import build_contribution_table
    from src.analytics.returns import monthly_returns_table
    from src.analytics.correlation import build_correlation_report
    from src.analytics.benchmark import build_benchmark_report

    logger.info("Computing risk analytics ...")
    risk_report = compute_risk_report(
        primary_returns,
        risk_free_annual=rf_rate,
        var_confidence_levels=var_levels,
        rolling_window=roll_window,
    )

    cov_matrix = daily_returns[[s for s in primary_weights if s in daily_returns.columns]].cov() * 252
    contrib_df = build_contribution_table(primary_weights, daily_returns, cov_matrix)

    # Correlation report
    sleeve_name_map = _sleeve_name_map_early  # already loaded above
    corr_matrix, corr_summary = build_correlation_report(
        daily_returns[[s for s in primary_weights if s in daily_returns.columns]],
        name_map=sleeve_name_map,
    )

    # Benchmark-relative
    bench_report = build_benchmark_report(port_returns_dict)

    # ── Stress testing ────────────────────────────────────────────────────
    from src.stress.scenarios import run_full_stress_suite

    logger.info("Running stress scenarios ...")
    stress_results = run_full_stress_suite(
        primary_returns, primary_weights, portfolio_name=primary_pid
    )

    # ── Turnover report ───────────────────────────────────────────────────
    from src.portfolio.turnover import build_turnover_report, build_turnover_summary

    turnover_reports = {}
    for pid in nav_dict:
        wh = weights_history_dict[pid]
        turnover_reports[pid] = build_turnover_report(wh, weights_dict[pid], pid)
    turnover_summary = build_turnover_summary(turnover_reports)

    # ── Build output tables ───────────────────────────────────────────────
    from src.reporting.tables import (
        build_portfolio_summary, build_monthly_returns_table,
        build_drawdown_table, build_var_es_table,
        build_stress_test_table, build_contribution_table as fmt_contrib,
    )
    from src.reporting.export import export_all_tables, export_all_figures, save_table

    monthly_table_raw = monthly_returns_table(primary_returns)

    tables = {
        "portfolio_summary": build_portfolio_summary(
            {pid: port_returns_dict[pid] for pid in nav_dict}, rf_rate
        ),
        "monthly_performance": build_monthly_returns_table(primary_returns, primary_pid),
        "drawdown_table": build_drawdown_table(primary_returns),
        "var_es_summary": build_var_es_table(
            {pid: port_returns_dict[pid] for pid in nav_dict}, var_levels
        ),
        "rolling_metrics": risk_report["rolling_metrics"],
        "asset_class_contributions": fmt_contrib(contrib_df),
        "stress_test_historical": stress_results["historical"],
        "stress_test_shocks": stress_results["custom_shocks"],
        "benchmark_relative_report": bench_report,
        "turnover_summary": turnover_summary,
    }

    # Per-portfolio turnover detail
    all_turnover = pd.concat(list(turnover_reports.values()), ignore_index=True)
    if not all_turnover.empty:
        tables["turnover_report"] = all_turnover

    # Correlation report (round for readability)
    tables["correlation_report"] = corr_matrix.round(4)

    stress_display = build_stress_test_table(
        stress_results["historical"], stress_results["custom_shocks"]
    )
    tables["stress_test_results"] = stress_display["historical"]

    logger.info("Exporting tables ...")
    export_all_tables(tables)

    # ── Build charts ──────────────────────────────────────────────────────
    from src.reporting.charts import (
        plot_cumulative_performance, plot_drawdown, plot_rolling_volatility,
        plot_rolling_sharpe, plot_monthly_returns_heatmap, plot_asset_allocation,
        plot_contribution_bar, plot_stress_comparison, plot_var_distribution,
        plot_correlation_heatmap,
    )
    from src.analytics.var_es import historical_var, historical_es
    from src.analytics.drawdown import drawdown_series as _dd_series
    from src.analytics.rolling import rolling_volatility, rolling_sharpe

    logger.info("Generating charts ...")

    drawdown_series_dict = {}
    rolling_vol_dict = {}
    rolling_sharpe_dict = {}

    for pid, pr in port_returns_dict.items():
        drawdown_series_dict[pid] = _dd_series(pr)
        rolling_vol_dict[pid] = rolling_volatility(pr, roll_window)
        rolling_sharpe_dict[pid] = rolling_sharpe(pr, roll_window, rf_rate)

    var_95 = historical_var(primary_returns, 0.95)
    es_95 = historical_es(primary_returns, 0.95)
    var_99 = historical_var(primary_returns, 0.99)

    monthly_table_for_chart = monthly_returns_table(primary_returns)

    contrib_series = (
        contrib_df["return_contribution"]
        if "return_contribution" in contrib_df.columns
        else contrib_df.iloc[:, 0]
    )

    # Build chart titles — in LIVE degraded runs make it explicit this is the
    # executed (renormalised) portfolio, not the target allocation.
    _primary_label = portfolio_labels.get(primary_pid, primary_pid)
    _primary_integrity = integrity_reports.get(primary_pid)
    _is_degraded = (
        not demo_mode
        and _primary_integrity is not None
        and _primary_integrity.run_status != "VALID"
    )
    _alloc_title = (
        f"Executed Portfolio Allocation — {_primary_label}"
        if _is_degraded
        else f"Portfolio Asset Allocation — {_primary_label}"
    )
    _contrib_title = (
        f"Return Contribution by Sleeve — {_primary_label} (Executed, Full Period)"
        if _is_degraded
        else f"Return Contribution by Sleeve — {_primary_label} (Full Period)"
    )
    _var_title = f"Daily Return Distribution — {_primary_label}"
    _stress_title = f"Historical Stress Scenarios — {_primary_label}"

    figures = {
        "cumulative_performance": plot_cumulative_performance(
            nav_df, labels=portfolio_labels, subtitle=multi_subtitle,
        ),
        "drawdown": plot_drawdown(
            drawdown_series_dict, labels=portfolio_labels, subtitle=multi_subtitle,
        ),
        "rolling_volatility": plot_rolling_volatility(
            rolling_vol_dict, labels=portfolio_labels, subtitle=multi_subtitle,
        ),
        "rolling_sharpe": plot_rolling_sharpe(
            rolling_sharpe_dict, labels=portfolio_labels, subtitle=multi_subtitle,
        ),
        "monthly_returns_heatmap": plot_monthly_returns_heatmap(
            monthly_table_for_chart,
            title=f"Monthly Returns (%) — {_primary_label}",
        ),
        "asset_allocation": plot_asset_allocation(
            primary_weights,
            title=_alloc_title,
            subtitle=port_subtitles.get(primary_pid),
        ),
        "contribution_bar": plot_contribution_bar(
            contrib_series,
            name_map=sleeve_name_map,
            title=_contrib_title,
        ),
        "var_distribution": plot_var_distribution(
            primary_returns, var_95, es_95, var_99,
            title=_var_title,
        ),
        "correlation_heatmap": plot_correlation_heatmap(corr_matrix),
    }

    # Stress comparison — no_data rows are filtered inside the chart function
    if not stress_results["historical"].empty:
        figures["stress_comparison"] = plot_stress_comparison(
            stress_results["historical"],
            value_col="portfolio_total_return",
            scenario_col="scenario_name",
            group_col=None,
            title=_stress_title,
        )

    export_all_figures(figures, docs_figures=_DOCS_FIGURES)

    # ── Tearsheet ─────────────────────────────────────────────────────────
    from src.reporting.tearsheet import build_tearsheet
    from src.data.mapping import get_asset_class_map, get_name_map

    import datetime
    run_date = datetime.date.today().isoformat()

    bench_row = None
    if not bench_report.empty:
        primary_bench = bench_report[bench_report["portfolio_id"] == primary_pid]
        if not primary_bench.empty:
            bench_row = primary_bench.iloc[0]

    build_tearsheet(
        portfolio_id=primary_pid,
        portfolio_name=portfolio_labels.get(primary_pid, primary_pid),
        run_date=run_date,
        run_status=primary_integrity.run_status if primary_integrity else "VALID",
        mode="DEMO" if demo_mode else "LIVE",
        common_history_start=common_start or str(start_date),
        executed_weights=primary_weights,
        risk_report=risk_report,
        stress_historical=stress_results["historical"],
        integrity_flags=primary_integrity.flags if primary_integrity else [],
        warnings_count=warnings.count(),
        benchmark_row=bench_row,
        dropped_sleeves=cleaning_report.dropped_sleeves,
        ac_map=get_asset_class_map(),
        name_map=get_name_map(),
    )

    # ── Save manifest and warnings ────────────────────────────────────────
    manifest.save()
    warnings.save()

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Backtest complete.")
    logger.info("  Tables      → outputs/tables/")
    logger.info("  Charts      → outputs/charts/")
    logger.info("  Tearsheet   → outputs/tearsheet.md")
    logger.info("  Manifest    → outputs/run_manifest.json")
    logger.info("  Warnings    → outputs/run_warnings.json")
    logger.info("  README imgs → docs/images/")
    logger.info("=" * 60)

    if session:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
