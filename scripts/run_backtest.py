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
    tables/portfolio_summary.csv
    tables/monthly_performance.csv
    tables/drawdown_table.csv
    tables/var_es_summary.csv
    tables/stress_test_results.csv
    tables/asset_class_contributions.csv
    tables/rolling_metrics.csv
    charts/cumulative_performance.png
    charts/drawdown.png
    charts/rolling_volatility.png
    charts/rolling_sharpe.png
    charts/monthly_returns_heatmap.png
    charts/asset_allocation.png
    charts/contribution_bar.png
    charts/stress_comparison.png
    charts/var_distribution.png
    docs/images/  (subset for README)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import yaml
import pandas as pd

from src.utils.logging_utils import configure_root_logger, get_logger

logger = get_logger(__name__)

_SETTINGS_PATH = Path("config/settings.yaml")
_DOCS_FIGURES = [
    "cumulative_performance",
    "drawdown",
    "rolling_volatility",
    "monthly_returns_heatmap",
    "stress_comparison",
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
    from src.data.cleaner import clean_pipeline, resample_to_monthly

    logger.info("Loading prices ...")
    raw_prices = load_prices(
        session=session,
        start=start_date,
        end=end_date,
        demo_mode=demo_mode,
    )
    prices, daily_returns = clean_pipeline(raw_prices)
    available_sleeves = list(daily_returns.columns)
    logger.info("Available sleeves after cleaning: %d", len(available_sleeves))

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

    for pid in portfolio_ids:
        try:
            weights = get_portfolio_weights(pid, available_sleeves=available_sleeves)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping portfolio '%s': %s", pid, exc)
            continue

        logger.info("Building portfolio: %s", pid)
        nav, wh = build_portfolio(
            daily_returns, weights, rebal_freq,
            initial_value=100.0, cost_bps=cost_bps,
        )
        nav.name = pid
        nav_dict[pid] = nav
        weights_dict[pid] = weights
        port_returns_dict[pid] = compute_portfolio_returns(nav)

    if not nav_dict:
        logger.error("No portfolios built. Exiting.")
        sys.exit(1)

    nav_df = pd.DataFrame(nav_dict)
    primary_pid = portfolio_ids[0]
    primary_returns = port_returns_dict[primary_pid]
    primary_weights = weights_dict[primary_pid]

    # ── Portfolio display names + LIVE disclosure ─────────────────────────
    # Loaded here so both chart labels and subtitle logic share one load.
    all_port_configs = load_all_portfolios().get("portfolios", {})
    portfolio_labels = {
        pid: all_port_configs.get(pid, {}).get("name", pid)
        for pid in nav_dict
    }

    # In live mode, dropped sleeves cause weights to be renormalised, which
    # can shift the executed allocation far from the strategy label
    # (e.g. balanced_60_40 runs at ~79% equity when both treasury sleeves are
    # missing).  Build per-portfolio disclosure strings so charts are honest.
    port_subtitles: dict = {}
    if not demo_mode:
        from src.data.mapping import get_asset_class_map as _get_ac_map
        _ac_map_live = _get_ac_map()
        for _pid in nav_dict:
            _raw_w = all_port_configs.get(_pid, {}).get("weights") or {}
            _used_w = weights_dict[_pid]
            _n_dropped = sum(1 for s in _raw_w if s not in _used_w)
            if _n_dropped:
                _eq = sum(w for s, w in _used_w.items()
                          if _ac_map_live.get(s) == "Equity") * 100
                _fi = sum(w for s, w in _used_w.items()
                          if _ac_map_live.get(s) in ("Sovereign Bond", "Credit")) * 100
                port_subtitles[_pid] = (
                    f"Live: {_n_dropped} sleeve(s) dropped — "
                    f"executed equity {_eq:.0f}% / fixed income {_fi:.0f}%"
                )

    # Multi-portfolio charts get one combined subtitle line.
    if port_subtitles:
        _parts = []
        for _pid, _note in port_subtitles.items():
            _short = " ".join(portfolio_labels.get(_pid, _pid).split()[:2])
            _alloc = _note.split("— ")[1]
            _parts.append(f"{_short}: {_alloc}")
        multi_subtitle: str | None = "Live run — weights renormalised: " + "  ·  ".join(_parts)
    else:
        multi_subtitle = None

    # ── Analytics ─────────────────────────────────────────────────────────
    from src.analytics.risk import compute_risk_report
    from src.analytics.contributions import build_contribution_table
    from src.analytics.returns import monthly_returns_table

    logger.info("Computing risk analytics ...")
    risk_report = compute_risk_report(
        primary_returns,
        risk_free_annual=rf_rate,
        var_confidence_levels=var_levels,
        rolling_window=roll_window,
    )

    cov_matrix = daily_returns[[s for s in primary_weights if s in daily_returns.columns]].cov() * 252
    contrib_df = build_contribution_table(primary_weights, daily_returns, cov_matrix)

    # ── Stress testing ────────────────────────────────────────────────────
    from src.stress.scenarios import run_full_stress_suite

    logger.info("Running stress scenarios ...")
    stress_results = run_full_stress_suite(
        primary_returns, primary_weights, portfolio_name=primary_pid
    )

    # ── Build output tables ───────────────────────────────────────────────
    from src.reporting.tables import (
        build_portfolio_summary, build_monthly_returns_table,
        build_drawdown_table, build_var_es_table,
        build_stress_test_table, build_contribution_table as fmt_contrib,
    )
    from src.reporting.export import export_all_tables, export_all_figures

    monthly_ret_series = (1 + primary_returns).resample("ME").prod() - 1
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
    }

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
    )
    from src.analytics.var_es import historical_var, historical_es
    from src.data.mapping import get_name_map

    logger.info("Generating charts ...")

    # Sleeve display names for contribution bar
    sleeve_name_map = get_name_map()

    drawdown_series_dict = {}
    rolling_vol_dict = {}
    rolling_sharpe_dict = {}
    from src.analytics.drawdown import drawdown_series as _dd_series
    from src.analytics.rolling import rolling_volatility, rolling_sharpe

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
        "monthly_returns_heatmap": plot_monthly_returns_heatmap(monthly_table_for_chart),
        "asset_allocation": plot_asset_allocation(
            primary_weights, subtitle=port_subtitles.get(primary_pid),
        ),
        "contribution_bar": plot_contribution_bar(
            contrib_series, name_map=sleeve_name_map,
        ),
        "var_distribution": plot_var_distribution(
            primary_returns, var_95, es_95, var_99,
        ),
    }

    # Stress comparison — no_data rows are filtered inside the chart function
    if not stress_results["historical"].empty:
        figures["stress_comparison"] = plot_stress_comparison(
            stress_results["historical"],
            value_col="portfolio_total_return",
            scenario_col="scenario_name",
            group_col=None,
        )

    export_all_figures(figures, docs_figures=_DOCS_FIGURES)

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Backtest complete.")
    logger.info("  Tables → outputs/tables/")
    logger.info("  Charts → outputs/charts/")
    logger.info("  README images → docs/images/")
    logger.info("=" * 60)

    if session:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
