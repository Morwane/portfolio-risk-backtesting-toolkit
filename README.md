# Diversified Portfolio Backtesting & Risk Analytics Toolkit

A professional Python toolkit for multi-asset portfolio backtesting, risk analytics, and stress testing — built around LSEG market data.

---

## What this project does

This toolkit answers the question a risk team or multi-asset allocator asks every day:

> *How is the portfolio actually performing, what are its tail risks, and how would it survive a repeat of past crises?*

It builds a coherent pipeline from raw price data to a complete risk report:

```
LSEG data  →  cleaning  →  returns  →  portfolio construction
          →  risk analytics (VaR, ES, drawdowns, rolling metrics)
          →  stress testing (historical windows + custom shocks)
          →  contribution analysis  →  exportable tables and charts
```

The project is intentionally distinct from optimisation or factor modelling work. There is no Black-Litterman, no mean-variance optimisation, no ESG overlay. The focus is squarely on **risk monitoring, attribution, and scenario analysis** for a diversified portfolio.

---

## Universe: 22 Multi-Asset Sleeves

| # | Sleeve | Proxy ETF | Asset Class |
|---|--------|-----------|-------------|
| 1 | US Large Cap Equity | SPY | Equity |
| 2 | US Tech / Growth | QQQ | Equity |
| 3 | Europe Equity | EZU | Equity |
| 4 | UK Equity | EWU | Equity |
| 5 | Japan Equity | EWJ | Equity |
| 6 | Emerging Markets | EEM | Equity |
| 7 | US Small Cap | IWM | Equity |
| 8 | Global Min Volatility | USMV | Equity |
| 9 | US Treasuries 1-3Y | SHY | Sovereign Bond |
| 10 | US Treasuries 7-10Y | IEF | Sovereign Bond |
| 11 | Euro Government Bonds | IBGE.AS | Sovereign Bond |
| 12 | US TIPS (Inflation-Linked) | TIP | Sovereign Bond |
| 13 | UK Gilts | IGLT.L | Sovereign Bond |
| 14 | US Investment Grade Credit | LQD | Credit |
| 15 | US High Yield Credit | HYG | Credit |
| 16 | Euro Investment Grade Credit | IEAC.AS | Credit |
| 17 | Euro High Yield Credit | IHYG.AS | Credit |
| 18 | Gold | GLD | Commodity |
| 19 | Brent Crude Oil | BNO | Commodity |
| 20 | Broad Commodities | PDBC | Commodity |
| 21 | REITs / Listed Real Estate | VNQ | Real Estate |
| 22 | Cash (USD T-Bill) | BIL | Cash |

Each sleeve has a primary RIC, a fallback RIC, and a documented LSEG entitlement note. The discovery layer (`scripts/validate_universe.py`) checks which instruments are accessible before any analysis runs.

---

## Data Source

**Primary:** [LSEG Data Library / Eikon API](https://developers.lseg.com/)

The toolkit supports three LSEG Python backends in priority order:
1. `lseg-data` (newest LSEG Data Library)
2. `refinitiv-data` (previous branding, same library)
3. `eikon` (legacy Eikon Python API)

**Entitlement notes:**
- US-listed ETF price series (`TR.PriceClose`) are broadly available
- Exchange-suffixed RICs (`.AS`, `.L`, `.DE`) require EU/UK market data entitlements
- `TR.TotalReturn` (total return series) requires a premium data subscription; the toolkit defaults to price return and flags this clearly
- If any sleeve is inaccessible, the fallback RIC is tried automatically

**Demo mode (no LSEG required):**
```bash
python scripts/build_demo_outputs.py
```
This generates 20 years of synthetic multi-asset prices using a factor model with realistic volatilities, correlations, and crisis regimes (GFC, COVID, 2022 rate shock). All outputs are produced from synthetic data.

---

## Risk Methodology

### Value at Risk
| Method | Description |
|--------|-------------|
| Historical VaR | Empirical quantile of the return distribution. No distributional assumption. Preserves fat tails. |
| Parametric VaR | Gaussian: µ − z_α × σ. Assumes normality — will underestimate tail risk. Reported for comparison. |

### Expected Shortfall (CVaR)
Historical ES: mean of returns below the VaR threshold. More coherent than VaR; the standard metric in FRTB and AIFMD risk reporting.

### Drawdown
Rolling peak-to-trough drawdown series. Top-N drawdown event table with peak date, trough date, recovery date, and duration.

### Rolling Metrics
252-day rolling volatility and Sharpe ratio (63-day shorter window also available). Reveals how risk profile evolves through time — particularly useful for spotting regime changes.

### Contribution Analysis
- **Return contribution**: weight × asset return (Brinson-style)
- **Risk contribution %**: each asset's percentage of total portfolio variance
- **Marginal risk contribution**: sensitivity of portfolio vol to a unit increase in each sleeve's weight

---

## Stress Testing

### Historical Windows
| Scenario | Period | Description |
|----------|--------|-------------|
| COVID-19 Crash | Feb–Mar 2020 | -34% in 33 days |
| COVID Recovery | Mar–Aug 2020 | V-shaped stimulus rally |
| 2022 Rate Shock | Jan–Oct 2022 | Fed hiking cycle, 60/40 breakdown |
| GFC 2008-2009 | Sep 2008–Mar 2009 | Peak stress, -55% equities |
| Oil Collapse 2020 | Feb–Apr 2020 | Demand destruction |
| Taper Tantrum 2013 | May–Sep 2013 | Rate spike on tapering signal |
| EU Debt Crisis | Jul–Oct 2011 | Sovereign spread widening |

### Custom Shocks
Five synthetic factor shock scenarios: equity severe (-25%), rates +100bps, credit spread widening +250bps, stagflation, and commodity spike. Each applies a linear weight-scaled P&L approximation to the current portfolio.

---

## Outputs

### Tables (`outputs/tables/`)
| File | Description |
|------|-------------|
| `portfolio_summary.csv` | Annualised return, vol, Sharpe, Sortino, Calmar, max drawdown |
| `monthly_performance.csv` | Calendar heatmap of monthly returns |
| `drawdown_table.csv` | Top-10 drawdown episodes with duration and recovery |
| `var_es_summary.csv` | Historical and parametric VaR/ES at 95% and 99% |
| `stress_test_results.csv` | Portfolio return and drawdown per historical scenario |
| `asset_class_contributions.csv` | Return and risk contribution per sleeve |
| `rolling_metrics.csv` | Daily rolling volatility and Sharpe ratio |

## Charts Preview

### Cumulative Performance
![Cumulative Performance](docs/images/cumulative_performance.png)

### Drawdown
![Drawdown](docs/images/drawdown.png)

### Monthly Returns Heatmap
![Monthly Returns Heatmap](docs/images/monthly_returns_heatmap.png)

### Rolling Volatility
![Rolling Volatility](docs/images/rolling_volatility.png)

### Rolling Sharpe
![Rolling Sharpe](docs/images/rolling_sharpe.png)

### Asset Allocation
![Asset Allocation](docs/images/asset_allocation.png)

### Contribution Bar
![Contribution Bar](docs/images/contribution_bar.png)

### VaR Distribution
![VaR Distribution](docs/images/var_distribution.png)

### Stress Comparison
![Stress Comparison](docs/images/stress_comparison.png)

---

## Project Structure

```
portfolio-risk-backtesting-toolkit/
├── config/
│   ├── settings.yaml          # global settings (dates, risk-free rate, etc.)
│   ├── universe.yaml          # 22 sleeves with RICs, fallbacks, entitlement notes
│   ├── portfolio_weights.yaml # Strategic, 60/40, Defensive, Equal Weight
│   └── stress_scenarios.yaml  # Historical windows + custom shock vectors
├── data/
│   ├── raw/                   # gitignored — raw LSEG downloads
│   ├── processed/             # gitignored — parquet cache + validation report
│   └── sample/
│       └── generate_demo_data.py
├── docs/images/               # charts committed for README display
├── scripts/
│   ├── validate_universe.py   # probe LSEG before running backtest
│   ├── run_backtest.py        # main pipeline runner
│   └── build_demo_outputs.py  # generate all outputs with synthetic data
├── src/
│   ├── data/                  # lseg_session, discovery, loader, cleaner, mapping
│   ├── portfolio/             # weights, rebalancing, construction
│   ├── analytics/             # returns, performance, var_es, drawdown, rolling, contributions, risk
│   ├── stress/                # historical, shocks, scenarios
│   ├── reporting/             # tables, charts, export
│   └── utils/                 # logging_utils, validation
└── tests/
    ├── test_returns.py
    ├── test_var_es.py
    ├── test_drawdown.py
    ├── test_stress.py
    └── test_portfolio.py
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Then install your LSEG library:
pip install lseg-data       # or refinitiv-data, or eikon
```

### 2. Set credentials (LSEG mode)

```bash
cp .env.example .env
# Edit .env with your LSEG_APP_KEY or client credentials
```

### 3. Validate universe (recommended before first run)

```bash
python scripts/validate_universe.py
# Or in demo mode (no LSEG):
python scripts/validate_universe.py --demo
```

### 4. Run backtest

```bash
# Demo mode — no LSEG needed, uses synthetic data:
python scripts/run_backtest.py --demo --portfolio strategic_diversified

# Live LSEG data:
python scripts/run_backtest.py --portfolio strategic_diversified

# All portfolios simultaneously:
python scripts/run_backtest.py --demo --portfolio all

# Custom date range:
python scripts/run_backtest.py --demo --start 2010-01-01 --end 2023-12-31
```

### 5. Run tests

```bash
pytest
pytest --cov=src tests/
```

---

## Portfolio Configurations

| Portfolio | Description | Equity | Bonds | Credit | Real Assets |
|-----------|-------------|--------|-------|--------|-------------|
| `strategic_diversified` | Broad multi-asset (default) | 55% | 18% | 12% | 13% |
| `balanced_60_40` | Classic 60/40 benchmark | 60% | 40% | — | — |
| `defensive` | Risk-off allocation | 20% | 55% | 7% | 20% (incl. gold) |
| `equal_weight` | 1/N across available sleeves | dynamic | dynamic | dynamic | dynamic |

---

## Limitations

The following are explicit design constraints, not oversights:

1. **Price return, not total return**: The default uses adjusted close prices, not dividend-inclusive total return series. Total return series (`TR.TotalReturn`) require a premium LSEG entitlement. This understates long-run performance of bond and high-yield ETFs.

2. **ETF proxies, not indices**: All sleeves use liquid ETF proxies (SPY, IEF, LQD, etc.) rather than underlying index series. This introduces small tracking error vs. benchmarks but ensures data accessibility.

3. **Linear shock approximation**: Custom shocks apply a weight-scaled first-order P&L estimate. This does not capture convexity, correlation shifts, or cross-sleeve feedback during stress.

4. **No leverage or short positions**: The portfolio construction engine assumes long-only, fully invested portfolios.

5. **Single base currency (USD)**: EUR- and GBP-denominated sleeves embed FX risk. No explicit currency hedging is modelled.

6. **Synthetic demo data**: The demo mode generates statistically plausible but entirely synthetic data. It is calibrated for demonstration purposes only and does not replicate real historical returns.

---

## What makes this project professionally interesting

- **Cross-asset stress decomposition**: The toolkit isolates how much of a portfolio's crash was due to equity beta vs. duration risk vs. credit spread widening — which is the actual question asked in multi-asset risk reviews.
- **2022 regime exposure**: The 2022 rate shock is the most important modern test of 60/40 portfolios. The toolkit makes visible why the defensive allocation outperforms in that scenario while the classic 60/40 breaks down.
- **LSEG data engineering**: The discovery and fallback layer demonstrates real awareness of data entitlements, missing history, and proxy substitution — a practical skill valued in quant risk and portfolio analytics roles.
- **Reproducible design**: Config-driven settings, parquet caching, and demo mode make the repo fully runnable in any environment.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Time series alignment, resampling, pivot tables |
| `numpy` | Return and covariance computations |
| `scipy` | Gaussian VaR z-scores |
| `matplotlib` | All charts |
| `seaborn` | Monthly returns heatmap |
| `PyYAML` | Config loading |
| `pyarrow` | Parquet cache I/O |
| `lseg-data` / `refinitiv-data` / `eikon` | LSEG market data (one required for live mode) |
