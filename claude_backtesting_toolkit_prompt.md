# Diversified Portfolio Backtesting & Risk Analytics Toolkit

## Goal
Build a **serious multi-asset portfolio backtesting and risk analytics project** in Python using **LSEG data**.

This is **not** mainly an optimizer project. The core objective is to build a **portfolio analytics and monitoring engine** for a diversified portfolio, with:
- stress testing
- VaR
- ES / CVaR
- drawdown analysis
- rolling risk metrics
- monthly performance reporting
- time-series analytics
- asset-class contribution analysis
- reproducible charts and exportable reports

The final repo must be **GitHub-ready, interview-defensible, and realistic** for a quant risk / portfolio analytics / multi-asset internship profile.

---

## Project Identity
**Business name:** Diversified Portfolio Backtesting & Risk Analytics Toolkit  
**GitHub repo name:** `portfolio-risk-backtesting-toolkit`

### Positioning
This project should look like a **risk and performance analytics engine** for a diversified portfolio manager, allocator, or quant risk team.

It should answer this question:

> How can we monitor, stress test, and explain the risk/performance of a diversified multi-asset portfolio using Python and LSEG data?

---

## What the project must do
Build a modular Python project that can:

1. Pull market data from LSEG for a diversified universe
2. Clean and align time series across all assets
3. Compute returns at daily and monthly frequency
4. Build portfolio-level performance series
5. Compute risk metrics:
   - annualized return
   - annualized volatility
   - Sharpe ratio
   - Sortino ratio
   - max drawdown
   - rolling volatility
   - rolling Sharpe
   - historical VaR
   - parametric VaR
   - historical ES / CVaR
6. Build stress testing tools:
   - predefined historical crisis windows
   - asset shock scenarios
   - portfolio shock decomposition
7. Produce monthly performance tables
8. Produce asset-class contribution and risk-contribution tables
9. Export charts and tables for a README and final report
10. Run even if some instruments are unavailable by using a clean proxy-replacement logic

---

## Very important design rule
Do **not** make this project a random collection of metrics.

It must feel like a **professional toolkit** with a coherent flow:

**data -> cleaning -> returns -> portfolio construction -> analytics -> stress testing -> reporting**

---

## Universe choice
I want a **large but still clean and defendable universe**.
Do **not** build a huge messy list of random instruments.

Use a **multi-asset sleeve universe** with around **18 to 28 sleeves**.
That is large enough to be serious, but still clean enough for explanation and reporting.

### Recommended universe structure
Use the best LSEG-accessible proxies the environment can actually retrieve.
If exact instruments are unavailable, replace them with the closest liquid proxy and document the substitution.

#### Equities
- US Large Cap Equity
- US Tech / Growth
- Europe Equity
- UK Equity
- Japan Equity
- Emerging Markets Equity
- Small Caps Equity
- Global Minimum Volatility or Defensive Equity sleeve

#### Sovereign Bonds
- US Treasuries 1-3Y
- US Treasuries 7-10Y
- Euro Government Bonds
- Germany Bunds
- UK Gilts
- Inflation-Linked Bonds proxy

#### Credit
- US Investment Grade Credit
- US High Yield Credit
- Euro Investment Grade Credit
- Euro High Yield Credit

#### Real Assets / Commodities
- Gold
- Brent Crude
- Broad Commodities basket
- REITs / Listed Real Estate

#### Optional diversifiers
- USD index proxy or major FX proxy
- Bitcoin proxy only if accessible and if clearly marked as optional/high-volatility diversifier
- ESG equity sleeve only if easy to source cleanly

### Why this universe is good
This universe is interesting because it lets the toolkit analyze:
- equity/bond diversification
- duration shocks
- credit spread stress
- commodity inflation sensitivity
- defensive vs risky sleeves
- cross-asset drawdown behavior

It is much more interesting than a single-asset-class project.

---

## Portfolio logic
The project should support multiple portfolio construction modes:

1. **Static benchmark portfolio**
   - example: 60/40 or a diversified strategic allocation
2. **User-defined weights portfolio**
3. **Simple rule-based rebalanced portfolio**
4. **Optional comparison portfolios**
   - Equal Weight sleeves
   - Risk-budgeted sleeves
   - Defensive allocation

Important: this project is **not mainly about advanced optimization**.  
The focus is on **backtesting + risk analytics + monitoring**.

---

## Core outputs required
The repo must produce at least these outputs:

### Tables
- `portfolio_summary.csv`
- `monthly_performance.csv`
- `drawdown_table.csv`
- `var_es_summary.csv`
- `stress_test_results.csv`
- `asset_class_contributions.csv`
- `rolling_metrics.csv`

### Charts
- cumulative performance
- drawdown chart
- rolling volatility
- rolling Sharpe
- monthly returns heatmap
- asset allocation chart
- asset class contribution chart
- stress test comparison chart

### README visuals
Save a clean subset of charts to a `docs/images/` folder for GitHub display.

---

## Historical stress windows to include
Use a reasonable list of historical episodes, depending on data coverage:
- COVID crash
- 2022 inflation / rate shock
- 2023 regional bank stress if relevant
- 2020 oil collapse for commodity sleeves
- optional: 2025-2026 recent stress only if data coverage is clean

Also include **custom shock scenarios**:
- equity shock
- rates up shock
- credit spread widening shock
- commodity spike shock
- mixed stagflation shock

---

## Risk methodology expectations
Implement and clearly document:

### VaR
- historical VaR
- parametric Gaussian VaR

### ES / CVaR
- historical expected shortfall

### Drawdowns
- rolling peak-to-trough drawdown
- max drawdown table

### Rolling metrics
- rolling volatility
- rolling Sharpe
- rolling beta if feasible

### Contribution analysis
- contribution to return
- contribution to volatility
- risk contribution by sleeve if feasible

Important: keep the methodology **clear and defensible**, not overly flashy.

---

## Data engineering expectations
Use LSEG professionally.

Claude must first build a **discovery / validation layer** that answers:
- which instruments are actually accessible
- which fields work reliably
- which proxies are best if the preferred instrument fails
- what date coverage is available
- whether adjusted / total return series are available

Create a reusable validation script like:
- `src/data/discovery.py`
- `scripts/validate_universe.py`

The project should not assume perfect data access.
It should gracefully handle:
- missing fields
- partial history
- unavailable tickers
- frequency mismatches

---

## Questions Claude must answer before coding too much
Claude must explicitly answer these questions in the repo planning stage:

1. What exact LSEG instruments are accessible in this environment?
2. Which field should be used for return calculation for each sleeve?
3. Should the project use price returns or total returns where available?
4. What should be the base currency?
5. How should FX conversion be handled?
6. What is the minimum common history across the chosen sleeves?
7. Which sleeves need proxy substitution?
8. How should missing data be filled or excluded?
9. What is the official portfolio rebalancing frequency?
10. Which stress windows are fully supported by the chosen history?
11. Which VaR / ES methods are included in v1, and which are optional?
12. What outputs are mandatory for README vs optional for notebooks?

---

## Repo structure expected
```text
portfolio-risk-backtesting-toolkit/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── config/
│   ├── settings.yaml
│   ├── universe.yaml
│   ├── portfolio_weights.yaml
│   └── stress_scenarios.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docs/
│   └── images/
├── notebooks/
│   ├── 01_universe_discovery.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_backtest_engine.ipynb
│   ├── 04_risk_analytics.ipynb
│   └── 05_stress_testing.ipynb
├── scripts/
│   ├── validate_universe.py
│   ├── run_backtest.py
│   └── build_demo_outputs.py
├── src/
│   ├── data/
│   │   ├── lseg_session.py
│   │   ├── discovery.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── mapping.py
│   ├── portfolio/
│   │   ├── weights.py
│   │   ├── rebalancing.py
│   │   └── construction.py
│   ├── analytics/
│   │   ├── returns.py
│   │   ├── performance.py
│   │   ├── risk.py
│   │   ├── var_es.py
│   │   ├── drawdown.py
│   │   ├── rolling.py
│   │   └── contributions.py
│   ├── stress/
│   │   ├── historical.py
│   │   ├── shocks.py
│   │   └── scenarios.py
│   ├── reporting/
│   │   ├── tables.py
│   │   ├── charts.py
│   │   └── export.py
│   └── utils/
│       ├── logging_utils.py
│       └── validation.py
├── tests/
│   ├── test_returns.py
│   ├── test_var_es.py
│   ├── test_drawdown.py
│   ├── test_stress.py
│   └── test_reporting.py
└── outputs/
    ├── tables/
    ├── charts/
    └── reports/
```

---

## Coding instructions for Claude
Be strict, honest, and professional.

### Required coding behavior
- write production-style Python
- modularize everything
- add type hints where reasonable
- include docstrings on important functions
- add clear error handling
- avoid fake data unless in explicit sample/demo mode
- never invent unavailable LSEG data
- document every fallback or proxy substitution
- keep naming consistent
- make charts clean and minimal

### Do not do this
- do not overengineer with useless abstractions
- do not create fake institutional language with weak methodology
- do not hide missing data issues
- do not pretend intraday data exists if it does not
- do not claim ESG or factor data unless it is actually retrieved

---

## README expectations
The README must clearly explain:
1. project purpose
2. universe
3. data source and LSEG dependency
4. methodology
5. risk metrics used
6. stress framework
7. main outputs
8. limitations
9. how to run
10. what makes the project professionally interesting

The README must sound like a serious quant / portfolio analytics project, not a student homework file.

---

## What I want Claude to do first
Before writing the full repo, do the following in order:

1. Audit the project idea critically
2. Confirm the best universe design
3. Propose exact sleeves / proxies to validate in LSEG
4. Design the repo structure
5. Write the universe validation script first
6. Write the data loader and cleaning pipeline
7. Then build the analytics engine
8. Then build stress testing
9. Then build reporting
10. Then improve README and outputs

---

## Final instruction to Claude
I want a project that is:
- credible
- reusable
- interview-defensible
- GitHub-ready
- clearly different from my already-finished thematic allocation project

This project must look like a **professional portfolio risk and backtesting toolkit**, not a second version of the same optimizer.
