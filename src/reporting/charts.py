"""Chart generation for portfolio risk analytics.

All charts use a consistent clean style: white background, muted institutional
palette, minimal chrome. Figures are returned (not saved) so callers control
output paths.

Chart catalogue:
  plot_cumulative_performance   -> NAV index of one or more portfolios
  plot_drawdown                 -> Underwater equity curve (correct % scale)
  plot_rolling_volatility       -> Rolling annualised volatility
  plot_rolling_sharpe           -> Rolling Sharpe with regime shading
  plot_monthly_returns_heatmap  -> Calendar-style returns heat map
  plot_asset_allocation         -> Donut chart with asset-class colour families
  plot_contribution_bar         -> Horizontal bar of cumulative return contributions
  plot_stress_comparison        -> Scenario return bars (no-data excluded)
  plot_var_distribution         -> Return histogram with VaR / ES overlays
  plot_correlation_heatmap      -> Pairwise correlation matrix heatmap
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style constants ───────────────────────────────────────────────────────────

# Asset-class colour palette (used for allocation donut and legend grouping)
_AC_PALETTE = {
    "Equity":        "#2563EB",
    "Sovereign Bond":"#16A34A",
    "Credit":        "#D97706",
    "Commodity":     "#DC2626",
    "Real Estate":   "#7C3AED",
    "Cash":          "#6B7280",
}

# Per-sleeve colours — distinct shades within each asset-class family
# so the donut shows internal structure, not a monolithic blob
_SLEEVE_COLORS: Dict[str, str] = {
    # Equities — blue family (dark → light)
    "us_large_cap":       "#1E3A8A",
    "us_tech":            "#1D4ED8",
    "europe_equity":      "#2563EB",
    "uk_equity":          "#3B82F6",
    "japan_equity":       "#60A5FA",
    "emerging_markets":   "#0369A1",
    "us_small_cap":       "#0EA5E9",
    "global_min_vol":     "#38BDF8",
    # Sovereign bonds — green family
    "us_treasury_short":  "#14532D",
    "us_treasury_medium": "#166534",
    "euro_govt":          "#16A34A",
    "us_tips":            "#22C55E",
    "uk_gilts":           "#4ADE80",
    # Credit — amber/orange family
    "us_ig_credit":       "#78350F",
    "us_hy_credit":       "#92400E",
    "euro_ig_credit":     "#B45309",
    "euro_hy_credit":     "#D97706",
    # Commodities — red family
    "gold":               "#7F1D1D",
    "brent_crude":        "#991B1B",
    "broad_commodities":  "#DC2626",
    # Real Estate — purple
    "reits":              "#5B21B6",
    # Cash — grey
    "cash":               "#6B7280",
}

_LINE_COLORS = ["#1D4ED8", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]
_FIG_WIDTH = 12
_FIG_HEIGHT = 5
_DPI = 150
_FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family":        _FONT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         _DPI,
})

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save_or_return(fig: plt.Figure, path: Optional[str]) -> plt.Figure:
    if path:
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def _resolve_labels(
    ids: List[str],
    labels: Optional[Dict[str, str]],
) -> List[str]:
    """Map technical IDs to display labels if a mapping is provided."""
    if labels is None:
        return ids
    return [labels.get(i, i) for i in ids]


# ── 1. Cumulative Performance ─────────────────────────────────────────────────

def plot_cumulative_performance(
    nav_df: pd.DataFrame,
    title: str = "Cumulative Performance",
    labels: Optional[Dict[str, str]] = None,
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot NAV index series for one or more portfolios.

    Args:
        nav_df: DataFrame of NAV values (date index, one column per portfolio).
        title: Chart title.
        labels: Optional {column_id: display_name} mapping for the legend.
        subtitle: Optional disclosure line rendered below the title in grey italic.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 5.5))

    # Faint start-level reference
    ax.axhline(100, color="#9CA3AF", linewidth=0.7, linestyle="--", zorder=1)

    display_names = _resolve_labels(list(nav_df.columns), labels)

    for i, (col, name) in enumerate(zip(nav_df.columns, display_names)):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.plot(nav_df.index, nav_df[col], label=name, color=color,
                linewidth=1.8, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=26 if subtitle else 12)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", fontstyle="italic", annotation_clip=False)
    ax.set_ylabel("Portfolio Value (rebased to 100)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#E5E7EB")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 2. Drawdown ───────────────────────────────────────────────────────────────

def plot_drawdown(
    drawdown_series_dict: Dict[str, pd.Series],
    title: str = "Portfolio Drawdown",
    labels: Optional[Dict[str, str]] = None,
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot underwater equity curves.

    IMPORTANT: drawdown series must be in fractional form (e.g. -0.25 for -25%).
    PercentFormatter(xmax=1) converts fractions to percentages.
    Do NOT multiply the series by 100 before passing — that causes a 100× scale
    error on the y-axis.

    Args:
        drawdown_series_dict: Dict of {portfolio_id: drawdown_series} where
            values are in [-1, 0] fractional form.
        title: Chart title.
        labels: Optional {portfolio_id: display_name} mapping.
        subtitle: Optional disclosure line rendered below the title in grey italic.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    items = list(drawdown_series_dict.items())
    display_names = _resolve_labels([k for k, _ in items], labels)

    for i, ((name, series), display_name) in enumerate(zip(items, display_names)):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        # Fill only for the first (primary) portfolio to keep chart readable
        if i == 0:
            ax.fill_between(
                series.index, series, 0,
                alpha=0.18, color=color, zorder=2,
            )
        ax.plot(
            series.index, series,
            label=display_name, color=color,
            linewidth=1.5, zorder=3,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=26 if subtitle else 12)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", fontstyle="italic", annotation_clip=False)
    ax.set_ylabel("Drawdown (%)")

    # PercentFormatter(xmax=1) expects fractional values and formats as %.
    # e.g. -0.258 → "-26%"  (NOT -2580% — that bug came from multiplying
    # the data by 100 before this formatter, which double-scaled the axis).
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Tight y-limit: floor at 10 percentage points below worst drawdown,
    # capped at -80% to avoid extreme blank space on synthetic/low-vol data.
    worst = min(s.min() for s in drawdown_series_dict.values())
    ax.set_ylim(max(worst * 1.15, -0.80), 0.005)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="#E5E7EB")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 3. Rolling Volatility ─────────────────────────────────────────────────────

def plot_rolling_volatility(
    rolling_vol_dict: Dict[str, pd.Series],
    title: str = "Rolling 1-Year Volatility (Annualised)",
    labels: Optional[Dict[str, str]] = None,
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot rolling annualised volatility for one or more portfolios."""
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    items = list(rolling_vol_dict.items())
    display_names = _resolve_labels([k for k, _ in items], labels)

    # Shade a "calm regime" band (e.g. 6%–14% vol)
    # Computed from all series data range to stay data-driven
    all_vals = np.concatenate([s.dropna().values for _, s in items]) * 100
    p25, p75 = np.percentile(all_vals, 25), np.percentile(all_vals, 75)
    first_series = items[0][1]
    ax.fill_between(
        first_series.index, p25, p75,
        alpha=0.06, color="#6B7280", zorder=1,
        label=f"IQR range ({p25:.0f}%–{p75:.0f}%)",
    )

    for i, ((pid, series), name) in enumerate(zip(items, display_names)):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.plot(series.index, series * 100, label=name,
                color=color, linewidth=1.7, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=26 if subtitle else 12)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", fontstyle="italic", annotation_clip=False)
    ax.set_ylabel("Annualised Volatility (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate(rotation=30)
    # Suppress legend if only one portfolio — the title is sufficient context
    if len(items) > 1:
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="#E5E7EB")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 4. Rolling Sharpe ─────────────────────────────────────────────────────────

def plot_rolling_sharpe(
    rolling_sharpe_dict: Dict[str, pd.Series],
    title: str = "Rolling 1-Year Sharpe Ratio",
    labels: Optional[Dict[str, str]] = None,
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio with regime shading."""
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    items = list(rolling_sharpe_dict.items())
    first_series = items[0][1].dropna()
    idx = first_series.index

    # Compute actual data range from all series before plotting so regime bands
    # span the real y-extent (ax.get_ylim() before any plot returns default (0,1)).
    _all_vals = np.concatenate([s.dropna().values for _, s in items])
    _ymin = min(_all_vals.min(), -0.5) * 1.2
    _ymax = max(_all_vals.max(), 1.5) * 1.2

    # Regime shading — helps the reader interpret at a glance
    ax.fill_between(idx, 1, _ymax, alpha=0.04, color="#16A34A", zorder=1)  # above 1 → good
    ax.fill_between(idx, _ymin, 0,  alpha=0.04, color="#DC2626", zorder=1)  # below 0 → poor

    # Reference lines — clearly visible
    ax.axhline(0, color="#374151", linewidth=1.0, linestyle="--", zorder=2,
               label="Sharpe = 0")
    ax.axhline(1, color="#16A34A", linewidth=1.0, linestyle="--", zorder=2,
               alpha=0.7, label="Sharpe = 1")

    display_names = _resolve_labels([k for k, _ in items], labels)

    for i, ((pid, series), name) in enumerate(zip(items, display_names)):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.plot(series.index, series, label=name,
                color=color, linewidth=1.7, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=26 if subtitle else 12)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", fontstyle="italic", annotation_clip=False)
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate(rotation=30)

    # Show legend only if multiple portfolios; reference lines always labelled
    handles, lgs = ax.get_legend_handles_labels()
    ax.legend(handles, lgs, loc="upper right", framealpha=0.9,
              edgecolor="#E5E7EB", fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 5. Monthly Returns Heatmap ────────────────────────────────────────────────

def plot_monthly_returns_heatmap(
    monthly_table: pd.DataFrame,
    title: str = "Monthly Returns (%)",
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a calendar heatmap of monthly returns.

    Args:
        monthly_table: months × years DataFrame of decimal return fractions.
            Row index may be integers (1-12) or month-name strings.
        title: Chart title.
        save_path: Optional save path.
    """
    data_pct = monthly_table.copy() * 100

    # Map integer month indices to abbreviated names
    if len(data_pct.index) > 0 and isinstance(data_pct.index[0], (int, np.integer)):
        data_pct.index = [_MONTH_NAMES[int(m) - 1] for m in data_pct.index]

    n_years = data_pct.shape[1]
    # Scale figure: fixed height per row, width scales with years
    fig_w = max(10, n_years * 0.85)
    fig_h = max(5.5, len(data_pct) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    non_nan = data_pct.values[~np.isnan(data_pct.values)]
    vmax = max(abs(non_nan).max(), 1) if len(non_nan) else 5
    # Cap symmetric scale at 8% to keep typical months well differentiated
    vmax = min(vmax, 8.0)

    sns.heatmap(
        data_pct,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="#E5E7EB",
        cbar_kws={"label": "Monthly Return (%)", "shrink": 0.8},
        annot_kws={"size": 8, "weight": "normal"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=26 if subtitle else 14)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", fontstyle="italic", annotation_clip=False)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 6. Asset Allocation Donut ─────────────────────────────────────────────────

def plot_asset_allocation(
    weights: Dict[str, float],
    asset_class_map: Optional[Dict[str, str]] = None,
    name_map: Optional[Dict[str, str]] = None,
    title: str = "Portfolio Asset Allocation",
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a donut chart of portfolio weights by sleeve.

    Sleeves within the same asset class share a colour family, making the
    internal structure of the equity or bond allocation visible at a glance.
    """
    from src.data.mapping import get_asset_class_map, get_name_map
    ac_map = asset_class_map or get_asset_class_map()
    n_map = name_map or get_name_map()

    # Sort: by asset class order then by weight descending within class
    ac_order = ["Equity", "Sovereign Bond", "Credit", "Commodity",
                "Real Estate", "Cash"]

    def sort_key(item):
        sid, w = item
        ac = ac_map.get(sid, "Other")
        return (ac_order.index(ac) if ac in ac_order else 99, -w)

    sorted_weights = [(sid, w) for sid, w in sorted(weights.items(), key=sort_key)
                      if w > 0]

    labels, sizes, colors = [], [], []
    for sid, w in sorted_weights:
        labels.append(n_map.get(sid, sid))
        sizes.append(w)
        colors.append(_SLEEVE_COLORS.get(sid, _AC_PALETTE.get(ac_map.get(sid, ""), "#9CA3AF")))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 3.5 else "",
        startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 1.8},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_fontweight("bold")
        at.set_color("white")

    # Group legend entries by asset class with section labels
    legend_entries = []
    current_ac = None
    for (sid, w), label, wedge in zip(sorted_weights, labels, wedges):
        ac = ac_map.get(sid, "Other")
        if ac != current_ac:
            # Section header as a blank handle + bold label
            legend_entries.append((mpatches.Patch(color="none"), f"── {ac} ──"))
            current_ac = ac
        legend_entries.append((wedge, f"{label}  ({w*100:.1f}%)"))

    handles = [h for h, _ in legend_entries]
    leg_labels = [l for _, l in legend_entries]

    ax.legend(
        handles, leg_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5,
        framealpha=0.95,
        edgecolor="#E5E7EB",
        handlelength=1.2,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=26 if subtitle else 16)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", fontstyle="italic", annotation_clip=False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 7. Contribution Bar Chart ─────────────────────────────────────────────────

def plot_contribution_bar(
    contributions: pd.Series,
    name_map: Optional[Dict[str, str]] = None,
    title: str = "Cumulative Return Contribution by Sleeve (Full Period)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of cumulative return contributions.

    Args:
        contributions: Series indexed by sleeve_id, values are cumulative
            return contributions (decimal fractions) over the full backtest.
        name_map: Optional {sleeve_id: display_name} for y-axis labels.
        title: Chart title.
        save_path: Optional save path.
    """
    # Rename index to human-readable labels
    if name_map:
        idx_labels = [name_map.get(sid, sid) for sid in contributions.index]
        plot_series = pd.Series(contributions.values, index=idx_labels,
                                name=contributions.name)
    else:
        plot_series = contributions.copy()

    # Sort ascending so the largest contributor appears at top
    plot_series = plot_series.sort_values(ascending=True)
    colors = ["#DC2626" if v < 0 else "#1D4ED8" for v in plot_series]

    n = len(plot_series)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, n * 0.42)))

    bars = ax.barh(
        range(n), plot_series.values * 100,
        color=colors, height=0.65, zorder=3,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_series.index, fontsize=8.5)
    ax.axvline(0, color="#374151", linewidth=0.9, zorder=4)

    # Value labels on each bar
    for bar, val in zip(bars, plot_series.values):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = 0.3 if x >= 0 else -0.3
        ax.text(
            x + offset, bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center", ha=ha, fontsize=7.5, color="#374151",
        )

    ax.set_xlabel("Cumulative Return Contribution (%)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 8. Stress Test Comparison ─────────────────────────────────────────────────

def plot_stress_comparison(
    stress_df: pd.DataFrame,
    value_col: str = "portfolio_total_return",
    scenario_col: str = "scenario_name",
    group_col: Optional[str] = None,
    title: str = "Historical Stress Scenarios",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of portfolio returns across historical stress scenarios.

    Scenarios with status='no_data' are automatically excluded rather than
    shown as zero-height bars (which would be analytically misleading).

    Bars are coloured by sign: negative → red, positive → teal.
    """
    # Filter: only scenarios with actual data
    if "status" in stress_df.columns:
        plot_df = stress_df[stress_df["status"] == "ok"].copy()
    else:
        plot_df = stress_df.dropna(subset=[value_col]).copy()

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 4))
        ax.text(0.5, 0.5, "No scenario data available for this date range.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#6B7280")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()
        return _save_or_return(fig, save_path)

    if group_col and group_col in plot_df.columns:
        pivoted = (
            plot_df.pivot(index=scenario_col, columns=group_col, values=value_col)
            * 100
        )
    else:
        pivoted = plot_df.set_index(scenario_col)[[value_col]] * 100

    n_scenarios = len(pivoted)
    n_groups = len(pivoted.columns)
    fig_w = max(_FIG_WIDTH, n_scenarios * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    x = np.arange(n_scenarios)
    width = 0.65 / max(n_groups, 1)

    for i, col in enumerate(pivoted.columns):
        vals = pivoted[col].values
        offset = (i - n_groups / 2 + 0.5) * width

        # Colour by sign — green for positive, red for negative
        bar_colors = ["#16A34A" if v > 0 else "#DC2626" for v in vals]

        bars = ax.bar(
            x + offset, vals,
            width=width * 0.88, color=bar_colors,
            alpha=0.88, zorder=3,
            label=str(col) if n_groups > 1 else None,
        )

        # Data labels above/below each bar
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                y_pos = val + (0.4 if val >= 0 else -0.4)
                va = "bottom" if val >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos, f"{val:.1f}%",
                    ha="center", va=va,
                    fontsize=7.5, fontweight="bold",
                    color="#374151", zorder=5,
                )

    ax.axhline(0, color="#374151", linewidth=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pivoted.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Portfolio Return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Footnote: clarify if any scenarios were excluded due to data gaps
    if "status" in stress_df.columns:
        n_excluded = (stress_df["status"] == "no_data").sum()
        if n_excluded > 0:
            excluded_names = stress_df.loc[stress_df["status"] == "no_data", scenario_col].tolist()
            note = f"Note: {n_excluded} scenario(s) excluded — insufficient history: {', '.join(excluded_names)}"
            fig.text(0.5, -0.02, note, ha="center", fontsize=7.5,
                     color="#6B7280", style="italic")

    if n_groups > 1:
        ax.legend(loc="lower right", framealpha=0.9, edgecolor="#E5E7EB")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 9. VaR Distribution ───────────────────────────────────────────────────────

def plot_var_distribution(
    daily_returns: pd.Series,
    var_95: float,
    es_95: float,
    var_99: float,
    title: str = "Daily Return Distribution with VaR & ES",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of daily returns with VaR and ES vertical lines overlaid.

    Lines are ordered from most extreme (99% VaR, leftmost) to least extreme
    (95% VaR, rightmost) in both position and legend order.

    Args:
        daily_returns: Daily simple return Series.
        var_95: 95% Historical VaR as a positive loss fraction.
        es_95:  95% Historical ES as a positive loss fraction.
        var_99: 99% Historical VaR as a positive loss fraction.
        title: Chart title.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5))
    r = daily_returns.dropna() * 100

    ax.hist(r, bins=80, color="#3B82F6", alpha=0.55,
            edgecolor="white", linewidth=0.3, zorder=2)

    # Draw in order: 99% VaR (most extreme / leftmost), 95% ES, 95% VaR
    line_specs = [
        (-var_99 * 100, "#7C3AED", ":",  f"99% VaR  = -{var_99*100:.2f}%"),
        (-es_95  * 100, "#DC2626", "-.", f"95% ES   = -{es_95*100:.2f}%"),
        (-var_95 * 100, "#D97706", "--", f"95% VaR  = -{var_95*100:.2f}%"),
    ]
    for xval, color, ls, label in line_specs:
        ax.axvline(xval, color=color, linewidth=2.0, linestyle=ls,
                   label=label, zorder=4)

    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(framealpha=0.9, edgecolor="#E5E7EB", fontsize=9,
              loc="upper left")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 10. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Asset Return Correlations",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a labelled pairwise correlation matrix heatmap (lower triangle).

    Args:
        corr: Square correlation DataFrame (columns and index = display names).
        title: Chart title.
        save_path: Optional save path.
    """
    n = len(corr)
    fig_size = max(8.0, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdYlGn",
        vmin=-1.0, vmax=1.0, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": max(6, 9 - n // 5)},
        linewidths=0.4, linecolor="#E5E7EB",
        cbar_kws={"shrink": 0.7, "label": "Pearson correlation"},
        square=True,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return _save_or_return(fig, save_path)
