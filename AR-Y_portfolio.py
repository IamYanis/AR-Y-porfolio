
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from risk_models import shrink_covariance
from portfolio_validation import (run_validation, write_report, minimum_risk,
                                  compute_frontier, maximum_sharpe, return_limits,
                                  feasible_cloud, exclusion_weights)
from price_cache import load_close
from fund_nav import load_fund_nav, ISIN as FUND_ISIN, NAME as FUND_NAME
from robustness import proposed_weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
# 1. SETTINGS
# ============================================================

parser = argparse.ArgumentParser(description="GBP portfolio frontier analysis")
parser.add_argument("--refresh-data", action="store_true",
                    help="Download and replace cached adjusted prices for the configured dates")
parser.add_argument("--bond-series", choices=["actual", "proxy"], default="actual",
                    help="Use actual BlackRock NAVs (default), or explicitly select EUNH.DE proxy")
parser.add_argument("--no-show", action="store_true", help="Save charts without opening a window")
parser.add_argument("--validate", action="store_true", help="Run rolling covariance and diversification checks")
parser.add_argument("--as-of", default=datetime.now(ZoneInfo("Europe/London")).date().isoformat(),
                    help="Exclude this date and later to avoid incomplete trading sessions (YYYY-MM-DD)")
parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "reports" / "latest"))
args = parser.parse_args()
AS_OF = pd.Timestamp(args.as_of)
if AS_OF > pd.Timestamp(datetime.now(ZoneInfo("Europe/London")).date()):
    raise ValueError("--as-of cannot be later than today in Europe/London")
OUTPUT_DIR = Path(args.output_dir).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent / "data" / "prices"

START = "2024-10-21"
END = "2026-10-31"  # Exclusive end date

TRADING_DAYS = 252
RISK_FREE_RATE = 0.03
MAX_POSITION = 0.20

N_FRONTIER_POINTS = 150
N_RANDOM_PORTFOLIOS = 100000
RANDOM_SEED = 42

# Historical returns are estimates, not forecasts.
# The analysis is in GBP and uses the same historical
# sample for both investment universes.


# ============================================================
# 2. CURRENT UMUSHROOM PORTFOLIO
# ============================================================

holdings = {
    "RTX":     4.26,
    "AZN.L":   4.20,
    "MU":      3.82,
    "LLY":     3.19,
    "BE":      3.03,
    "VST":     2.88,
    "IONQ":    2.75,
    "TTMI":    2.64,
    "NEE":     2.62,
    "SYK":     2.05,
    "CEG":     1.95,
    "AMZN":    1.79,
    "GOOGL":   1.71,
    FUND_ISIN: 6.98,  # Actual holding; calculation series is selected separately
    "CBGA.L":  7.04,
    "XDWS.L":  6.90,
    "XDWC.L":  6.88,
}

currencies = {
    "RTX": "USD",
    "AZN.L": "GBP",
    "MU": "USD",
    "LLY": "USD",
    "BE": "USD",
    "VST": "USD",
    "IONQ": "USD",
    "TTMI": "USD",
    "NEE": "USD",
    "SYK": "USD",
    "CEG": "USD",
    "AMZN": "USD",
    "GOOGL": "USD",
    FUND_ISIN: "EUR",
    "CBGA.L": "GBP",
    "XDWS.L": "USD",
    "XDWC.L": "USD",
}

existing_assets = list(holdings.keys())

# ============================================================
# 3. ADD NEW INVESTMENT CANDIDATES
# ============================================================

# These assets are NOT currently held.
# Zero current weight means the existing portfolio
# remains unchanged.

candidate_assets = {
    # TECHNOLOGY
    "AAPL": 0.0,
    "MSFT": 0.0,
    "NVDA": 0.0,
    "AVGO": 0.0,
    "AMD": 0.0,

    # FINANCIALS
    "JPM": 0.0,
    "GS": 0.0,
    "BAC": 0.0,
    "V": 0.0,

    # HEALTHCARE
    "JNJ": 0.0,
    "ABBV": 0.0,
    "UNH": 0.0,
    "MRK": 0.0,

    # INDUSTRIALS
    "CAT": 0.0,
    "DE": 0.0,
    "GE": 0.0,
    "HON": 0.0,

    # ENERGY
    "XOM": 0.0,
    "CVX": 0.0,
    "SLB": 0.0,

    # CONSUMER
    "WMT": 0.0,
    "COST": 0.0,
    "PG": 0.0,
    "KO": 0.0,

    # INTERNATIONAL EQUITIES
    "EWJ": 0.0,
    "EEM": 0.0,
    "VGK": 0.0,
    "EWU": 0.0,

    # BONDS
    "TLT": 0.0,
    "IEF": 0.0,
    "TIP": 0.0,
    "SHY": 0.0,

    # ALTERNATIVE ASSETS
    "IAU": 0.0,
    "GLD": 0.0,
    "VNQ": 0.0,
    "PDBC": 0.0,
}

holdings.update(candidate_assets)

for ticker in candidate_assets:
    currencies[ticker] = "USD"

# Calculate remaining GBP cash
invested_weight = sum(holdings.values())
cash_weight = 100 - invested_weight

if cash_weight < -1e-6:
    raise ValueError("Portfolio weights exceed 100%.")

holdings["CASH_GBP"] = cash_weight
existing_assets.append("CASH_GBP")

print("\nCURRENT UMUSHROOM PORTFOLIO")
print("=" * 45)

for ticker, weight in holdings.items():
    print(f"{ticker:12s}: {weight:6.2f}%")

print(f"Actual bond holding: {FUND_NAME}, {FUND_ISIN}; calculation series: {args.bond_series}")
pd.DataFrame([{"Asset": a, "Actual_Name": FUND_NAME if a == FUND_ISIN else a,
               "Actual_ISIN": FUND_ISIN if a == FUND_ISIN else "",
               "Weight_pct": w,
               "Calculation_Series": ("BlackRock official NAV" if args.bond_series == "actual" else "EUNH.DE ETF PROXY") if a == FUND_ISIN else a,
               "Status": ("actual NAV" if args.bond_series == "actual" else "PROXY: different fund, benchmark, fees and trading basis") if a == FUND_ISIN else "as configured"}
              for a,w in holdings.items() if w > 0]).to_csv(OUTPUT_DIR / "holdings_register.csv",index=False)

print(f"\nTotal weight: {sum(holdings.values()):.2f}%")
print(f"Existing assets (including cash): {len(existing_assets)}")
print(f"New candidates: {len(candidate_assets)}")


# ============================================================
# 4. DOWNLOAD HISTORICAL PRICES
# ============================================================

tickers = [t for t in holdings if t != "CASH_GBP"]

def download_close(ticker):
    if ticker == FUND_ISIN:
        if args.bond_series == "actual":
            return load_fund_nav(START, END, DATA_DIR.parent / "nav", args.refresh_data)
        print(f"PROXY: {FUND_NAME} ({FUND_ISIN}) represented by EUNH.DE")
        return load_close("EUNH.DE", START, END, DATA_DIR, yf.download,
                          refresh=args.refresh_data).rename(FUND_ISIN)
    return load_close(ticker, START, END, DATA_DIR, yf.download,
                      refresh=args.refresh_data)


prices = pd.concat(
    [download_close(t) for t in tickers],
    axis=1
).sort_index()


# ============================================================
# 5. CONVERT ALL PRICES TO GBP
# ============================================================

usd_gbp = download_close("USDGBP=X")
eur_gbp = download_close("EURGBP=X")

fx = pd.concat(
    [
        usd_gbp.rename("USDGBP"),
        eur_gbp.rename("EURGBP")
    ],
    axis=1
)

# Align currency rates to asset dates
# Preserve FX-only dates before alignment, then carry native asset prices
# across holidays BEFORE converting: otherwise foreign FX exposure is frozen.
fx = fx.reindex(fx.index.union(prices.index)).sort_index().ffill(limit=3).reindex(prices.index)
raw_prices = prices.copy()
prices = prices.ffill(limit=3)

prices_gbp = pd.DataFrame(index=prices.index)

for ticker in tickers:

    currency = currencies[ticker]

    if currency == "USD":
        prices_gbp[ticker] = (
            prices[ticker] * fx["USDGBP"]
        )

    elif currency == "EUR":
        prices_gbp[ticker] = (
            prices[ticker] * fx["EURGBP"]
        )

    elif currency == "GBP":
        prices_gbp[ticker] = prices[ticker]

    else:
        raise ValueError(
            f"Unknown currency for {ticker}"
        )

# GBp versus GBP quotation units do not affect returns.
# Missing days due to market holidays are forward-filled
# for up to 3 dates.

prices_gbp = (
    prices_gbp.loc[prices_gbp.index < AS_OF]

)

quality_rows = []
for ticker in tickers:
    raw = raw_prices.loc[raw_prices.index < AS_OF, ticker]
    valid = raw.dropna()
    quality_rows.append({"Asset": ticker, "Quote_Currency": currencies[ticker],
                         "First_Price": valid.index.min(), "Last_Price": valid.index.max(),
                         "Raw_Observations": len(valid), "Missing_Union_Dates": int(raw.isna().sum()),
                         "Filled_Union_Dates": int((raw.isna() & prices.loc[raw.index, ticker].notna()).sum()),
                         "Holding_Status": args.bond_series if ticker == FUND_ISIN else "as configured"})
pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "data_quality.csv", index=False)
if len(prices_gbp.dropna()) < 150:
    raise ValueError(
        "Insufficient common historical data. "
        "Check the asset tickers and available histories."
    )

print(
    f"\nCommon historical sample: "
    f"{prices_gbp.dropna().index.min().date()} to "
    f"{prices_gbp.index.max().date()}"
)

print(f"Common price observations: {len(prices_gbp.dropna())}")


# ============================================================
# 6. DAILY RETURNS
# ============================================================

# Calculate returns before dropping incomplete rows: never bridge a missing
# observation with a multi-session return and call it one daily observation.
returns = prices_gbp.pct_change(fill_method=None).dropna()

# Deterministic assumed cash yield
daily_cash_return = (
    (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
)

returns["CASH_GBP"] = daily_cash_return

assets = list(holdings.keys())
returns = returns[assets]


# ============================================================
# 7. EXPECTED RETURNS AND COVARIANCE
# ============================================================



mu = returns.mean() * TRADING_DAYS
risky_assets = [a for a in assets if a != "CASH_GBP"]
if not 50 <= len(risky_assets) <= 60:
    raise ValueError("Initial universe must contain 50–60 investments")
sample_cov = returns.cov() * TRADING_DAYS
risky_cov, shrinkage = shrink_covariance(returns[risky_assets])
cov = pd.DataFrame(0.0, index=assets, columns=assets)
cov.loc[risky_assets, risky_assets] = risky_cov * TRADING_DAYS
print(f"\nOAS covariance shrinkage: {shrinkage:.2%}")
print("Risky covariance condition number (sample / OAS):",
      np.linalg.cond(sample_cov.loc[risky_assets, risky_assets]),
      np.linalg.cond(risky_cov))
# Both universes use principal submatrices of this SAME fitted estimator.
# Cash return uses the same arithmetic annualisation as other assets.


mu_array = mu.to_numpy()
cov_matrix = cov.to_numpy()

cov_matrix = 0.5 * (
    cov_matrix + cov_matrix.T
)

n_assets = len(assets)

w_current = np.array([
    holdings[t] / 100
    for t in assets
])

print("\nANNUALISED HISTORICAL RETURN ESTIMATES")
print("=" * 45)
print((mu * 100).round(2).to_string())


# ============================================================
# 8. PORTFOLIO MATHEMATICS
# ============================================================

def portfolio_return(w):
    return float(w @ mu_array)


def portfolio_variance(w):
    return float(w @ cov_matrix @ w)


def portfolio_volatility(w):
    return np.sqrt(
        max(portfolio_variance(w), 0.0)
    )


def portfolio_sharpe(w):
    vol = portfolio_volatility(w)

    if vol < 1e-6:
        return np.nan

    return (
        portfolio_return(w) - daily_cash_return * TRADING_DAYS
    ) / vol


def portfolio_metrics(w):
    return {
        "Return": portfolio_return(w),
        "Volatility": portfolio_volatility(w),
        "Sharpe": portfolio_sharpe(w)
    }


# ============================================================
# 9. CONSTRAINTS AND BOTH PORTFOLIO FRONTIERS
# ============================================================

# Hold actual cash fixed in all investable comparisons. The 20% cap applies
# to each security; the configured 35.31% cash holding is exempt.
expanded_bounds = [(cash_weight/100, cash_weight/100) if a == "CASH_GBP"
                   else (0, MAX_POSITION) for a in assets]
original_bounds = [(0, 0) if a in candidate_assets else b
                   for a, b in zip(assets, expanded_bounds)]
print(f"\nOptimisation cash constraint: fixed at {cash_weight:.2f}%")

original_frontier = compute_frontier(cov_matrix, mu_array, original_bounds,
                                     N_FRONTIER_POINTS)
expanded_frontier = compute_frontier(cov_matrix, mu_array, expanded_bounds,
                                     N_FRONTIER_POINTS)
w_minvar_original = original_frontier["minvar_weights"]
w_minvar_expanded = expanded_frontier["minvar_weights"]
w_sharpe_original = maximum_sharpe(cov_matrix, mu_array, original_bounds,
                                  original_frontier, daily_cash_return*TRADING_DAYS)
w_sharpe = maximum_sharpe(cov_matrix, mu_array, expanded_bounds,
                         expanded_frontier, daily_cash_return*TRADING_DAYS)


def weights_at_target(bounds, target):
    return minimum_risk(cov_matrix, bounds, mu_array, target)


anchor = portfolio_return(w_current)
base_weights = weights_at_target(original_bounds, anchor)
full_weights = weights_at_target(expanded_bounds, anchor)
base_risk = portfolio_volatility(base_weights)
full_risk = portfolio_volatility(full_weights)
if (full_risk > base_risk + 1e-6 or
        portfolio_variance(w_minvar_expanded) > portfolio_variance(w_minvar_original)+1e-8):
    raise RuntimeError("Expanded universe unexpectedly increases optimal risk")

portfolios = {
    "Current": w_current,
    "Original Min Variance": w_minvar_original,
    "Expanded Min Variance": w_minvar_expanded,
    "Original Max Sharpe": w_sharpe_original,
    "Expanded Max Sharpe": w_sharpe,
    "Original Matched Return": base_weights,
    "Expanded Matched Return": full_weights,
}
comparison = pd.DataFrame({name: portfolio_metrics(w) for name, w in portfolios.items()})
metrics_current = portfolio_metrics(w_current)
print("\nPORTFOLIO COMPARISON (annualised estimates; returns and volatility in %)")
display = comparison.T.copy()
display[["Return", "Volatility"]] *= 100
print(display.round(4).to_string())
print(f"At the current estimated return, risk falls by "
      f"{100*(metrics_current['Volatility']-base_risk):.2f} pp with existing assets; "
      f"a further {100*(base_risk-full_risk):.2f} pp with candidates.")

# Scale only the five suggested positions, not the full optimised portfolio.
# Use this run's matched-return solution rather than stale hardcoded weights.
suggested_assets = ["PDBC", "JNJ", "KO", "CVX", "AAPL"]
suggested_weights = full_weights[[assets.index(a) for a in suggested_assets]]
fractional_rows = []
for fraction in [0., .25, .50, .75, 1.]:
    w = proposed_weights(w_current, assets, suggested_assets,
                         float(fraction*suggested_weights.sum()),
                         addition_weights=suggested_weights)
    box = np.asarray(expanded_bounds)
    if (not np.isfinite(w).all() or abs(w.sum()-1) > 1e-8 or
            (w < box[:, 0]-1e-8).any() or (w > box[:, 1]+1e-8).any()):
        raise RuntimeError(f"Invalid fractional allocation at {fraction:.0%}")
    metrics = portfolio_metrics(w)
    fractional_rows.append({"Fraction_pct": 100*fraction,
        "Suggested_Allocation_pct": 100*fraction*suggested_weights.sum(),
        "Cash_Weight_pct": 100*w[assets.index("CASH_GBP")],
        "Return_pct": 100*metrics["Return"],
        "Volatility_pct": 100*metrics["Volatility"], "Sharpe": metrics["Sharpe"],
        **{a+"_Weight_pct": 100*w[assets.index(a)] for a in suggested_assets}})
fractional_scenarios = pd.DataFrame(fractional_rows)
fractional_scenarios.to_csv(OUTPUT_DIR / "umushroom_fractional_allocations.csv", index=False)
print("\nFRACTIONAL SUGGESTED ALLOCATIONS (estimated returns are not held fixed)")
print(fractional_scenarios.round(4).to_string(index=False))


# ============================================================
# 10. ONE WEIGHT COMPARISON AND STRAIGHTFORWARD ADD/REMOVE CHECKS
# ============================================================

weights_table = pd.DataFrame({"Asset": assets,
                              **{name: 100*w for name, w in portfolios.items()}})
for name in portfolios:
    if name != "Current":
        weights_table[name + " Change_pp"] = weights_table[name]-weights_table["Current"]
weights_table.to_csv(OUTPUT_DIR / "umushroom_expanded_weights.csv", index=False)

rankings = []
for ticker in candidate_assets:
    i = assets.index(ticker)
    add_bounds = list(original_bounds)
    add_bounds[i] = expanded_bounds[i]
    add_weights = weights_at_target(add_bounds, anchor)
    without_weights = exclusion_weights(cov_matrix, mu_array, expanded_bounds, i, anchor)
    if without_weights is None:
        raise RuntimeError("Removing a candidate should retain the feasible existing portfolio")
    add_risk = portfolio_volatility(add_weights)
    if add_risk > base_risk + 1e-6:
        raise RuntimeError(f"Adding {ticker} unexpectedly increases optimal risk")
    rankings.append({
        "Asset": ticker,
        "Add_One_Risk_Reduction_pp": 100*(base_risk-add_risk),
        "Removal_Risk_Increase_pp": 100*(portfolio_volatility(without_weights)-full_risk),
        "Add_One_Weight_pct": 100*add_weights[i],
        "Expanded_Weight_pct": 100*full_weights[i],
    })
ranking = pd.DataFrame(rankings).sort_values("Add_One_Risk_Reduction_pp", ascending=False)
ranking.to_csv(OUTPUT_DIR / "umushroom_diversification_ranking.csv", index=False)

review_rows = []
for ticker in existing_assets:
    if ticker == "CASH_GBP":
        continue
    i = assets.index(ticker)
    row = {"Asset": ticker, "Current_Weight_pct": 100*w_current[i],
           "Existing_Weight_pct": 100*base_weights[i],
           "Expanded_Weight_pct": 100*full_weights[i],
           "Existing_Change_pp": 100*(base_weights[i]-w_current[i]),
           "Expanded_Change_pp": 100*(full_weights[i]-w_current[i]),
           "Negligible_Existing_Weight": bool(base_weights[i] <= .001),
           "Negligible_Expanded_Weight": bool(full_weights[i] <= .001)}
    for label, bounds, risk in [("Existing", original_bounds, base_risk),
                                ("Expanded", expanded_bounds, full_risk)]:
        without = exclusion_weights(cov_matrix, mu_array, bounds, i, anchor)
        row[label + "_Exclusion_Status"] = "infeasible at target" if without is None else "ok"
        delta = np.nan if without is None else 100*(portfolio_volatility(without)-risk)
        if without is not None and delta < -1e-4:
            raise RuntimeError(f"Excluding {ticker} unexpectedly improves optimal risk")
        row[label + "_Exclusion_Risk_Increase_pp"] = delta
    review_rows.append(row)
review = pd.DataFrame(review_rows).sort_values("Existing_Exclusion_Risk_Increase_pp",
                                             ascending=False, na_position="first")
review.to_csv(OUTPUT_DIR / "umushroom_existing_positions.csv", index=False)
print("\nTOP ADDITIONS AT THE CURRENT ESTIMATED RETURN (in-sample only)")
print(ranking.head(10).round(4).to_string(index=False))
print("Negligible means at most 0.1% of portfolio value; a zero weight is not a sell recommendation.")


# ============================================================
# 11. TWO DISTINCT VISUALISATIONS
# ============================================================

colors = {"Existing": "#c16b16", "Expanded": "#193e70"}
series_label = ("BlackRock official NAV" if args.bond_series == "actual"
                else "EUNH.DE PROXY for the BlackRock fund")
sample_label = (f"{series_label} · {returns.index.min().date()} to "
                f"{returns.index.max().date()}")


def plot_boundary(ax, frontier, bounds, name, lower=False):
    cloud = feasible_cloud(bounds, N_RANDOM_PORTFOLIOS, RANDOM_SEED)
    vols = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", cloud, cov_matrix, cloud), 0))
    ax.scatter(100*vols, 100*(cloud @ mu_array), s=3, alpha=.08,
               color=colors[name], rasterized=True)
    ax.plot(100*frontier["vols"], 100*frontier["returns"], lw=2.5,
            color=colors[name], label=name + " efficient frontier")
    if lower and len(frontier["inefficient_returns"]):
        ax.plot(100*frontier["inefficient_vols"], 100*frontier["inefficient_returns"],
                ":", lw=2, color=colors[name], label=name + " inefficient branch")
    w = frontier["minvar_weights"]
    ax.scatter(100*portfolio_volatility(w), 100*portfolio_return(w), s=85,
               color=colors[name], edgecolor="white", zorder=6,
               label=name + " minimum variance")


def finish_plot(fig, ax, title, note, filename):
    ax.set(xlabel="Annualised volatility (%)", ylabel="Estimated annualised return (%)",
           title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.text(.5, .015, note, ha="center", fontsize=8, color="#454545")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    if args.no_show:
        plt.close(fig)


# A: same fitted risky covariance, zero cash, unchanged 20% security cap.
fig, ax = plt.subplots(figsize=(12, 8))
for name, bounds in [("Existing", original_bounds), ("Expanded", expanded_bounds)]:
    risky_bounds = [(0, 0) if a == "CASH_GBP" else b for a, b in zip(assets, bounds)]
    frontier = compute_frontier(cov_matrix, mu_array, risky_bounds,
                                N_FRONTIER_POINTS, include_inefficient=True)
    plot_boundary(ax, frontier, risky_bounds, name, lower=True)
finish_plot(fig, ax, "A — Classical Markowitz portfolios · risky assets only (GBP)\n" + sample_label,
            "Long-only; 20% cap per security; cash excluded. Clouds sample part of the feasible region.\n"
            "Dotted lower boundaries are inefficient. Historical means are not forecasts.",
            "umushroom_classical_frontiers.png")

# B: retain actual cash, and mark the exact current-return optimisations.
fig, ax = plt.subplots(figsize=(12, 8))
for name, frontier, bounds, matched_w, sharpe_w in [
        ("Existing", original_frontier, original_bounds, base_weights, w_sharpe_original),
        ("Expanded", expanded_frontier, expanded_bounds, full_weights, w_sharpe)]:
    plot_boundary(ax, frontier, bounds, name)
    for w, marker, label in [(matched_w, "^", "fully optimised at current return"),
                              (sharpe_w, "D", "efficient-branch max Sharpe")]:
        ax.scatter(100*portfolio_volatility(w), 100*portfolio_return(w), marker=marker,
                   s=100, color=colors[name], edgecolor="white", zorder=8,
                   label=name + " " + label)
ax.scatter(100*metrics_current["Volatility"], 100*metrics_current["Return"],
           marker="*", s=240, color="#c43e3e", edgecolor="black", zorder=10,
           label="Current portfolio")
ax.plot(fractional_scenarios.Volatility_pct, fractional_scenarios.Return_pct,
        "o-", color="#18856b", lw=2, ms=5, zorder=9,
        label="Fraction of five suggested allocations")
for row in fractional_scenarios.itertuples(index=False):
    offset = (-8, 12) if row.Fraction_pct == 100 else (5, 12 if row.Fraction_pct in [25, 75] else -14)
    ax.annotate(f"{row.Fraction_pct:.0f}%", (row.Volatility_pct, row.Return_pct),
                xytext=offset, textcoords="offset points", fontsize=8,
                ha="right" if row.Fraction_pct == 100 else "left",
                color="#12624f", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5),
                zorder=11)

# A few reference assets only; a standalone security breaches the 20% cap.
references = list(ranking.sort_values("Expanded_Weight_pct", ascending=False).head(3).Asset)
for ticker in references + ["CBGA.L", FUND_ISIN]:
    i = assets.index(ticker)
    x, y = 100*np.sqrt(cov_matrix[i, i]), 100*mu_array[i]
    label = ("BlackRock NAV" if args.bond_series == "actual" else "EUNH.DE proxy") if ticker == FUND_ISIN else ticker
    ax.scatter(x, y, s=22, color="#777777", zorder=4)
    ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
finish_plot(fig, ax, f"B — Actual portfolio optimisation · GBP cash fixed at {cash_weight:.2f}%\n" + sample_label,
            "Long-only; 20% cap per security. Triangles preserve the current estimated return.\n"
            "Grey assets are 100% reference points, outside portfolio constraints. Estimates are not forecasts.",
            "umushroom_expanded_frontiers.png")
if not args.no_show:
    plt.show()


# ============================================================
# 12. PRESERVE SUPPORTING CSV AND RESEARCH REPORT INTERFACES
# ============================================================

# Compare the full common feasible return range; flag dominated lower points.
low, high = return_limits(mu_array, original_bounds)
matched_rows = []
for target in np.linspace(low, high, 30):
    wo = weights_at_target(original_bounds, target)
    we = weights_at_target(expanded_bounds, target)
    old_risk, new_risk = portfolio_volatility(wo), portfolio_volatility(we)
    if new_risk > old_risk + 1e-6:
        raise RuntimeError("Expanded universe unexpectedly increases optimal risk")
    matched_rows.append({"Target_Return": target, "Existing_Volatility": old_risk,
                         "Expanded_Volatility": new_risk,
                         "Volatility_Reduction_pp": 100*(old_risk-new_risk),
                         "Existing_Branch": "efficient" if target >= portfolio_return(w_minvar_original)-1e-10 else "inefficient",
                         "Expanded_Branch": "efficient" if target >= portfolio_return(w_minvar_expanded)-1e-10 else "inefficient"})
matched = pd.DataFrame(matched_rows)
matched.to_csv(OUTPUT_DIR / "umushroom_matched_frontiers.csv", index=False)
pd.DataFrame([{"Portfolio": name, **portfolio_metrics(w)} for name, w in [
    ("Current holdings", w_current), ("Existing universe at current return", base_weights),
    ("Expanded universe at current return", full_weights)]]).to_csv(
        OUTPUT_DIR / "matched_current_return.csv", index=False)
pd.DataFrame({"Asset": assets, "Existing_Weight": base_weights,
              "Expanded_Weight": full_weights}).to_csv(OUTPUT_DIR / "matched_current_weights.csv", index=False)
for filename, frontier in [("umushroom_original_frontier.csv", original_frontier),
                            ("umushroom_expanded_frontier.csv", expanded_frontier)]:
    pd.DataFrame({"Volatility": frontier["vols"], "Expected_Return": frontier["returns"],
                  "Sharpe": [portfolio_sharpe(w) for w in frontier["weights"]]}).to_csv(
                      OUTPUT_DIR / filename, index=False)
comparison.to_csv(OUTPUT_DIR / "umushroom_portfolio_comparison.csv")
# Keep this legacy allocation-only export for downstream users.
weights_table[weights_table.Asset.isin(candidate_assets)].sort_values(
    "Expanded Max Sharpe", ascending=False).to_csv(
        OUTPUT_DIR / "umushroom_candidate_allocations.csv", index=False)

# Reproducible research outputs; no trades or changes to recorded holdings.
returns.to_csv(OUTPUT_DIR / "gbp_daily_returns.csv", index_label="Date")
validation = None
if args.validate:
    validation = run_validation(returns, holdings, candidate_assets, MAX_POSITION, OUTPUT_DIR)
write_report(OUTPUT_DIR, returns, holdings, candidate_assets, comparison, matched,
             ranking, cov, sample_cov, shrinkage, validation, args.as_of, DATA_DIR, args.bond_series)
print(f"\nAnalysis complete. Outputs: {OUTPUT_DIR}")
print("Plots: umushroom_classical_frontiers.png, umushroom_expanded_frontiers.png")
print("Weights: umushroom_expanded_weights.csv (percentages and percentage-point changes)")
print("Additions: umushroom_diversification_ranking.csv")
print("Existing positions: umushroom_existing_positions.csv")
print("Fractional scenarios: umushroom_fractional_allocations.csv")
print(f"Summary: {OUTPUT_DIR / 'BASELINE_REPORT.md'}")
