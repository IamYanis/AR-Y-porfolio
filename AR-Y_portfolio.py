
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from risk_models import shrink_covariance
from portfolio_validation import run_validation, write_report, minimum_risk
from price_cache import load_close
from fund_nav import load_fund_nav, ISIN as FUND_ISIN, NAME as FUND_NAME

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.optimize import minimize, linprog

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
# 9. CONSTRAINTS
# ============================================================

# Max 20% for individual assets.
# GBP cash can be between 0% and 100%.

expanded_bounds = [
    (0, 1) if ticker == "CASH_GBP"
    else (0, MAX_POSITION)
    for ticker in assets
]

# The original universe excludes candidates
# by fixing their weights at zero.

original_bounds = [
    (0, 0) if ticker in candidate_assets
    else bound
    for ticker, bound in zip(
        assets, expanded_bounds
    )
]

constraints_sum = {
    "type": "eq",
    "jac": lambda w: np.ones(n_assets),
    "fun": lambda w: np.sum(w) - 1
}


# ============================================================
# 10. GENERAL OPTIMISATION FUNCTION
# ============================================================

def solve_portfolio(
    objective,
    constraints,
    bounds,
    x0
):
    if objective is portfolio_variance:
        target = -constraints[1]["fun"](np.zeros(n_assets)) if len(constraints) > 1 else None
        return minimum_risk(cov_matrix, bounds, mu_array if target is not None else None, target)
    result = minimize(
        (lambda w: 100 * portfolio_variance(w))
        if objective is portfolio_variance else objective,
        jac=(lambda w: 200 * cov_matrix @ w)
        if objective is portfolio_variance else None,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 3000,
            "ftol": 1e-12
        }
    )

    if not result.success:
        raise RuntimeError(result.message)

    w = result.x

    if abs(np.sum(w) - 1) > 1e-5:
        raise RuntimeError("Weights do not sum to 1")

    for weight, (lower, upper) in zip(w, bounds):
        if (
            weight < lower - 1e-5
            or weight > upper + 1e-5
        ):
            raise RuntimeError(
                "Optimised weights violate bounds"
            )

    return w


# ============================================================
# 11. EFFICIENT FRONTIER FUNCTION
# ============================================================

def compute_frontier(
    bounds,
    n_points=N_FRONTIER_POINTS
):

    # Global minimum variance
    w_min = solve_portfolio(
        portfolio_variance,
        [constraints_sum],
        bounds,
        w_current.copy()
    )

    # Maximum feasible estimated return
    max_result = linprog(
        c=-mu_array,
        A_eq=np.ones((1, n_assets)),
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs"
    )

    if not max_result.success:
        raise RuntimeError(
            "Maximum return optimisation failed: "
            + max_result.message
        )

    w_max = max_result.x

    min_return = portfolio_return(w_min)
    max_return = portfolio_return(w_max)

    targets = np.linspace(
        min_return,
        max_return,
        n_points
    )

    frontier_returns = []
    frontier_vols = []
    frontier_weights = []

    for target in targets:

        constraints = [
            constraints_sum,
            {
                "type": "eq",
                "jac": lambda w: mu_array,
                "fun": lambda w, t=target:
                    portfolio_return(w) - t
            }
        ]

        # Feasible initial guess
        if abs(max_return - min_return) < 1e-12:
            initial = w_min.copy()
        else:
            alpha = (
                (target - min_return)
                / (max_return - min_return)
            )

            initial = (
                (1 - alpha) * w_min
                + alpha * w_max
            )

        try:
            w = solve_portfolio(
                portfolio_variance,
                constraints,
                bounds,
                initial
            )

        except RuntimeError:
            print(
                f"Skipping frontier target "
                f"{target:.2%}"
            )
            continue

        if abs(portfolio_return(w) - target) > 1e-5:
            continue

        frontier_returns.append(
            portfolio_return(w)
        )

        frontier_vols.append(
            portfolio_volatility(w)
        )

        frontier_weights.append(w)

    if not frontier_weights:
        raise RuntimeError(
            "No feasible frontier points found"
        )

    return {
        "vols": np.array(frontier_vols),
        "returns": np.array(frontier_returns),
        "weights": np.array(frontier_weights),
        "minvar_weights": w_min
    }


# ============================================================
# 12. CALCULATE BOTH EFFICIENT FRONTIERS
# ============================================================

print("\nCalculating original efficient frontier...")

original_frontier = compute_frontier(
    original_bounds
)

print("\nCalculating expanded efficient frontier...")

expanded_frontier = compute_frontier(
    expanded_bounds
)

# Original frontier data
original_vols = original_frontier["vols"]
original_returns = original_frontier["returns"]

# Expanded frontier data
frontier_vols = expanded_frontier["vols"]
frontier_returns = expanded_frontier["returns"]
frontier_weights = expanded_frontier["weights"]

# Minimum variance portfolios
w_minvar_original = original_frontier["minvar_weights"]
w_minvar_expanded = expanded_frontier["minvar_weights"]

# Maximum Sharpe on expanded frontier
frontier_sharpes = np.array([
    portfolio_sharpe(w)
    for w in frontier_weights
])

finite_sharpes = np.where(
    np.isfinite(frontier_sharpes),
    frontier_sharpes,
    -np.inf
)

if not np.any(np.isfinite(frontier_sharpes)):
    raise RuntimeError(
        "Cannot estimate a valid maximum Sharpe portfolio"
    )

idx_sharpe = np.argmax(finite_sharpes)
w_sharpe = frontier_weights[idx_sharpe]


# ============================================================
# 13. COMPARE PORTFOLIOS
# ============================================================

metrics_current = portfolio_metrics(w_current)

metrics_original_minvar = portfolio_metrics(
    w_minvar_original
)

metrics_expanded_minvar = portfolio_metrics(
    w_minvar_expanded
)

metrics_sharpe = portfolio_metrics(w_sharpe)

comparison = pd.DataFrame({
    "Current": metrics_current,
    "Original Min Variance": metrics_original_minvar,
    "Expanded Min Variance": metrics_expanded_minvar,
    "Expanded Max Sharpe": metrics_sharpe
})

print("\nPORTFOLIO COMPARISON")
print("=" * 65)

print(comparison.round(4))


# ============================================================
# 14. RANDOM PORTFOLIOS + TWO SIDE-BY-SIDE PLOTS
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

random_weights = rng.dirichlet(
    np.ones(n_assets),
    size=N_RANDOM_PORTFOLIOS
)

# Keep feasible random portfolios
upper_bounds = np.array([
    bound[1]
    for bound in expanded_bounds
])

valid = np.all(
    random_weights <= upper_bounds + 1e-10,
    axis=1
)

random_weights = random_weights[valid]

# Expected portfolio returns
random_returns = random_weights @ mu_array

# Portfolio variances
random_variances = np.einsum(
    "ij,jk,ik->i",
    random_weights,
    cov_matrix,
    random_weights
)

random_vols = np.sqrt(
    np.maximum(random_variances, 0.0)
)

print(
    f"\nRandom feasible portfolios: "
    f"{len(random_weights)}"
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(17, 7)
)

# LEFT: decimal values
# RIGHT: percentages

for ax, scale, title in zip(
    axes,
    [1, 100],
    [
        "Expected Return vs Volatility",
        "Expected Return (%) vs Volatility (%)"
    ]
):

    # Possible portfolios
    ax.scatter(
        random_vols * scale,
        random_returns * scale,
        s=8,
        alpha=0.20,
        color="steelblue",
        label="Possible Expanded Portfolios"
    )

    # Existing investment universe
    ax.plot(
        original_vols * scale,
        original_returns * scale,
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        label="Existing Universe Frontier",
        zorder=7
    )

    # Expanded investment universe
    ax.plot(
        frontier_vols * scale,
        frontier_returns * scale,
        color="darkblue",
        linewidth=3,
        label="Expanded Universe Frontier",
        zorder=8
    )

    # Our current portfolio
    ax.scatter(
        metrics_current["Volatility"] * scale,
        metrics_current["Return"] * scale,
        color="red",
        marker="*",
        s=350,
        edgecolor="black",
        zorder=12,
        label="Current Umushroom Portfolio"
    )

    # Original minimum variance
    ax.scatter(
        metrics_original_minvar["Volatility"] * scale,
        metrics_original_minvar["Return"] * scale,
        color="green",
        marker="o",
        s=120,
        zorder=10,
        label="Original Minimum Variance"
    )

    # Expanded minimum variance
    ax.scatter(
        metrics_expanded_minvar["Volatility"] * scale,
        metrics_expanded_minvar["Return"] * scale,
        color="limegreen",
        marker="s",
        s=110,
        edgecolor="black",
        zorder=10,
        label="Expanded Minimum Variance"
    )

    # Maximum Sharpe
    ax.scatter(
        metrics_sharpe["Volatility"] * scale,
        metrics_sharpe["Return"] * scale,
        color="purple",
        marker="D",
        s=130,
        zorder=11,
        label="Expanded Maximum Sharpe"
    )

    # Individual investments
    for i, ticker in enumerate(assets):

        if ticker == "CASH_GBP":
            continue

        asset_vol = np.sqrt(
            max(cov_matrix[i, i], 0)
        )

        asset_return = mu_array[i]

        # New assets are displayed in purple;
        # existing assets are grey.
        if ticker in candidate_assets:
            point_color = "mediumpurple"
        else:
            point_color = "grey"

        ax.scatter(
            asset_vol * scale,
            asset_return * scale,
            color=point_color,
            s=30,
            zorder=6
        )

        ax.annotate(
            ticker,
            xy=(
                asset_vol * scale,
                asset_return * scale
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7
        )

    if scale == 1:
        ax.set_xlabel("Annualised Volatility")
        ax.set_ylabel("Estimated Annual Return")
    else:
        ax.set_xlabel("Annualised Volatility (%)")
        ax.set_ylabel("Estimated Annual Return (%)")

    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc="best")

fig.suptitle(
    "A-Y Capital Partners — Markowitz Portfolio Optimisation\n"
    f"Existing vs Expanded Investment Universe (GBP); bond series: {args.bond_series}",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "umushroom_expanded_frontiers.png",
    dpi=200,
    bbox_inches="tight"
)

if args.no_show:
    plt.close(fig)
else:
    plt.show()


# ============================================================
# 15. COMPARE PORTFOLIO WEIGHTS
# ============================================================

weights_table = pd.DataFrame({
    "Asset": assets,
    "Current": w_current * 100,
    "Original Min Variance": w_minvar_original * 100,
    "Expanded Min Variance": w_minvar_expanded * 100,
    "Expanded Max Sharpe": w_sharpe * 100
})

print("\nPORTFOLIO WEIGHTS (%)")
print("=" * 80)

print(
    weights_table.round(2).to_string(index=False)
)

weights_table.to_csv(
    OUTPUT_DIR / "umushroom_expanded_weights.csv",
    index=False
)


# ============================================================
# 16. CANDIDATE ALLOCATIONS (NOT A DIVERSIFICATION RANKING)
# ============================================================

candidate_results = weights_table[
    weights_table["Asset"].isin(candidate_assets)
].copy()

candidate_results = candidate_results.sort_values(
    "Expanded Max Sharpe",
    ascending=False
)

print("\nNEW INVESTMENT CANDIDATES")
print("=" * 65)

print(
    candidate_results.round(2).to_string(index=False)
)

candidate_results.to_csv(
    OUTPUT_DIR / "umushroom_candidate_allocations.csv",
    index=False
)


# ============================================================
# 17. MEASURE MINIMUM-VARIANCE IMPROVEMENT
# ============================================================

def weights_at_target(bounds, target):
    """Minimum risk at a fixed historical return, with a feasible LP start."""
    feasible = linprog(
        np.zeros(n_assets),
        A_eq=np.vstack([np.ones(n_assets), mu_array]),
        b_eq=[1.0, target], bounds=bounds, method="highs"
    )
    if not feasible.success:
        raise RuntimeError(f"Infeasible comparison return: {target:.2%}")
    return solve_portfolio(
        portfolio_variance,
        [constraints_sum, {"type": "eq",
                           "jac": lambda w: mu_array,
                           "fun": lambda w: portfolio_return(w) - target}],
        bounds, feasible.x
    )


# Unrestricted cash makes global minimum variance trivial. Use a common
# target grid instead, and the current portfolio's return as an anchor.
common_low = max(original_returns.min(), frontier_returns.min())
common_high = min(original_returns.max(), frontier_returns.max())
common_targets = np.linspace(common_low, common_high, 30)
matched_rows = []
for target in common_targets:
    wo = weights_at_target(original_bounds, target)
    we = weights_at_target(expanded_bounds, target)
    old_risk, new_risk = portfolio_volatility(wo), portfolio_volatility(we)
    if new_risk > old_risk + 1e-5:
        raise RuntimeError("Expanded universe unexpectedly increases optimal risk")
    matched_rows.append({
        "Target_Return": target, "Existing_Volatility": old_risk,
        "Expanded_Volatility": new_risk,
        "Volatility_Reduction_pp": 100 * (old_risk - new_risk)
    })
matched = pd.DataFrame(matched_rows)
matched.to_csv(OUTPUT_DIR / "umushroom_matched_frontiers.csv", index=False)

anchor = portfolio_return(w_current)
base_weights = weights_at_target(original_bounds, anchor)
full_weights = weights_at_target(expanded_bounds, anchor)
base_risk = portfolio_volatility(base_weights)
full_risk = portfolio_volatility(full_weights)
pd.DataFrame([
    {"Portfolio": label, **portfolio_metrics(w)}
    for label,w in [("Current holdings",w_current),
                    ("Existing universe at current return",base_weights),
                    ("Expanded universe at current return",full_weights)]
]).to_csv(OUTPUT_DIR / "matched_current_return.csv",index=False)
pd.DataFrame({"Asset":assets,"Existing_Weight":base_weights,
              "Expanded_Weight":full_weights}).to_csv(OUTPUT_DIR / "matched_current_weights.csv",index=False)
rankings = []
for ticker in candidate_assets:
    i = assets.index(ticker)
    add_bounds = list(original_bounds)
    add_bounds[i] = expanded_bounds[i]
    add_weights = weights_at_target(add_bounds, anchor)
    without_bounds = list(expanded_bounds)
    without_bounds[i] = (0, 0)
    without_weights = weights_at_target(without_bounds, anchor)
    rankings.append({
        "Asset": ticker,
        "Add_One_Risk_Reduction_pp": 100 * (
            base_risk - portfolio_volatility(add_weights)),
        "Removal_Risk_Increase_pp": 100 * (
            portfolio_volatility(without_weights) - full_risk),
        "Expanded_Weight_pct": 100 * full_weights[i],
    })
ranking = pd.DataFrame(rankings).sort_values(
    "Add_One_Risk_Reduction_pp", ascending=False
)
ranking.to_csv(OUTPUT_DIR / "umushroom_diversification_ranking.csv", index=False)
print("\nDIVERSIFICATION AT MATCHED HISTORICAL RETURN")
print(f"Target: {anchor:.2%}; existing risk: {base_risk:.2%}; "
      f"expanded risk: {full_risk:.2%}")
print(ranking.round(4).to_string(index=False))
print("In-sample estimates: validate on later data before interpreting as durable benefits.")

# ============================================================
# 18. SAVE FRONTIER DATA
# ============================================================

original_frontier_table = pd.DataFrame({
    "Volatility": original_vols,
    "Expected_Return": original_returns
})

expanded_frontier_table = pd.DataFrame({
    "Volatility": frontier_vols,
    "Expected_Return": frontier_returns,
    "Sharpe": frontier_sharpes
})

original_frontier_table.to_csv(
    OUTPUT_DIR / "umushroom_original_frontier.csv",
    index=False
)

expanded_frontier_table.to_csv(
    OUTPUT_DIR / "umushroom_expanded_frontier.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "umushroom_portfolio_comparison.csv"
)

print("\nANALYSIS COMPLETE")
print("=" * 65)

print("Files saved:")
print(" - umushroom_expanded_frontiers.png")
print(" - umushroom_expanded_weights.csv")
print(" - umushroom_candidate_allocations.csv")
print(" - umushroom_matched_frontiers.csv")
print(" - umushroom_diversification_ranking.csv")
print(" - umushroom_original_frontier.csv")
print(" - umushroom_expanded_frontier.csv")
print(" - umushroom_portfolio_comparison.csv")

# Reproducible research outputs; no trades or changes to recorded holdings.
returns.to_csv(OUTPUT_DIR / "gbp_daily_returns.csv", index_label="Date")
validation = None
if args.validate:
    validation = run_validation(returns, holdings, candidate_assets, MAX_POSITION, OUTPUT_DIR)
write_report(OUTPUT_DIR, returns, holdings, candidate_assets, comparison, matched,
             ranking, cov, sample_cov, shrinkage, validation, args.as_of, DATA_DIR, args.bond_series)
print(f"Report: {OUTPUT_DIR / 'BASELINE_REPORT.md'}")
