
"""
MARKOWITZ PORTFOLIO REBALANCING
================================

Stage 1:
    Optimise existing 17 holdings.

Stage 2:
    Include fixed tactical allocations:
    NVDA 3%, AVGO 3%, VRT 2.5%,
    META 2%, PLTR 2%.

Cash remains fixed at 35.08%.

Input:
    CSV containing daily GBP decimal returns.

Output:
    reports/tactical-optimisation/
        allocation_comparison.csv
        portfolio_comparison.csv
        proposed_trades.csv

Install:
    pip install numpy pandas scipy scikit-learn

Run:
    python optimise_tactical_portfolio.py \
        --returns data/daily_returns_gbp.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.covariance import OAS


# =====================================================
# 1. ACTUAL UMUSHROOM PORTFOLIO
# =====================================================

PORTFOLIO_VALUE = 100154.10
CASH_WEIGHT = 0.3508

# Weights copied from the Umushroom table.
# They sum to 64.88%, due to rounding.
# The code normalises them to 64.92%.

EXISTING = {
    "RTX": 0.0425,
    "AZN.L": 0.0420,
    "MU": 0.0386,
    "LLY": 0.0322,
    "BE": 0.0303,
    "VST": 0.0288,
    "IONQ": 0.0275,
    "TTMI": 0.0264,
    "NEE": 0.0262,
    "SYK": 0.0205,
    "CEG": 0.0195,
    "AMZN": 0.0191,
    "GOOGL": 0.0171,

    # Exact fund identity, not a proxy ticker.
    "IE000VYDX630": 0.0699,

    # Confirmed research alias for the Treasury ETF.
    "CBGA.L": 0.0704,

    # Local identifiers. Map to verified return series.
    "XDWS.L": 0.0690,  # Xtrackers MSCI World Consumer Staples
    "XDWC.L": 0.0688,  # Xtrackers MSCI World Consumer Discretionary
}

TACTICAL = {
    "NVDA": 0.0300,
    "AVGO": 0.0300,
    "VRT": 0.0250,
    "META": 0.0200,
    "PLTR": 0.0200,
}

MAX_EXISTING_WEIGHT = 0.10

# Penalty strength for unnecessary trading.
# Set to zero for pure Markowitz minimum variance.
TURNOVER_PENALTY = 0.0002

LOOKBACK = 252
MIN_OBSERVATIONS = 126


# =====================================================
# 2. WEIGHT PREPARATION
# =====================================================

def prepare_weights():

    existing = pd.Series(
        EXISTING,
        dtype=float,
    )

    tactical = pd.Series(
        TACTICAL,
        dtype=float,
    )

    if set(existing.index) & set(tactical.index):
        raise ValueError(
            "A ticker appears in both groups"
        )

    if any(existing < 0) or any(tactical < 0):
        raise ValueError("Negative weights")

    tactical_total = tactical.sum()

    if CASH_WEIGHT + tactical_total >= 1:
        raise ValueError(
            "Cash and tactical allocations exceed budget"
        )

    invested_budget = 1 - CASH_WEIGHT

    existing_current = (
        existing
        / existing.sum()
        * invested_budget
    )

    all_assets = (
        list(existing.index)
        + list(tactical.index)
    )

    current = pd.Series(
        0.0,
        index=all_assets,
    )

    current.loc[existing.index] = (
        existing_current
    )

    prorata = current.copy()

    remaining = (
        invested_budget - tactical_total
    )

    prorata.loc[existing.index] = (
        existing_current
        / invested_budget
        * remaining
    )

    prorata.loc[tactical.index] = tactical

    assert np.isclose(
        current.sum() + CASH_WEIGHT, 1
    )

    assert np.isclose(
        prorata.sum() + CASH_WEIGHT, 1
    )

    return (
        existing,
        tactical,
        current,
        prorata,
        all_assets,
    )


# =====================================================
# 3. LOAD GBP RETURNS
# =====================================================

def load_returns(path, assets):

    data = pd.read_csv(path)

    if "Date" not in data.columns:
        raise ValueError(
            "Returns CSV must contain Date"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    )

    data = (
        data.set_index("Date")
        .sort_index()
    )

    if data.index.has_duplicates:
        raise ValueError("Duplicate dates")

    missing = [
        asset
        for asset in assets
        if asset not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing GBP return columns: "
            + ", ".join(missing)
        )

    data = data[assets].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Shared dates ensure every portfolio
    # is evaluated on the same observations.
    data = data.dropna().tail(LOOKBACK)

    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(data)} overlapping observations; "
            f"need at least {MIN_OBSERVATIONS}"
        )

    # Guard against input in percentage points.
    if (data.abs() > 1).any().any():
        raise ValueError(
            "Check returns format: use 0.01 for 1%, "
            "not 1.0"
        )

    if (data <= -1).any().any():
        raise ValueError(
            "Invalid daily return <= -100%"
        )

    return data


# =====================================================
# 4. COVARIANCE ESTIMATION
# =====================================================

def covariance_matrix(returns):

    # OAS: shrink noisy covariance estimates.
    estimator = OAS(
        assume_centered=False
    )

    estimator.fit(returns.values)

    daily_cov = estimator.covariance_

    annual_cov = daily_cov * 252

    return pd.DataFrame(
        annual_cov,
        index=returns.columns,
        columns=returns.columns,
    )


# =====================================================
# 5. PORTFOLIO METRICS
# =====================================================

def annual_volatility(weights, covariance):

    w = weights.reindex(
        covariance.index
    ).fillna(0).values

    variance = (
        w @ covariance.values @ w
    )

    return np.sqrt(
        max(variance, 0)
    )


def historical_return(weights, returns):

    w = weights.reindex(
        returns.columns
    ).fillna(0)

    # Arithmetic annualisation, for comparison.
    # Not a forward return prediction.
    return (
        returns.mean().dot(w) * 252
    )


def turnover(old, new):

    difference = (
        new.reindex(old.index).fillna(0)
        - old
    )

    # Sum of absolute traded notional weights.
    return float(
        difference.abs().sum()
    )


# =====================================================
# 6. OPTIMISER
# =====================================================

def optimise(
    covariance,
    current,
    existing_names,
    tactical_fixed=None,
    max_weight=MAX_EXISTING_WEIGHT,
    turnover_penalty=TURNOVER_PENALTY,
):
    """
    Optimise the existing asset weights.

    Cash is outside the covariance matrix and
    fixed at CASH_WEIGHT.

    The tactical assets can either be excluded
    (Stage 1) or fixed to the specified weights
    (Stage 2).

    Objective:
        variance + optional smooth turnover penalty

    Long-only, fully invested, bounded allocation.
    """

    tactical_fixed = (
        tactical_fixed
        if tactical_fixed is not None
        else pd.Series(dtype=float)
    )

    fixed_total = float(
        tactical_fixed.sum()
    )

    budget = (
        1 - CASH_WEIGHT - fixed_total
    )

    if budget < -1e-10:
        raise ValueError("Negative budget")

    if budget > len(existing_names) * max_weight:
        raise ValueError(
            "Position caps make the budget infeasible"
        )

    # Starting point: scale existing weights.
    start = current.loc[
        existing_names
    ].copy()

    start = (
        start / start.sum() * budget
    )

    # If the starting point exceeds the position
    # cap, create a feasible initial allocation.
    if (start > max_weight).any():
        start[:] = budget / len(start)

    fixed_vector = pd.Series(
        0.0,
        index=covariance.index,
    )

    for ticker, weight in tactical_fixed.items():
        fixed_vector[ticker] = weight

    existing_current = (
        current.loc[existing_names].values
    )

    cov = covariance.values

    names = list(covariance.index)

    indices = [
        names.index(ticker)
        for ticker in existing_names
    ]

    def build_weights(x):

        full = fixed_vector.values.copy()

        full[indices] = x

        return full

    def objective(x):

        full = build_weights(x)

        variance = full @ cov @ full

        delta = (
            x - existing_current
        )

        # Smooth absolute turnover approximation.
        trade_penalty = np.sum(
            np.sqrt(delta ** 2 + 1e-10)
        )

        return (
            variance
            + turnover_penalty * trade_penalty
        )

    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x) - budget,
    }]

    bounds = [
        (0.0, max_weight)
        for _ in existing_names
    ]

    result = minimize(
        objective,
        x0=start.values,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Optimisation failed: "
            + result.message
        )

    optimal = pd.Series(
        build_weights(result.x),
        index=covariance.index,
    )

    if not np.isclose(
        optimal.sum() + CASH_WEIGHT,
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Optimised weights do not sum to 100%"
        )

    return optimal


# =====================================================
# 7. COMPARISON
# =====================================================

def compare_portfolios(
    portfolios,
    covariance,
    returns,
    current,
    value,
    cost_bps,
):

    rows = []

    cost_rate = cost_bps / 10000

    for name, weights in portfolios.items():

        trade_fraction = turnover(
            current, weights
        )

        rows.append({
            "portfolio": name,

            "annual_volatility_pct": (
                annual_volatility(
                    weights, covariance
                ) * 100
            ),

            "historical_return_pct": (
                historical_return(
                    weights, returns
                ) * 100
            ),

            "turnover_pct": (
                trade_fraction * 100
            ),

            "estimated_cost_gbp": (
                trade_fraction
                * value
                * cost_rate
            ),

            "cash_weight_pct": (
                CASH_WEIGHT * 100
            ),
        })

    return pd.DataFrame(rows)


# =====================================================
# 8. MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--returns",
        required=True,
        help="CSV of GBP daily decimal returns",
    )

    parser.add_argument(
        "--output",
        default="reports/tactical-optimisation",
    )

    parser.add_argument(
        "--value",
        type=float,
        default=PORTFOLIO_VALUE,
    )

    parser.add_argument(
        "--cost-bps",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--turnover-penalty",
        type=float,
        default=TURNOVER_PENALTY,
    )

    args = parser.parse_args()

    if args.value <= 0 or args.cost_bps < 0:
        parser.error(
            "Portfolio value must be positive "
            "and cost-bps nonnegative"
        )

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        existing,
        tactical,
        current,
        prorata,
        assets,
    ) = prepare_weights()

    returns = load_returns(
        args.returns,
        assets,
    )

    covariance = covariance_matrix(
        returns
    )

    # Stage 1: existing holdings only
    stage1 = optimise(
        covariance,
        current,
        list(existing.index),
        turnover_penalty=args.turnover_penalty,
    )

    # Stage 2: tactical holdings fixed
    stage2 = optimise(
        covariance,
        current,
        list(existing.index),
        tactical_fixed=tactical,
        turnover_penalty=args.turnover_penalty,
    )

    portfolios = {
        "Current": current,
        "Stage1_existing_only": stage1,
        "Tactical_pro_rata": prorata,
        "Stage2_tactical_optimised": stage2,
    }

    comparison = compare_portfolios(
        portfolios,
        covariance,
        returns,
        current,
        args.value,
        args.cost_bps,
    )

    # Complete allocation comparison
    allocation = pd.DataFrame({
        "asset": assets + ["CASH"],
        "current_pct": (
            list(current.values * 100)
            + [CASH_WEIGHT * 100]
        ),
        "stage1_pct": (
            list(stage1.values * 100)
            + [CASH_WEIGHT * 100]
        ),
        "prorata_pct": (
            list(prorata.values * 100)
            + [CASH_WEIGHT * 100]
        ),
        "stage2_pct": (
            list(stage2.values * 100)
            + [CASH_WEIGHT * 100]
        ),
    })

    # Exact trades relative to current allocation
    trades = pd.DataFrame({
        "asset": assets,
        "current_weight": current.values,
        "target_weight": stage2.values,
    })

    trades["change_weight"] = (
        trades["target_weight"]
        - trades["current_weight"]
    )

    trades["trade_gbp"] = (
        trades["change_weight"] * args.value
    )

    trades["action"] = np.where(
        trades["trade_gbp"] > 0.01,
        "BUY",
        np.where(
            trades["trade_gbp"] < -0.01,
            "SELL",
            "HOLD",
        ),
    )

    trades = trades.sort_values(
        "trade_gbp",
        ascending=False,
    )

    # Exports
    allocation.to_csv(
        output / "allocation_comparison.csv",
        index=False,
    )

    comparison.to_csv(
        output / "portfolio_comparison.csv",
        index=False,
    )

    trades.to_csv(
        output / "proposed_trades.csv",
        index=False,
    )

    covariance.to_csv(
        output / "covariance_annual.csv"
    )

    # Display
    print("\nPORTFOLIO COMPARISON\n")

    print(
        comparison.to_string(index=False)
    )

    print("\nOPTIMISED TARGET WEIGHTS\n")

    print(
        allocation[
            [
                "asset",
                "current_pct",
                "stage1_pct",
                "stage2_pct",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nPROPOSED TRADES\n")

    print(
        trades[
            [
                "asset",
                "action",
                "trade_gbp",
                "target_weight",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nInput observations:",
        len(returns),
    )

    print(
        "Evaluation dates:",
        returns.index.min().date(),
        "to",
        returns.index.max().date(),
    )

    print(
        "\nResults saved to:",
        output.resolve(),
    )

    print(
        "\nIMPORTANT: Historical covariance and "
        "return statistics are not forecasts. "
        "The output is a research allocation, "
        "not an execution instruction."
    )


if __name__ == "__main__":
    main()
