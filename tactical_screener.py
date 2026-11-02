
"""
TACTICAL OPPORTUNITY SCREENER V2.1
Umushroom / Markowitz Portfolio Research

Features:
- Price drawdowns and volatility
- Fundamental screening
- Relative valuation
- Speculative-company classification
- Illustrative DCF scenarios
- Historical forward-EPS snapshots
- Optional GBP portfolio risk comparison
- CSV research exports

Run:
    python tactical_screener.py

With actual portfolio:
    python tactical_screener.py --holdings holdings.csv

Research only. No automatic trading.
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# 1. CONFIGURATION
# =========================================================

UNIVERSE = {
    "NVDA": "Semiconductors",
    "AMD": "Semiconductors",
    "AVGO": "Semiconductors",
    "MSFT": "Hyperscalers",
    "GOOGL": "Hyperscalers",
    "AMZN": "Hyperscalers",
    "META": "Hyperscalers",
    "CRWV": "AI infrastructure",
    "NBIS": "AI infrastructure",
    "PLTR": "Software",
    "NOW": "Software",
    "CRWD": "Cybersecurity",
    "VRT": "Power & cooling",
    "ETN": "Power & cooling",
    "IONQ": "Quantum",
    "RGTI": "Quantum",
    "ISRG": "Healthcare growth",
    "MELI": "Consumer growth",
}

# Illustrative DCF sensitivities.
# Tuple: annual FCF growth, equity discount rate,
# terminal growth.
#
# These are NOT company-specific forecasts.

DCF_SCENARIOS = {
    "bear": (0.00, 0.13, 0.02),
    "base": (0.10, 0.11, 0.03),
    "bull": (0.20, 0.10, 0.035),
}

DCF_YEARS = 5
MIN_PEERS = 3
MIN_SCORE_COVERAGE = 0.70
DEFAULT_SLEEVE = 0.05

SCORE_COMPONENTS = {
    "revenue_growth": (0.25, True),
    "operating_margin": (0.20, True),
    "fcf_yield": (0.20, True),
    "forward_pe": (0.15, False),
    "net_debt_revenue": (0.10, False),
    "drawdown_252": (0.10, False),
}


# =========================================================
# 2. GENERAL UTILITIES
# =========================================================

def num(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def valid(value):
    return np.isfinite(num(value))


def clean_date_index(series):
    series = series.copy()

    index = pd.to_datetime(series.index)

    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)

    series.index = index.normalize()

    return (
        series[
            ~series.index.duplicated(keep="last")
        ]
        .sort_index()
    )


def close_history(symbol, period="18mo"):

    hist = yf.Ticker(symbol).history(
        period=period,
        auto_adjust=True,
    )

    if hist.empty or "Close" not in hist:
        raise ValueError(
            f"No price history for {symbol}"
        )

    close = hist["Close"].dropna().astype(float)

    if close.empty:
        raise ValueError(
            f"Empty closing prices for {symbol}"
        )

    return clean_date_index(close)


def percentile_rank(
    series,
    higher_is_better=True,
):
    values = pd.to_numeric(
        series, errors="coerce"
    )

    if not higher_is_better:
        values = -values

    return values.rank(pct=True) * 100


# =========================================================
# 3. MARKET AND FUNDAMENTAL DATA
# =========================================================

def get_stock(symbol, segment, prices, currencies):

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}

    hist = ticker.history(
        period="18mo",
        auto_adjust=True,
    )

    if hist.empty or "Close" not in hist:
        raise ValueError("No historical prices")

    close = clean_date_index(
        hist["Close"].dropna().astype(float)
    )

    if len(close) < 65:
        raise ValueError(
            "Insufficient price observations"
        )

    prices[symbol] = close

    currency = info.get("currency")

    if not currency:
        raise ValueError(
            "Missing quote currency"
        )

    currencies[symbol] = currency

    price = float(close.iloc[-1])

    peak = float(close.tail(252).max())

    drawdown = price / peak - 1

    return_63 = (
        price / float(close.iloc[-64]) - 1
    )

    daily_returns = close.pct_change().dropna()

    volatility = (
        daily_returns.tail(63).std()
        * np.sqrt(252)
    )

    market_cap = num(info.get("marketCap"))
    fcf = num(info.get("freeCashflow"))

    revenue = num(info.get("totalRevenue"))
    debt = num(info.get("totalDebt"))
    cash = num(info.get("totalCash"))

    forward_eps = num(info.get("forwardEps"))
    forward_pe = num(info.get("forwardPE"))

    net_debt = (
        debt - cash
        if valid(debt) and valid(cash)
        else np.nan
    )

    fcf_yield = (
        fcf / market_cap
        if valid(fcf)
        and valid(market_cap)
        and market_cap > 0
        else np.nan
    )

    net_debt_revenue = (
        net_debt / revenue
        if valid(net_debt)
        and valid(revenue)
        and revenue > 0
        else np.nan
    )

    return {
        "ticker": symbol,
        "segment": segment,
        "name": info.get("shortName", symbol),
        "currency": currency,
        "market_date": str(close.index[-1].date()),
        "retrieved_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "price": price,
        "drawdown_252": drawdown,
        "return_63": return_63,
        "volatility_63": volatility,
        "market_cap": market_cap,
        "fcf": fcf,
        "fcf_yield": fcf_yield,
        "revenue_growth": num(
            info.get("revenueGrowth")
        ),
        "operating_margin": num(
            info.get("operatingMargins")
        ),
        "forward_eps": forward_eps,
        "forward_pe": forward_pe,
        "net_debt": net_debt,
        "net_debt_revenue": net_debt_revenue,
    }


# =========================================================
# 4. COMPANY CLASSIFICATION
# =========================================================

def classify_companies(df):
    """
    Positive FCF, forward EPS, and forward P/E
    are minimum screening requirements.

    This does not prove that earnings or FCF
    are sustainable.
    """

    df = df.copy()

    eligible = (
        df["fcf"].notna()
        & (df["fcf"] > 0)
        & df["forward_eps"].notna()
        & (df["forward_eps"] > 0)
        & df["forward_pe"].notna()
        & (df["forward_pe"] > 0)
        & df["market_cap"].notna()
        & (df["market_cap"] > 0)
    )

    df["valuation_eligible"] = eligible

    df["research_category"] = np.where(
        eligible,
        "CONVENTIONAL_RESEARCH",
        "SPECULATIVE_OR_INSUFFICIENT_DATA",
    )

    return df


# =========================================================
# 5. RELATIVE VALUATION
# =========================================================

def add_peer_valuation(df):
    """
    Compare P/E only inside configured segments.

    Require at least three positive-P/E peers.
    Exclude the target company itself when
    calculating its peer median.
    """

    df = df.copy()

    df["peer_median_pe"] = np.nan
    df["peer_count"] = 0

    positive = df["forward_pe"].where(
        df["forward_pe"] > 0
    )

    for index, row in df.iterrows():

        peers = positive.loc[
            (df["segment"] == row["segment"])
            & (df.index != index)
        ].dropna()

        df.at[index, "peer_count"] = len(peers)

        if len(peers) >= MIN_PEERS:
            df.at[
                index, "peer_median_pe"
            ] = peers.median()

    df["pe_discount_to_peers"] = (
        df["forward_pe"]
        / df["peer_median_pe"]
        - 1
    )

    df.loc[
        df["forward_pe"] <= 0,
        "pe_discount_to_peers",
    ] = np.nan

    return df


# =========================================================
# 6. ILLUSTRATIVE DCF
# =========================================================

def dcf_value(
    fcf,
    market_cap,
    price,
    growth,
    discount,
    terminal_growth,
):
    """
    Simplified equity FCF valuation.

    Assumes reported FCF approximates sustainable
    cash available to equity investors.

    Does not subtract net debt because the
    starting cash flow is treated as equity FCF.

    Important:
    - Not an enterprise FCFF/WACC model.
    - Does not model dilution or capital structure.
    - Growth rates are illustrative.
    """

    if not all(valid(x) for x in (
        fcf,
        market_cap,
        price,
    )):
        return np.nan

    if (
        fcf <= 0
        or market_cap <= 0
        or price <= 0
        or discount <= terminal_growth
    ):
        return np.nan

    pv = 0.0

    for year in range(1, DCF_YEARS + 1):

        future_fcf = (
            fcf * (1 + growth) ** year
        )

        pv += (
            future_fcf
            / (1 + discount) ** year
        )

    last_fcf = (
        fcf * (1 + growth) ** DCF_YEARS
    )

    terminal_value = (
        last_fcf
        * (1 + terminal_growth)
        / (discount - terminal_growth)
    )

    pv += (
        terminal_value
        / (1 + discount) ** DCF_YEARS
    )

    return price * pv / market_cap


def add_valuations(df):

    df = df.copy()

    for scenario, assumptions in (
        DCF_SCENARIOS.items()
    ):

        growth, discount, terminal_growth = (
            assumptions
        )

        fair_col = f"fair_value_{scenario}"
        upside_col = f"upside_{scenario}"

        df[fair_col] = np.nan

        eligible = df["valuation_eligible"]

        if eligible.any():

            df.loc[eligible, fair_col] = (
                df.loc[eligible].apply(
                    lambda row: dcf_value(
                        row["fcf"],
                        row["market_cap"],
                        row["price"],
                        growth,
                        discount,
                        terminal_growth,
                    ),
                    axis=1,
                )
            )

        df[upside_col] = (
            df[fair_col] / df["price"] - 1
        )

    return df


# =========================================================
# 7. HISTORICAL EPS SNAPSHOTS
# =========================================================

def save_eps_snapshots(df, output_dir):
    """
    Saves current Yahoo forward EPS snapshots.

    After at least 20 calendar days, compares
    the latest snapshot with the most recent
    snapshot dated at least 20 days earlier.

    This is NOT a verified historical consensus
    earnings revision series.
    """

    df = df.copy()

    path = output_dir / "eps_snapshots.csv"

    today = pd.Timestamp.now(
        tz="UTC"
    ).date()

    current = df[
        ["ticker", "forward_eps"]
    ].copy()

    current.insert(
        0, "date", str(today)
    )

    if path.exists():

        history = pd.read_csv(path)

        expected = {
            "date", "ticker", "forward_eps"
        }

        if not expected.issubset(
            history.columns
        ):
            raise ValueError(
                "Invalid existing EPS snapshot file"
            )

        history = pd.concat(
            [history, current],
            ignore_index=True,
        )

    else:
        history = current.copy()

    history["forward_eps"] = (
        pd.to_numeric(
            history["forward_eps"],
            errors="coerce",
        )
    )

    history = (
        history.drop_duplicates(
            ["date", "ticker"],
            keep="last",
        )
        .sort_values(["ticker", "date"])
    )

    history.to_csv(
        path, index=False
    )

    revisions = {}

    for symbol in df["ticker"]:

        rows = history.loc[
            history["ticker"] == symbol
        ].copy()

        rows["date"] = pd.to_datetime(
            rows["date"]
        )

        rows = rows.sort_values("date")

        cutoff = (
            pd.Timestamp(today)
            - pd.Timedelta(days=20)
        )

        old = rows.loc[
            (rows["date"] <= cutoff)
            & (rows["forward_eps"] > 0)
        ]

        latest = rows.loc[
            rows["forward_eps"] > 0
        ]

        if old.empty or latest.empty:
            revisions[symbol] = np.nan
            continue

        previous_eps = float(
            old.iloc[-1]["forward_eps"]
        )

        latest_eps = float(
            latest.iloc[-1]["forward_eps"]
        )

        revisions[symbol] = (
            latest_eps / previous_eps - 1
        )

    df["eps_revision_observed"] = (
        df["ticker"].map(revisions)
    )

    return df


# =========================================================
# 8. RISK FLAGS
# =========================================================

def risk_flags(row):

    flags = []

    if not valid(row["fcf"]):
        flags.append("FCF_MISSING")

    elif row["fcf"] <= 0:
        flags.append("NON_POSITIVE_FCF")

    if not valid(row["forward_eps"]):
        flags.append("EPS_MISSING")

    elif row["forward_eps"] <= 0:
        flags.append("NON_POSITIVE_EPS")

    if (
        valid(row["drawdown_252"])
        and row["drawdown_252"] < -0.50
    ):
        flags.append("DRAWDOWN_OVER_50_PERCENT")

    if not valid(row["peer_median_pe"]):
        flags.append("INSUFFICIENT_PEERS")

    if (
        valid(row["net_debt_revenue"])
        and row["net_debt_revenue"] > 1
    ):
        flags.append("HIGH_NET_DEBT_TO_REVENUE")

    if not row["valuation_eligible"]:
        flags.append("NOT_CONVENTIONALLY_ELIGIBLE")

    return ";".join(flags) if flags else "NONE"


# =========================================================
# 9. RESEARCH SCORING
# =========================================================

def add_scores(df):
    """
    Only conventionally eligible companies
    receive numerical research scores.

    DCF upside is deliberately excluded.

    Scores reflect relative characteristics
    of this research universe only.
    """

    df = df.copy()

    eligible = df["valuation_eligible"]

    numerator = pd.Series(
        0.0, index=df.index
    )

    denominator = pd.Series(
        0.0, index=df.index
    )

    for column, (
        weight,
        higher_is_better,
    ) in SCORE_COMPONENTS.items():

        values = df[column].copy()

        # Do not let speculative companies
        # influence the conventional ranking.
        values = values.where(eligible)

        if column == "forward_pe":
            values = values.where(values > 0)

        scores = percentile_rank(
            values,
            higher_is_better,
        )

        numerator += (
            scores.fillna(0) * weight
        )

        denominator += (
            scores.notna() * weight
        )

    df["score_coverage"] = denominator

    df["research_score"] = (
        numerator
        / denominator.replace(0, np.nan)
    )

    # CRITICAL FIX:
    # Apply eligibility exclusion AFTER scoring.
    df.loc[
        ~eligible
        | (
            df["score_coverage"]
            < MIN_SCORE_COVERAGE
        ),
        "research_score",
    ] = np.nan

    df["risk_flags"] = df.apply(
        risk_flags, axis=1
    )

    # Assert the classification is respected.
    speculative = df.loc[~eligible]

    assert speculative[
        "research_score"
    ].isna().all(), (
        "Speculative companies received scores"
    )

    return df


# =========================================================
# 10. GBP RETURNS
# =========================================================

def gbp_returns(
    symbol,
    prices,
    currencies,
    fx_cache,
):
    """
    Converts local prices into GBP.

    Yahoo FX pair CURRENCYGBP=X denotes
    GBP per unit of quote currency.

    GBP cash is modelled as a zero-return asset.
    """

    if symbol not in prices:
        prices[symbol] = close_history(symbol)

    close = prices[symbol]

    currency = currencies.get(symbol)

    if currency is None:
        info = yf.Ticker(symbol).info or {}
        currency = info.get("currency")
        currencies[symbol] = currency

    if currency == "GBP":
        gbp = close

    elif currency in ("GBp", "GBX"):
        gbp = close / 100

    elif currency:

        pair = f"{currency}GBP=X"

        if pair not in fx_cache:
            fx_cache[pair] = close_history(
                pair, period="18mo"
            )

        fx = fx_cache[pair]

        frame = pd.concat(
            [
                close.rename("stock"),
                fx.rename("fx"),
            ],
            axis=1,
        ).sort_index()

        frame["fx"] = frame["fx"].ffill()

        # Only calculate values where
        # an actual stock close exists.
        gbp = (
            frame["stock"] * frame["fx"]
        ).dropna()

    else:
        raise ValueError(
            f"Unknown currency for {symbol}"
        )

    return (
        gbp.pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


# =========================================================
# 11. PORTFOLIO IMPACT
# =========================================================

def compare_portfolio(
    df,
    holdings_file,
    prices,
    currencies,
    sleeve=DEFAULT_SLEEVE,
):
    """
    Descriptive historical comparison.

    Moves a fixed percentage of total portfolio
    weight from GBP cash into each candidate.

    Tests eligible and speculative candidates
    separately in the output.

    Constant-weight daily-rebalanced approximation.

    Limitations:
    - No dealing costs
    - No GBP cash yield
    - Historical volatility is not forecast risk
    - No actual historical portfolio trade path
    """

    holdings = pd.read_csv(
        holdings_file
    )

    required = {"ticker", "weight"}

    if not required.issubset(
        holdings.columns
    ):
        raise ValueError(
            "CSV needs ticker,weight columns"
        )

    holdings["ticker"] = (
        holdings["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    holdings["weight"] = (
        pd.to_numeric(
            holdings["weight"],
            errors="raise",
        )
    )

    holdings = (
        holdings.groupby(
            "ticker", as_index=False
        )["weight"]
        .sum()
    )

    if (
        not np.isfinite(
            holdings["weight"]
        ).all()
        or (holdings["weight"] < 0).any()
    ):
        raise ValueError(
            "Weights must be finite and nonnegative"
        )

    if not np.isclose(
        holdings["weight"].sum(),
        1.0,
        atol=0.001,
    ):
        raise ValueError(
            "Portfolio weights must sum to 1"
        )

    weights = dict(zip(
        holdings["ticker"],
        holdings["weight"],
    ))

    if weights.get("CASH", 0) < sleeve:
        raise ValueError(
            "Not enough GBP cash for allocation"
        )

    held = [
        symbol
        for symbol in weights
        if symbol != "CASH"
    ]

    symbols = set(held) | set(df["ticker"])

    fx_cache = {}
    returns = {}
    failures = {}

    for symbol in sorted(symbols):

        try:
            returns[symbol] = gbp_returns(
                symbol,
                prices,
                currencies,
                fx_cache,
            )

        except Exception as exc:
            failures[symbol] = str(exc)

    missing_holdings = [
        symbol for symbol in held
        if symbol not in returns
    ]

    if missing_holdings:
        raise ValueError(
            "Missing current portfolio data: "
            + ", ".join(missing_holdings)
        )

    records = []

    for row in df.itertuples():

        candidate = row.ticker

        if candidate not in returns:

            records.append({
                "ticker": candidate,
                "status": failures.get(
                    candidate, "Missing prices"
                ),
                "research_category": (
                    row.research_category
                ),
            })

            continue

        assets = sorted(
            set(held + [candidate])
        )

        frame = pd.concat(
            {
                symbol: returns[symbol]
                for symbol in assets
            },
            axis=1,
        ).dropna().tail(252)

        if len(frame) < 126:

            records.append({
                "ticker": candidate,
                "status": (
                    "Insufficient overlapping history"
                ),
                "research_category": (
                    row.research_category
                ),
            })

            continue

        old_weights = pd.Series({
            symbol: weights.get(symbol, 0.0)
            for symbol in assets
        })

        new_weights = old_weights.copy()

        new_weights[candidate] += sleeve

        baseline = frame.dot(old_weights)
        proposed = frame.dot(new_weights)

        baseline_vol = (
            baseline.std() * np.sqrt(252)
        )

        proposed_vol = (
            proposed.std() * np.sqrt(252)
        )

        records.append({
            "ticker": candidate,
            "status": "evaluated",
            "research_category": (
                row.research_category
            ),
            "observations": len(frame),
            "baseline_volatility": baseline_vol,
            "proposed_volatility": proposed_vol,
            "volatility_change_pp": (
                proposed_vol - baseline_vol
            ) * 100,
            "baseline_return_annual": (
                baseline.mean() * 252
            ),
            "proposed_return_annual": (
                proposed.mean() * 252
            ),
        })

    return pd.DataFrame(records)


# =========================================================
# 12. MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(UNIVERSE),
    )

    parser.add_argument(
        "--holdings",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="reports/tactical-v2",
    )

    parser.add_argument(
        "--sleeve",
        type=float,
        default=DEFAULT_SLEEVE,
    )

    args = parser.parse_args()

    if not 0 < args.sleeve <= 1:
        parser.error(
            "--sleeve must be between 0 and 1"
        )

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices = {}
    currencies = {}
    rows = []
    errors = []

    # -------------------------------
    # Retrieve companies
    # -------------------------------

    for symbol in dict.fromkeys(args.tickers):

        symbol = symbol.upper()

        try:

            result = get_stock(
                symbol,
                UNIVERSE.get(
                    symbol, "Other"
                ),
                prices,
                currencies,
            )

            rows.append(result)

            print(f"OK  {symbol}")

        except Exception as exc:

            errors.append({
                "ticker": symbol,
                "error": str(exc),
            })

            print(
                f"ERR {symbol}: {exc}"
            )

    if not rows:
        raise RuntimeError(
            "No stocks retrieved"
        )

    df = pd.DataFrame(rows)

    # -------------------------------
    # Correct processing order
    # -------------------------------

    df = classify_companies(df)

    df = add_peer_valuation(df)

    df = add_valuations(df)

    df = save_eps_snapshots(
        df, output
    )

    df = add_scores(df)

    df = df.sort_values(
        "research_score",
        ascending=False,
        na_position="last",
    )

    # -------------------------------
    # Export opportunities
    # -------------------------------

    df.to_csv(
        output / "opportunities.csv",
        index=False,
    )

    # -------------------------------
    # Export DCF sensitivities
    # -------------------------------

    scenario_columns = [
        "ticker",
        "research_category",
        "price",
        "fair_value_bear",
        "upside_bear",
        "fair_value_base",
        "upside_base",
        "fair_value_bull",
        "upside_bull",
    ]

    df[scenario_columns].to_csv(
        output / "dcf_scenarios.csv",
        index=False,
    )

    # -------------------------------
    # Export speculative cases
    # -------------------------------

    speculative = df.loc[
        ~df["valuation_eligible"]
    ].copy()

    speculative.to_csv(
        output / "speculative_research.csv",
        index=False,
    )

    # -------------------------------
    # Optional portfolio comparison
    # -------------------------------

    if args.holdings:

        try:

            impact = compare_portfolio(
                df,
                args.holdings,
                prices,
                currencies,
                sleeve=args.sleeve,
            )

            impact.to_csv(
                output / "portfolio_impact.csv",
                index=False,
            )

        except Exception as exc:

            errors.append({
                "ticker": "PORTFOLIO",
                "error": str(exc),
            })

            print(
                "Portfolio comparison unavailable:",
                exc,
            )

    # -------------------------------
    # Export errors
    # -------------------------------

    pd.DataFrame(
        errors,
        columns=["ticker", "error"],
    ).to_csv(
        output / "errors.csv",
        index=False,
    )

    # -------------------------------
    # Terminal results
    # -------------------------------

    print("\nCONVENTIONAL RESEARCH RANKING\n")

    display = [
        "ticker",
        "drawdown_252",
        "forward_pe",
        "pe_discount_to_peers",
        "upside_base",
        "eps_revision_observed",
        "research_score",
        "research_category",
        "risk_flags",
    ]

    conventional = df.loc[
        df["valuation_eligible"]
    ]

    print(
        conventional[display].to_string(
            index=False
        )
    )

    print("\nSPECULATIVE / INCOMPLETE DATA\n")

    if speculative.empty:
        print("None")
    else:
        print(
            speculative[
                [
                    "ticker",
                    "drawdown_252",
                    "fcf_yield",
                    "forward_eps",
                    "research_category",
                    "risk_flags",
                ]
            ].to_string(index=False)
        )

    print(
        "\nExcluded from conventional scoring:",
        speculative["ticker"].tolist(),
    )

    print(
        "\nSaved to:",
        output.resolve(),
    )

    print(
        "\nCAUTION: Rankings are exploratory. "
        "DCF scenarios are illustrative, not "
        "validated investment price targets."
    )


if __name__ == "__main__":
    main()
