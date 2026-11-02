
from pathlib import Path

import pandas as pd
import yfinance as yf


SOURCE = Path(
    "reports/latest/gbp_daily_returns.csv"
)

OUTPUT = Path(
    "data/tactical_gbp_returns.csv"
)

TACTICAL = ["NVDA", "AVGO", "VRT", "META", "PLTR"]


def get_prices(symbol):
    data = yf.Ticker(symbol).history(
        period="3y",
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError(f"No prices: {symbol}")

    close = data["Close"].dropna()

    index = pd.to_datetime(close.index)

    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)

    close.index = index.normalize()

    return close[~close.index.duplicated()]


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    existing = pd.read_csv(SOURCE)

    date_column = (
        "Date" if "Date" in existing.columns
        else "date" if "date" in existing.columns
        else None
    )

    if date_column is None:
        raise ValueError("Missing date column")

    existing[date_column] = pd.to_datetime(
        existing[date_column]
    )

    existing = existing.set_index(
        date_column
    ).sort_index()

    # Yahoo USDGBP=X is GBP per USD.
    usdgbp = get_prices("USDGBP=X")

    for symbol in TACTICAL:

        if symbol in existing.columns:
            print(f"Already present: {symbol}")
            continue

        usd_close = get_prices(symbol)

        aligned = pd.concat(
            [
                usd_close.rename("stock"),
                usdgbp.rename("fx"),
            ],
            axis=1,
        ).sort_index()

        aligned["fx"] = aligned["fx"].ffill()

        gbp_price = (
            aligned["stock"] * aligned["fx"]
        ).dropna()

        gbp_returns = gbp_price.pct_change()

        existing[symbol] = gbp_returns.reindex(
            existing.index
        )

        print(
            f"Added {symbol}: "
            f"{existing[symbol].notna().sum()} observations"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    existing.index.name = "Date"

    existing.to_csv(OUTPUT)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
