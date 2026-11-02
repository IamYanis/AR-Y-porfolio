"""Versioned, local cache of Yahoo adjusted closing prices in quote currency."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd


def load_close(ticker, start, end, cache_dir, downloader, refresh=False):
    # Exact windows avoid treating a partial history as a complete cache hit.
    request = {"ticker": ticker, "start": start, "end": end,
               "auto_adjust": True, "schema": 1}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    safe_ticker = ''.join(c if c.isalnum() or c in '.-_' else '_' for c in ticker)
    path = Path(cache_dir) / f"{safe_ticker}_{key}.csv"
    metadata_path = path.with_suffix('.json')

    def validate(series):
        series = series.astype(float).sort_index()
        if (series.empty or series.index.has_duplicates
                or series.index.isna().any()
                or not np.isfinite(series.to_numpy()).all()
                or (series <= 0).any()):
            raise ValueError(f"Invalid cached/downloaded prices for {ticker}")
        if (series.index < pd.Timestamp(start)).any() or (series.index >= pd.Timestamp(end)).any():
            raise ValueError(f"Prices outside requested dates for {ticker}")
        series.index.name = "Date"
        series.name = ticker
        return series

    if path.exists() and not refresh:
        try:
            frame = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
            result = validate(frame['Close'])
            print(f"Cached {ticker}: {len(result)} observations")
            return result
        except Exception as exc:
            raise ValueError(f"Cannot read {path}; use --refresh-data to replace it") from exc

    print(f"Downloading {ticker}...")
    data = downloader(ticker, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}; existing cache preserved")
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = validate(close.dropna())
    metadata = dict(request, source='Yahoo Finance via yfinance',
                    downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                    first_observation=close.index.min().isoformat(),
                    last_observation=close.index.max().isoformat(),
                    observations=len(close), currency='original quote currency')
    path.parent.mkdir(parents=True, exist_ok=True)

    def atomic_write(target, write):
        fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix='.tmp')
        os.close(fd)
        try:
            write(Path(temp_name))
            os.replace(temp_name, target)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    atomic_write(path, lambda temp: close.rename('Close').to_csv(temp, index_label='Date'))
    atomic_write(metadata_path, lambda temp: temp.write_text(json.dumps(metadata, indent=2)+'\n'))
    return close
