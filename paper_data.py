"""Explicit refresh of observed prices, with separate raw-data vintages."""
from datetime import datetime
from pathlib import Path
import json
import tempfile
import os
import pandas as pd
from paper_test import load_protocol,atomic_text,LONDON,digest
from price_cache import load_close
from fund_nav import load_fund_nav,ISIN


def refresh_prices(directory,as_of):
    import yfinance as yf
    directory=Path(directory);p=load_protocol(directory)
    cutoff=pd.Timestamp(as_of).normalize()
    if cutoff>pd.Timestamp(datetime.now(LONDON).date()):raise ValueError('Cannot request future prices')
    start=str((pd.Timestamp(p['earliest_entry_date'])-pd.Timedelta(days=7)).date())
    end=str(cutoff.date());data_dir=directory/'data';data_dir.mkdir(exist_ok=True)
    timestamp=datetime.now(LONDON).strftime('%Y%m%dT%H%M%S%f%z')
    snapshots=data_dir/'snapshots';snapshots.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        staging=Path(temporary)/'snapshot';staging.mkdir()
        series={}
        for asset in p['active_assets']:
            if asset=='CASH_GBP':continue
            if asset==ISIN:
                series[asset]=load_fund_nav(start,end,staging/'nav',refresh=True)
            else:
                series[asset]=load_close(asset,start,end,staging/'prices',yf.download,refresh=True)
        fx={currency:load_close(currency+'GBP=X',start,end,staging/'prices',yf.download,refresh=True)
            for currency in sorted(set(p['quote_currencies'].values())-{'GBP'})}
        # No interpolation or carry-forward in the forward valuation ledger.
        prices=pd.concat({a:s if p['quote_currencies'][a]=='GBP' else s*fx[p['quote_currencies'][a]]
                          for a,s in series.items()},axis=1).sort_index()
        prices.index.name='Date';prices.to_csv(staging/'gbp_prices.csv')
        metadata={'as_of_exclusive':end,'retrieved_at_london':datetime.now(LONDON).isoformat(),
                  'source':'Yahoo adjusted closes and FX; actual BlackRock share-class NAVs',
                  'source_last_dates':{a:str(s.index.max().date()) for a,s in series.items()},
                  'gbp_price_file_sha256':digest(staging/'gbp_prices.csv'),
                  'raw_file_sha256':{str(path.relative_to(staging)):digest(path) for path in staging.rglob('*') if path.is_file()}}
        (staging/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        destination=snapshots/timestamp
        os.replace(staging,destination)
    atomic_text(data_dir/'latest_snapshot.json',json.dumps({'directory':str(destination.relative_to(data_dir)),
                'metadata_sha256':digest(destination/'metadata.json')},indent=2)+'\n')
    return prices,str(destination.relative_to(directory))


def import_prices(directory,path):
    directory=Path(directory);path=Path(path)
    prices=pd.read_csv(path,index_col='Date',parse_dates=True)
    folder=directory/'data'/'imports';folder.mkdir(parents=True,exist_ok=True)
    target=folder/(datetime.now(LONDON).strftime('%Y%m%dT%H%M%S%f%z')+'.csv')
    atomic_text(target,path.read_text())
    atomic_text(directory/'data'/'latest_snapshot.json',json.dumps({'file':str(target.relative_to(directory/'data')),
                'sha256':digest(target)},indent=2)+'\n')
    return prices,'import:'+str(target.relative_to(directory))


def cached_prices(directory):
    directory=Path(directory);data_dir=directory/'data';pointer=data_dir/'latest_snapshot.json'
    if not pointer.exists():return None,None
    link=json.loads(pointer.read_text())
    if 'file' in link:
        path=data_dir/link['file']
        if digest(path)!=link['sha256']:raise ValueError('Imported price snapshot changed')
        return pd.read_csv(path,index_col='Date',parse_dates=True),'import:'+str(path.relative_to(directory))
    snapshot=data_dir/link['directory']
    if digest(snapshot/'metadata.json')!=link['metadata_sha256']:raise ValueError('Cached snapshot metadata changed')
    metadata=json.loads((snapshot/'metadata.json').read_text())
    if digest(snapshot/'gbp_prices.csv')!=metadata['gbp_price_file_sha256']:raise ValueError('Cached prices changed')
    return pd.read_csv(snapshot/'gbp_prices.csv',index_col='Date',parse_dates=True),str(snapshot.relative_to(directory))
