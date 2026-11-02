"""Official daily NAVs for the actual accumulating EUR share class."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ISIN = 'IE000VYDX630'
NAME = 'BlackRock Euro Government Enhanced Index Fund D EUR Acc'
URL = ('https://www.blackrock.com/uk/individual/products/326426/fund/'
       '1527484370694.ajax?fileType=xls&fileName='
       'BlackRock-Euro-Government-Enhanced-Index-Fund-D-Acc-EUR_fund&dataType=fund')


def parse_nav(payload):
    """BlackRock's XLS download is XML Spreadsheet, not a binary workbook."""
    root = ET.fromstring(payload.decode('utf-8-sig'))
    ns = {'s':'urn:schemas-microsoft-com:office:spreadsheet'}
    ns_uri = '{'+ns['s']+'}'
    all_text = [e.text for e in root.findall('.//s:Data',ns)]
    if ISIN not in all_text:
        raise ValueError('BlackRock download does not identify the exact share class')
    overview = next(w for w in root.findall('s:Worksheet',ns) if w.attrib[ns_uri+'Name']=='Overview')
    overview_rows = [[e.text for e in r.findall('s:Cell/s:Data',ns)] for r in overview.findall('s:Table/s:Row',ns)]
    if not any('Share Class Currency' in row and 'EUR' in row for row in overview_rows):
        raise ValueError('Unexpected fund currency')
    sheet = next(w for w in root.findall('s:Worksheet',ns) if w.attrib[ns_uri+'Name']=='Historical')
    rows = sheet.findall('s:Table/s:Row',ns)
    header = [e.text for e in rows[0].findall('s:Cell/s:Data',ns)]
    if header[:2] != ['As Of','NAV per Share']:
        raise ValueError('Unexpected historical NAV columns')
    observations=[]
    for row in rows[1:]:
        cells=[c.find('s:Data',ns).text if c.find('s:Data',ns) is not None else None for c in row.findall('s:Cell',ns)]
        date=pd.to_datetime(cells[0].replace('Sept','Sep'),format='%d/%b/%Y')
        observations.append((date,float(cells[1])))
    series=pd.Series([v for d,v in observations],index=pd.DatetimeIndex([d for d,v in observations],name='Date'),name=ISIN).sort_index()
    if series.empty or series.index.has_duplicates or not np.isfinite(series).all() or (series<=0).any():
        raise ValueError('Invalid official NAV history')
    return series


def load_fund_nav(start, end, directory, refresh=False):
    directory=Path(directory)
    path=directory/(ISIN+'_blackrock.xml')
    if refresh or not path.exists():
        import requests
        response=requests.get(URL,timeout=40)
        response.raise_for_status()
        series=parse_nav(response.content) # Validate before replacing existing cache.
        directory.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(dir=directory,suffix='.tmp')
        os.close(fd)
        try:
            Path(temp).write_bytes(response.content)
            os.replace(temp,path)
        finally:
            Path(temp).unlink(missing_ok=True)
        metadata={'isin':ISIN,'name':NAME,'currency':'EUR','source_url':URL,
                  'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
                  'sha256':hashlib.sha256(response.content).hexdigest(),
                  'observations':len(series),'first_nav':str(series.index.min().date()),
                  'last_nav':str(series.index.max().date())}
        path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        series.to_csv(directory/(ISIN+'.csv'),index_label='Date')
    else:
        series=parse_nav(path.read_bytes())
    window=series.loc[(series.index>=pd.Timestamp(start)) & (series.index<pd.Timestamp(end))]
    if window.empty:
        raise ValueError('No actual fund NAVs in requested dates; use --bond-series proxy explicitly')
    print(f'Official NAV {ISIN}: {len(window)} observations, {window.index.min().date()} to {window.index.max().date()}')
    return window
