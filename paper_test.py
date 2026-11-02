"""Registered forward paper portfolios. No broker integration or real trades."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
import pandas as pd
from portfolio_validation import fit_covariances
from robustness import proposed_weights

LONDON=ZoneInfo('Europe/London')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        Path(tmp).write_text(text);os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)


def load_protocol(directory):
    directory=Path(directory);path=directory/'protocol.json'
    if digest(path)!=(directory/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol changed; create a new version instead of editing this trial')
    protocol=json.loads(path.read_text())
    if digest(directory/'training_returns.csv')!=protocol['training_sha256']:
        raise ValueError('Frozen training data changed')
    return protocol


def register(directory,baseline,now=None,allocation=.05,capital=100000.):
    directory=Path(directory);baseline=Path(baseline)
    now=now or datetime.now(LONDON)
    if directory.exists(): raise FileExistsError('Trial directory already exists; use update or a new directory')
    if now.tzinfo is None: raise ValueError('Registration timestamp must include timezone')
    today=pd.Timestamp(now.astimezone(LONDON).date())
    manifest=json.loads((baseline/'run_manifest.json').read_text())
    if manifest['bond_series']!='actual':raise ValueError('Forward trial requires the actual fund NAV baseline')
    r=pd.read_csv(baseline/'gbp_daily_returns.csv',index_col='Date',parse_dates=True)
    if not r.index.is_monotonic_increasing or r.index.has_duplicates or not np.isfinite(r.to_numpy()).all():
        raise ValueError('Training returns must have unique ordered dates and finite values')
    r=r.loc[r.index<today].tail(252)
    if len(r)<252:raise ValueError('Need 252 completed historical return observations')
    if (today-r.index.max()).days>7:raise ValueError('Baseline is stale; rebuild it from current completed observations before registering')
    if capital<=0 or not .0<allocation<=.1:raise ValueError('Use positive capital and a sleeve no larger than 10%')
    assets=list(r);holdings=manifest['holdings_percent']
    if set(assets)!=set(holdings) or 'IE000VYDX630' not in assets or holdings['IE000VYDX630']<=0:
        raise ValueError('Baseline must include all recorded holdings and the actual BlackRock fund')
    current=np.array([holdings[a]/100 for a in assets])
    candidates=manifest['candidate_assets']
    if not np.isclose(current.sum(),1) or (current<0).any():raise ValueError('Invalid configured weights')
    currencies=pd.read_csv(baseline/'data_quality.csv').set_index('Asset').Quote_Currency.to_dict()
    if any(currencies.get(a) not in ['USD','EUR','GBP'] for a in assets if a!='CASH_GBP'):
        raise ValueError('Every investment needs an explicit quote currency')
    covariance,_=fit_covariances(r)
    cov=covariance['OAS'];corr=r.drop(columns='CASH_GBP').corr()
    scores=[]
    for a in candidates:
        w=proposed_weights(current,assets,(a,),allocation)
        scores.append({'Asset':a,'Predicted_OAS_Volatility':float(np.sqrt(w@cov@w))})
    scores=sorted(scores,key=lambda x:(x['Predicted_OAS_Volatility'],x['Asset']))
    single=scores[0]['Asset'];pair_scores=[]
    for a in candidates:
        if a!=single and abs(corr.loc[single,a])<=.75:
            w=proposed_weights(current,assets,(single,a),allocation)
            pair_scores.append({'First':single,'Second':a,'Training_Correlation':float(corr.loc[single,a]),
                                'Predicted_OAS_Volatility':float(np.sqrt(w@cov@w))})
    if not pair_scores:raise ValueError('No eligible complementary pair under the frozen correlation limit')
    pair_scores.sort(key=lambda x:(x['Predicted_OAS_Volatility'],x['Second']))
    second=pair_scores[0]['Second'];control='CBGA.L'
    if control not in assets or current[assets.index(control)]<=0:raise ValueError('Existing CBGA control is required')
    weights={'current':current,'existing_bond':proposed_weights(current,assets,(control,),allocation,allow_held=True),
             'single':proposed_weights(current,assets,(single,),allocation),
             'pair':proposed_weights(current,assets,(single,second),allocation)}
    names={'current':'Current holdings','existing_bond':f'Add {allocation:.0%} to existing {control}',
           'single':f'Add {allocation:.0%} {single}','pair':f'Add {allocation/2:.1%} each: {single} + {second}'}
    for w in weights.values():
        if np.max(w[[i for i,a in enumerate(assets) if a!='CASH_GBP']])>.20+1e-10:
            raise ValueError('Paper weights exceed 20% investment cap')
    directory.mkdir(parents=True)
    r.to_csv(directory/'training_returns.csv',index_label='Date')
    predictions={key:{model:float(np.sqrt(max(w@c@w,0))) for model,c in covariance.items()} for key,w in weights.items()}
    start=today+pd.offsets.BDay(1)
    protocol={'schema':1,'registered_at_london':now.astimezone(LONDON).isoformat(),
        'decision_date':str(today.date()),'earliest_entry_date':str(start.date()),
        'entry_rule':'First common observed closing valuation on or after earliest_entry_date',
        'training_start':str(r.index.min().date()),'training_end':str(r.index.max().date()),
        'training_sha256':digest(directory/'training_returns.csv'),
        'baseline_manifest_sha256':digest(baseline/'run_manifest.json'),
        'paper_capital_gbp':capital,'assets':assets,'quote_currencies':currencies,
        'active_assets':[a for i,a in enumerate(assets) if a!='CASH_GBP' and any(w[i]>1e-12 for w in weights.values())],
        'names':names,'weights':{key:dict(zip(assets,map(float,w))) for key,w in weights.items()},
        'selected_single':single,'selected_pair':[single,second],'sleeve_allocation':allocation,
        'selection_rule':'Lowest OAS estimated risk; pair includes single plus best second asset with absolute training correlation <= 0.75',
        'predicted_volatility':predictions,'primary_model':'OAS','comparison_model':'Sample',
        'covariance_policy':'252-observation covariance frozen at registration; no retrospective refits or asset reselection',
        'rebalance_months':3,'review_months':1,'cash_yield':.03,'cash_day_count':365.25,
        'cash_policy':'Identical initial target cash weight; drift between quarterly target rebalances',
        'cost_bps_per_side':[0,10,25],'primary_cost_bps':25,'minimum_risk_observations':20,
        'data_policy':'Only observed joint closing prices; first-recorded interval returns are append-only',
        'source_mode':'Actual BlackRock share-class NAVs; no bond proxy permitted'}
    atomic_text(directory/'protocol.json',json.dumps(protocol,indent=2)+'\n')
    atomic_text(directory/'protocol.sha256',digest(directory/'protocol.json')+'\n')
    pd.DataFrame(scores).to_csv(directory/'single_selection.csv',index=False)
    pd.DataFrame(pair_scores).to_csv(directory/'pair_selection.csv',index=False)
    pd.DataFrame(weights,index=assets).rename_axis('Asset').to_csv(directory/'frozen_weights.csv')
    return protocol


def read_observations(directory):
    directory=Path(directory);path=directory/'observations.csv'
    if not path.exists():return pd.DataFrame()
    expected=json.loads((directory/'observations_manifest.json').read_text())['sha256']
    if digest(path)!=expected:raise ValueError('Observation ledger changed outside the update workflow')
    frame=pd.read_csv(path,parse_dates=['Date','From_Date'])
    if frame.Date.duplicated().any() or not frame.Date.is_monotonic_increasing:
        raise ValueError('Observation ledger dates are invalid')
    return frame


def record_prices(directory,prices,as_of,source='import',now=None):
    """Freeze interval total returns using one consistent price-data vintage."""
    directory=Path(directory);p=load_protocol(directory)
    now=now or datetime.now(LONDON)
    cutoff=pd.Timestamp(as_of).normalize()
    if cutoff>pd.Timestamp(now.astimezone(LONDON).date()):raise ValueError('Future as-of dates are not permitted')
    if cutoff<pd.Timestamp(p['decision_date']):raise ValueError('Cutoff predates trial registration')
    active=p['active_assets'];prices=prices.copy()
    if not isinstance(prices.index,pd.DatetimeIndex):raise ValueError('Prices require a date index')
    if prices.index.tz is not None:prices.index=prices.index.tz_localize(None)
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:raise ValueError('Prices need unique ascending dates')
    if any(a not in prices for a in active):raise ValueError('Missing prices for one or more active investments')
    prices=prices[active].astype(float).loc[prices.index<cutoff]
    # Strict common valuations; no forward-filling NAVs or asset prices.
    prices=prices.dropna()
    if not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any():raise ValueError('Invalid GBP total-return prices')
    prices=prices.loc[prices.index>=pd.Timestamp(p['earliest_entry_date'])]
    ledger=read_observations(directory)
    if not ledger.empty and cutoff<=ledger.Date.max():raise ValueError('Cutoff would hide already recorded observations')
    timestamp=now.isoformat();snapshot_hash=hashlib.sha256(prices.to_csv().encode()).hexdigest()
    revisions=[]
    if not ledger.empty:
        for row in ledger.itertuples(index=False):
            if row.Date in prices.index and row.From_Date in prices.index and row.Date!=row.From_Date:
                candidate=prices.loc[row.Date]/prices.loc[row.From_Date]-1
                stored=ledger.loc[ledger.Date==row.Date,active].iloc[0]
                for a in active:
                    if abs(candidate[a]-stored[a])>1e-8:
                        revisions.append({'Date':str(row.Date.date()),'Asset':a,'Recorded_Return':stored[a],
                                          'Latest_Source_Return':candidate[a],'Checked_At':timestamp})
    rows=[]
    if ledger.empty:
        if prices.empty:return ledger,{'appended':0,'revisions':0}
        entry=prices.index[0]
        rows.append(dict(Date=entry,From_Date=entry,Recorded_At=timestamp,Source=source,Snapshot_SHA256=snapshot_hash,
                         **{a:0. for a in active},CASH_GBP=0.))
        previous=entry
    else:previous=pd.Timestamp(ledger.Date.iloc[-1])
    newer=prices.index[prices.index>previous]
    if len(newer) and previous not in prices.index:
        raise ValueError('Price snapshot must include the last recorded valuation date to link new returns')
    for date in newer:
        interval=prices.loc[date]/prices.loc[previous]-1
        cash=(1+p['cash_yield'])**((date-previous).days/p['cash_day_count'])-1
        rows.append(dict(Date=date,From_Date=previous,Recorded_At=timestamp,Source=source,Snapshot_SHA256=snapshot_hash,
                         **interval.to_dict(),CASH_GBP=cash))
        previous=date
    if rows:
        added=pd.DataFrame(rows)
        ledger=added if ledger.empty else pd.concat([ledger,added],ignore_index=True)
        atomic_text(directory/'observations.csv',ledger.to_csv(index=False,date_format='%Y-%m-%d'))
        atomic_text(directory/'observations_manifest.json',json.dumps({'sha256':digest(directory/'observations.csv'),
                    'last_recorded_at':timestamp,'rows':len(ledger)},indent=2)+'\n')
    if revisions:
        path=directory/'source_revision_log.csv'
        old=pd.read_csv(path) if path.exists() else pd.DataFrame()
        atomic_text(path,pd.concat([old,pd.DataFrame(revisions)],ignore_index=True).to_csv(index=False))
    return ledger,{'appended':len(rows),'revisions':len(revisions)}


def simulate(protocol,ledger):
    """Buy and hold, with precommitted target rebalances every three months."""
    if ledger.empty:
        columns=['Date','From_Date','Portfolio','Gross_Value_GBP','Gross_Interval_Return','Cash_Weight']+[f'Net_Value_{bps}bps_GBP' for bps in protocol['cost_bps_per_side']]
        return pd.DataFrame(columns=columns),pd.DataFrame(columns=['Date','Portfolio','Event','Traded_Notional_Fraction'])
    active=protocol['active_assets']+['CASH_GBP'];entry=pd.Timestamp(ledger.Date.iloc[0])
    paths=[];events=[]
    for key,target_map in protocol['weights'].items():
        target=np.array([target_map[a] for a in active]);weights=target.copy()
        current=np.array([protocol['weights']['current'][a] for a in active])
        traded=float(abs(target-current).sum())
        gross=protocol['paper_capital_gbp'];net={bps:gross*(1-bps/10000*traded) for bps in protocol['cost_bps_per_side']}
        events.append({'Date':str(entry.date()),'Portfolio':key,'Event':'Initial allocation','Traded_Notional_Fraction':traded})
        month_count=protocol['rebalance_months'];boundary=entry+pd.DateOffset(months=month_count)
        for i,row in enumerate(ledger.to_dict('records')):
            date=pd.Timestamp(row['Date']);growth=np.array([1+row[a] for a in active])
            period=float(weights@growth);daily=period-1
            gross*=period
            for bps in net:net[bps]*=period
            weights=weights*growth/period
            traded_today=traded if i==0 else 0.
            if date>=boundary:
                traded_today=float(abs(weights-target).sum())
                for bps in net:net[bps]*=1-bps/10000*traded_today
                events.append({'Date':str(date.date()),'Portfolio':key,'Event':'Quarterly rebalance',
                               'Traded_Notional_Fraction':traded_today})
                weights=target.copy()
                while boundary<=date:
                    month_count+=protocol['rebalance_months'];boundary=entry+pd.DateOffset(months=month_count)
            paths.append({'Date':date,'From_Date':pd.Timestamp(row['From_Date']),'Portfolio':key,'Gross_Value_GBP':gross,
                          'Gross_Interval_Return':daily,'Cash_Weight':float(weights[-1]),
                          **{f'Net_Value_{bps}bps_GBP':value for bps,value in net.items()}})
    return pd.DataFrame(paths),pd.DataFrame(events)


def realised_volatility(frame,min_observations=20):
    intervals=frame.loc[frame.Date>frame.From_Date]
    if len(intervals)<min_observations:return None
    elapsed=sum((intervals.Date-intervals.From_Date).dt.days)
    return float(intervals.Gross_Interval_Return.std(ddof=1)*np.sqrt(365.25*len(intervals)/elapsed))
