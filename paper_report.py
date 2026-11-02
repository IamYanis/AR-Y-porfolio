"""Readable recurring reports from the registered protocol and recorded outcomes."""
from datetime import datetime
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from paper_test import load_protocol,read_observations,simulate,realised_volatility,atomic_text,LONDON,digest
from research_report import Report


def generate_paper_report(directory,now=None):
    directory=Path(directory);now=now or datetime.now(LONDON)
    p=load_protocol(directory);ledger=read_observations(directory)
    paths,events=simulate(p,ledger)
    rows=[]
    for key,name in p['names'].items():
        branch=paths.query('Portfolio == @key') if not paths.empty else pd.DataFrame()
        vol=realised_volatility(branch,p['minimum_risk_observations']) if not branch.empty else None
        intervals=max(len(branch)-1,0)
        last=branch.iloc[-1] if len(branch) else None
        rows.append({'Portfolio':name,'Key':key,'Observations':intervals,
            'Predicted_OAS_Volatility_pct':100*p['predicted_volatility'][key]['OAS'],
            'Predicted_Sample_Volatility_pct':100*p['predicted_volatility'][key]['Sample'],
            'Realised_Volatility_pct':None if vol is None else vol*100,
            'Volatility_Forecast_Error_pp':None if vol is None else 100*(vol-p['predicted_volatility'][key]['OAS']),
            'Sample_Volatility_Forecast_Error_pp':None if vol is None else 100*(vol-p['predicted_volatility'][key]['Sample']),
            'Gross_Return_pct':None if last is None else 100*(last.Gross_Value_GBP/p['paper_capital_gbp']-1),
            'Net_Return_25bps_pct':None if last is None else 100*(last.Net_Value_25bps_GBP/p['paper_capital_gbp']-1),
            'Net_Value_25bps_GBP':None if last is None else last.Net_Value_25bps_GBP,
            'Total_Traded_Notional_pct':0 if events.empty else 100*events.query('Portfolio == @key').Traded_Notional_Fraction.sum()})
    summary=pd.DataFrame(rows)
    if ledger.empty:
        state='Awaiting entry valuation';entry=None;last_date=None;completed=0
    else:
        entry=pd.Timestamp(ledger.Date.iloc[0]);last_date=pd.Timestamp(ledger.Date.iloc[-1]);completed=0
        while entry+pd.DateOffset(months=(completed+1)*p['rebalance_months'])<=last_date:completed+=1
        state='Quarterly review available' if completed else 'Collecting forward observations'
    due=pd.Timestamp(p['earliest_entry_date']) if entry is None else entry
    monthly=due+pd.DateOffset(months=p['review_months'])
    if last_date is not None:
        step=1
        while monthly<=last_date:step+=1;monthly=due+pd.DateOffset(months=step*p['review_months'])
    next_quarter=due+pd.DateOffset(months=(completed+1)*p['rebalance_months'])
    archive=directory/'reports'/now.strftime('%Y%m%dT%H%M%S%f%z');archive.mkdir(parents=True)
    report=Report();report.heading('Forward paper portfolio report',1)
    report.paragraph('Status: '+state+'. Generated '+now.strftime('%d %B %Y, %H:%M %Z')+'.')
    report.paragraph('Four hypothetical portfolios, each starting with £'+f"{p['paper_capital_gbp']:,.0f}"+'. No real orders or broker connections are involved.')
    report.heading('Registered before evaluation')
    report.table(pd.DataFrame([{'Registered':p['registered_at_london'],'Training ends':p['training_end'],
        'Earliest entry':p['earliest_entry_date'],'Actual entry':'Pending' if entry is None else str(entry.date()),
        'Latest valuation':'Pending' if last_date is None else str(last_date.date())}]))
    report.paragraph('Entry is the first observed common closing valuation on or after the registered entry date. Performance starts after that closing valuation; earlier historical returns do not count toward this test.')
    report.paragraph(f"All portfolios start with {p['weights']['current']['CASH_GBP']:.2%} target GBP cash. The new {p['sleeve_allocation']:.0%} allocation is funded pro rata from existing risky holdings. Cash weights then drift and return to their registered targets at quarterly rebalances.")
    report.heading('Frozen portfolios and predicted risk')
    display=summary[['Portfolio','Predicted_OAS_Volatility_pct','Predicted_Sample_Volatility_pct']].round(2)
    display.columns=['Paper portfolio','OAS estimated risk (%)','Sample estimated risk (%)'];report.table(display)
    report.paragraph('The single investment minimises estimated OAS portfolio risk among the original candidates. The pair includes that single plus the best second investment with absolute historical correlation no greater than 0.75, split equally. This constraint encourages a different exposure; it does not guarantee the pair will beat the single. Asset choices, target weights and both covariance estimates are frozen at registration.')
    report.heading('Observed results')
    if ledger.empty:
        report.paragraph('No forward entry valuation has been recorded. Realised returns, risk and performance rankings remain pending. The historical research results are not presented as forward performance.')
    else:
        display=summary[['Portfolio','Observations','Gross_Return_pct','Net_Return_25bps_pct','Realised_Volatility_pct','Volatility_Forecast_Error_pp','Sample_Volatility_Forecast_Error_pp','Total_Traded_Notional_pct']].round(3)
        display.columns=['Portfolio','Return intervals','Gross return (%)','Net return, 25bps (%)','Realised risk (%)','Risk minus OAS (pp)','Risk minus Sample (pp)','Traded notional (%)'];report.table(display)
        report.paragraph(f"Realised volatility is published only after {p['minimum_risk_observations']} forward return intervals. It is annualised using the number of observations and elapsed calendar time, because joint market/NAV valuation dates can have gaps. Positive forecast error means realised risk exceeded the frozen estimate. Returns are cumulative, not annualised.")
        if summary.Realised_Volatility_pct.notna().all():
            baseline=float(summary.loc[summary.Key=='current','Realised_Volatility_pct'].iloc[0])
            control=float(summary.loc[summary.Key=='existing_bond','Realised_Volatility_pct'].iloc[0])
            single=float(summary.loc[summary.Key=='single','Realised_Volatility_pct'].iloc[0])
            pair=float(summary.loc[summary.Key=='pair','Realised_Volatility_pct'].iloc[0])
            report.paragraph(f"So far, the single reduces annualised risk by {baseline-single:.2f} percentage points versus the current holdings and {control-single:.2f} points versus the existing-bond control. The pair's additional saving versus the single is {single-pair:.2f} points. These are provisional observations, not a purchase decision.")
    quarter_rows=[]
    if not paths.empty:
        boundaries=events.query('Portfolio == "current" and Event == "Quarterly rebalance"').Date.tolist()
        beginning=entry
        for number,end in enumerate(boundaries,1):
            end=pd.Timestamp(end)
            for key,name in p['names'].items():
                window=paths.query('Portfolio == @key and Date > @beginning and Date <= @end')
                risk=realised_volatility(window,p['minimum_risk_observations'])
                quarter_rows.append({'Quarter':number,'Start':str(beginning.date()),'End':str(end.date()),
                    'Portfolio':name,'Observations':len(window),
                    'Gross_Return_pct':100*(np.prod(1+window.Gross_Interval_Return)-1),
                    'Realised_Volatility_pct':None if risk is None else risk*100,
                    'OAS_Error_pp':None if risk is None else 100*(risk-p['predicted_volatility'][key]['OAS']),
                    'Sample_Error_pp':None if risk is None else 100*(risk-p['predicted_volatility'][key]['Sample'])})
            beginning=end
    quarters=pd.DataFrame(quarter_rows,columns=['Quarter','Start','End','Portfolio','Observations','Gross_Return_pct','Realised_Volatility_pct','OAS_Error_pp','Sample_Error_pp'])
    report.heading('Completed quarterly comparisons')
    if quarters.empty:report.paragraph('No complete forward quarter is available yet. A quarterly review becomes available after the boundary closing valuation has been recorded.')
    else:report.table(quarters.round(3))
    report.heading('Costs, turnover and rules')
    report.paragraph('Costs are illustrative 0, 10 and 25 bps per side of traded notional, including cash. The current portfolio has no initial allocation trade; the other portfolios include their initial reallocation cost. Quarterly rebalances use drifted weights. No terminal liquidation is charged. Net wealth scales for the modelled fee drag; this is a portfolio-level cost scenario, not a share-level execution ledger.')
    report.paragraph('Target portfolios rebalance at the first observed closing valuation on or after each three-month anniversary of entry. There is no asset reselection or retrospective covariance refit. Both frozen risk models are evaluated against later data. GBP cash accrues at the frozen 3% annual assumption using elapsed calendar days / 365.25. Actual commissions, tax, FX spreads and account access are unverified.')
    if not paths.empty:
        cost_rows=[]
        for key,branch in paths.groupby('Portfolio'):
            last=branch.iloc[-1]
            cost_rows.append({'Portfolio':p['names'][key],**{f'Net return at {bps}bps (%)':round(100*(last[f'Net_Value_{bps}bps_GBP']/p['paper_capital_gbp']-1),3) for bps in p['cost_bps_per_side']}})
        report.table(pd.DataFrame(cost_rows))
    report.heading('Monthly update and quarterly review')
    report.paragraph('Next planned monthly review: '+str(monthly.date())+'. Next quarterly review: '+str(next_quarter.date())+('. Dates are provisional until entry is observed.' if entry is None else '.'))
    report.paragraph('Reports update when the command is run; no background schedule or notification has been installed. Run .venv-portfolio/bin/python paper_portfolio.py update --refresh-data monthly to refresh data and generate a new report. Use the report command to regenerate a report without network access.')
    report.paragraph('At quarterly review, compare risk reduction with return differences, costs and the no-new-holding control. Keep the registered trial intact; any changed allocation, selection rule or forecast model belongs in a newly registered trial version.')
    report.heading('Data quality and audit')
    report.paragraph('Valuations require observed prices for all active investments and use the actual BlackRock share-class NAV. No bond proxy, interpolation or holiday carry-forward is used. Different exchange closing times remain a limitation. New intervals are calculated within one price vintage and appended to the ledger. Later provider revisions are logged rather than replacing recorded outcomes; late backfilled dates do not split previously recorded intervals.')
    if not ledger.empty:
        stale=(pd.Timestamp(now.astimezone(LONDON).date())-last_date).days
        gap=int((ledger.Date-ledger.From_Date).dt.days.max())
        revisions=pd.read_csv(directory/'source_revision_log.csv') if (directory/'source_revision_log.csv').exists() else pd.DataFrame()
        report.table(pd.DataFrame([{'Latest valuation age (calendar days)':stale,'Longest interval (calendar days)':gap,'Logged source revisions':len(revisions)}]))
        if stale>7 or gap>7:report.paragraph('Data-quality flag: a valuation gap or stale endpoint exceeds seven calendar days. Interpret risk estimates cautiously and check source coverage before drawing conclusions.')
    report.paragraph('The paper capital is illustrative. Historical selection favoured these candidates using information available at registration, so only subsequently recorded observations assess the forward test. A few months cannot establish performance across a market cycle.')
    report.heading('Frozen weights')
    weight_frame=pd.DataFrame(p['weights']).loc[p['active_assets']+['CASH_GBP']]*100
    weight_frame=weight_frame.rename(columns=p['names']).reset_index().rename(columns={'index':'Asset'})
    report.table(weight_frame.round(3))
    report.heading('Supporting files')
    for label,name in [('Registered protocol','../../protocol.json'),('Frozen weights','../../frozen_weights.csv'),
        ('Summary data','summary.csv'),('Quarterly comparisons','quarters.csv'),('Portfolio paths','paths.csv'),('Allocation and rebalance events','events.csv')]:report.link(label,name)
    if not paths.empty and len(ledger)>1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4))
        for key,branch in paths.groupby('Portfolio'):
            ax.plot(branch.Date,branch.Net_Value_25bps_GBP,label=p['names'][key])
        ax.set(title='Recorded forward paper values after 25bps cost scenario',ylabel='Paper value (GBP)')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
        chart=archive/'paper_values.png';fig.savefig(chart,dpi=150);plt.close(fig);report.image(chart,'Forward observations only; performance starts after the entry valuation.')
    report.save(archive)
    summary.to_csv(archive/'summary.csv',index=False)
    paths.to_csv(archive/'paths.csv',index=False);events.to_csv(archive/'events.csv',index=False)
    quarters.to_csv(archive/'quarters.csv',index=False)
    status={'status':state,'registered_at_london':p['registered_at_london'],'earliest_entry_date':p['earliest_entry_date'],
            'actual_entry':None if entry is None else str(entry.date()),'last_valuation':None if last_date is None else str(last_date.date()),
            'return_intervals':max(len(ledger)-1,0),'completed_quarters':completed,'next_monthly_review':str(monthly.date()),
            'next_quarterly_review':str(next_quarter.date()),'protocol_sha256':digest(directory/'protocol.json'),
            'observations_sha256':None if ledger.empty else digest(directory/'observations.csv')}
    atomic_text(archive/'status.json',json.dumps(status,indent=2)+'\n')
    atomic_text(archive/'run_manifest.json',json.dumps({
        'generated_at_london':now.isoformat(),'protocol_sha256':status['protocol_sha256'],
        'observations_sha256':status['observations_sha256'],'training_sha256':p['training_sha256'],
        'code_sha256':{name:digest(Path(__file__).parent/name) for name in
                       ['paper_test.py','paper_data.py','paper_report.py','paper_portfolio.py']}},indent=2)+'\n')
    latest=directory/'reports'/'latest';latest.mkdir(exist_ok=True)
    for file in archive.iterdir():
        if file.is_file():shutil.copyfile(file,latest/file.name)
    return latest/'SUMMARY_REPORT.html',status
