"""Narrative reports derived from saved calculation outputs; no fixed result claims."""
import base64
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio_validation import markdown_table
from fund_nav import ISIN,NAME,parse_nav


class Report:
    def __init__(self): self.markdown=[];self.html=[]
    def heading(self,title,level=2):
        self.markdown.append('#'*level+' '+title+'\n')
        self.html.append(f'<h{level}>{html.escape(title)}</h{level}>')
    def paragraph(self,text):
        self.markdown.append(text+'\n');self.html.append('<p>'+html.escape(text)+'</p>')
    def table(self,frame):
        frame=frame.copy().fillna('—')
        self.markdown.append(markdown_table(frame)+'\n')
        self.html.append('<div class="table-wrap">'+frame.to_html(index=False,escape=True,border=0)+'</div>')
    def link(self,title,url):
        self.markdown.append(f'[{title}]({url})\n')
        self.html.append(f'<p><a href="{html.escape(url,quote=True)}">{html.escape(title)}</a></p>')
    def image(self,path,caption):
        self.markdown.append(f'![{caption}]({path.name})\n')
        encoded=base64.b64encode(path.read_bytes()).decode()
        self.html.append(f'<figure><img alt="{html.escape(caption,quote=True)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    def save(self,output):
        (output/'SUMMARY_REPORT.md').write_text('\n'.join(self.markdown))
        style='''body{font-family:system-ui,sans-serif;color:#203044;max-width:1120px;margin:40px auto;padding:0 24px;line-height:1.6}h1{font-size:2.1rem;line-height:1.2}h2{margin-top:2.2rem;border-bottom:1px solid #d9e2ec;padding-bottom:.3rem}h3{margin-top:1.5rem}a{color:#165a91}table{border-collapse:collapse;font-size:.86rem;white-space:nowrap;width:100%}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #e0e7ee}th{background:#eaf1f7}tbody tr:nth-child(even){background:#f7f9fb}.table-wrap{overflow-x:auto;margin:20px 0}img{width:100%;height:auto}figure{margin:24px 0}figcaption{color:#526579;font-size:.85rem}footer{font-size:.8rem;color:#526579;margin-top:40px}@media print{body{max-width:none;margin:0;padding:0;font-size:10pt}.table-wrap{overflow:visible}table{white-space:normal;font-size:8pt}h2{break-after:avoid}tr,figure{break-inside:avoid}}'''
        document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Portfolio research summary</title><style>'+style+'</style></head><body>'+'\n'.join(self.html)+'<footer>Generated from saved research outputs. This standalone report can be printed or saved as PDF from a browser.</footer></body></html>'
        (output/'SUMMARY_REPORT.html').write_text(document)


def generate_report(output,baseline,source_review):
    output=Path(output);baseline=Path(baseline);source_review=Path(source_review)
    data={}
    for label in ['actual','proxy']:
        d=output/label
        data[label]={n:pd.read_csv(d/(n+'.csv')) for n in
                     ['covariance_calibration','universe_comparison','change_summary','practical_changes','change_forecasts','optimised_scenarios','change_selection']}
        data[label]['settings']=json.loads((d/'settings.json').read_text())
        data[label]['manifest']=json.loads((baseline/label/'run_manifest.json').read_text())
    actual=data['actual'];proxy=data['proxy']
    comparison=pd.read_csv(baseline/'actual'/'matched_current_return.csv')
    old=comparison.set_index('Portfolio').loc['Existing universe at current return']
    new=comparison.set_index('Portfolio').loc['Expanded universe at current return']
    calibration=pd.concat([d['covariance_calibration'].assign(Dataset=label) for label,d in data.items()])
    forecast=calibration.groupby(['Dataset','Lookback','Estimator']).agg(
        Quarters=('Fold','nunique'),Mean_Absolute_Error_pp=('Absolute_Forecast_Error_pp','mean')).reset_index()
    forecast.to_csv(output/'forecast_accuracy_summary.csv',index=False)
    a=actual['change_summary'].query('Type != "Existing holding control" and Total_Allocation == .05').set_index('Change')
    p=proxy['change_summary'].query('Type != "Existing holding control" and Total_Allocation == .05').set_index('Change')
    # Only combinations locked before evaluation in BOTH histories get a
    # cross-history score. Singles are always evaluated in both histories.
    keep=['Evaluation_Quarters','Mean_Realised_Risk_Reduction_pp','Worst_Realised_Risk_Reduction_pp',
          'Mean_Return_Difference_pp','Cumulative_Net_Return_Difference_25bps_pp','Positive_Forecast_Fraction',
          'Extra_Risk_Reduction_vs_CBGA.L_pp']
    cross=a[keep].add_prefix('Actual_').join(p[keep].add_prefix('Proxy_'),how='outer')
    cross['Positive_In_Both_Histories']=(cross.Actual_Mean_Realised_Risk_Reduction_pp>.01)&(cross.Proxy_Mean_Realised_Risk_Reduction_pp>.01)
    cross=cross.sort_values(['Positive_In_Both_Histories','Actual_Extra_Risk_Reduction_vs_CBGA.L_pp'],ascending=False)
    cross.to_csv(output/'ranked_changes.csv',index_label='Change')
    r=Report();r.heading('Portfolio research summary',1)
    r.paragraph('Generated '+datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z')+'. Calculations use frozen baseline inputs through '+actual['manifest']['return_end']+'.')
    r.heading('What the code found')
    r.paragraph(f"The baseline expansion reduces estimated annualised volatility from {old.Volatility:.2%} to {new.Volatility:.2%} at the same historical mean return ({old.Return:.2%}). These are in-sample estimates. The return target is not a forecast or an investment objective.")
    matched=actual['universe_comparison'].query('Lookback == 252 and Estimator == "OAS" and Cap == .20 and Objective == "historical_mean"')
    if len(matched):
        r.paragraph(f"In {matched.Fold.nunique()} actual-NAV evaluation quarter(s), the corresponding equal-cash portfolios averaged {matched.Realised_Volatility_Existing.mean():.2%} versus {matched.Realised_Volatility_Expanded.mean():.2%} realised annualised volatility: a {matched.Realised_Reduction_pp.mean():.2f} percentage-point reduction. The out-of-sample benefit is {'smaller' if matched.Realised_Reduction_pp.mean()<100*(old.Volatility-new.Volatility) else 'larger'} than the baseline estimate.")
    best_fixed=forecast.query('Dataset == "actual"').sort_values('Mean_Absolute_Error_pp').iloc[0]
    r.paragraph(f"For the unchanged holdings, the smallest mean absolute risk-forecast error in the actual-NAV checks is {best_fixed.Mean_Absolute_Error_pp:.2f} percentage points ({best_fixed.Estimator}, {int(best_fixed.Lookback)} observations). This comparison does not establish a durable model winner; the same few test periods are reused across assumptions.")
    metrics={'baseline_existing_risk':float(old.Volatility),'baseline_expanded_risk':float(new.Volatility),
             'baseline_historical_return':float(old.Return),
             'actual_evaluation_quarters':actual['settings']['folds'],'proxy_evaluation_quarters':proxy['settings']['folds'],
             'matched_realised_risk_reduction_pp':float(matched.Realised_Reduction_pp.mean()) if len(matched) else None,
             'best_actual_fixed_holdings_forecast':{'estimator':str(best_fixed.Estimator),'lookback':int(best_fixed.Lookback),
                                                  'mean_absolute_error_pp':float(best_fixed.Mean_Absolute_Error_pp)}}
    (output/'summary_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    positive=cross.query('Positive_In_Both_Histories').index.tolist()
    r.paragraph('At a 5% new sleeve, changes with positive average risk reduction in both histories include '+(', '.join(positive[:8]) if positive else 'none under the stated threshold')+'. Ranking below also compares them with adding to an existing bond holding, so reducing risk alone is not treated as evidence that a new holding is necessary.')
    r.heading('Data coverage and actual holdings')
    coverage=[]
    for label,d in data.items():
        m=d['manifest'];s=d['settings']
        coverage.append({'Calculation series':label,'First return':m['return_start'],'Last return':m['return_end'],
                         'Observations':m['observations'],'Evaluation quarters':s['folds'],'Unused trailing days':s['unused_tail_observations']})
    r.table(pd.DataFrame(coverage))
    r.paragraph(f"The actual bond holding remains {NAME}, ISIN {ISIN}, at {actual['manifest']['holdings_percent'][ISIN]:.2f}%. Cash is fixed at {actual['settings']['cash_weight']:.2%} at each rebalance. The proxy run represents the fund with EUNH.DE for calculations only; it does not change the holdings register.")
    source_rows=[]
    if (source_review/'requests.json').exists():
        for entry in json.loads((source_review/'requests.json').read_text()):
            label=entry['source'];status=entry.get('status',entry.get('error','unknown'))
            outcome={'official':'Exact-share-class NAV export retrieved; gap checked',
                     'finanzfluss':'Public fund page checked; no verified complete daily export obtained',
                     'morningstar':'Access denied; no history obtained'}.get(label,'Checked')
            source_rows.append({'Source':label,'Checked UTC':entry.get('retrieved_at_utc','unknown')[:10],'Response':status,'Outcome':outcome})
        r.table(pd.DataFrame(source_rows))
        if (source_review/'official.txt').exists():
            nav=parse_nav((source_review/'official.txt').read_bytes())
            gap=nav.index.to_series().diff().max()
            r.paragraph(f"The fresh official export still has a maximum gap of {gap.days} calendar days. No longer verified daily series was recovered in this source review. The analysis preserves missing history rather than interpolating NAVs or filling them with a different share class.")
    r.heading('Covariance and estimation-window checks')
    r.table(forecast.round(3))
    r.paragraph('Both 126- and 252-observation windows use the same evaluation quarters, each 63 observations long. Forecast errors above use identical configured holdings under both estimators, which isolates covariance forecast quality. Sample covariance and OAS are refitted using training data only; cash covariance is zero by assumption.')
    spec=[]
    for label,d in data.items():
        for objective,g in d['universe_comparison'].groupby('Objective'):
            spec.append({'Series':label,'Objective':objective,'Paired scenarios':len(g),
                         'Unique quarters':g.Fold.nunique(),'Average predicted saving (pp)':g.Predicted_Reduction_pp.mean(),
                         'Average realised saving (pp)':g.Realised_Reduction_pp.mean(),
                         'Worst realised saving (pp)':g.Realised_Reduction_pp.min()})
    r.heading('How sensitive is the frontier benefit?')
    r.table(pd.DataFrame(spec).round(3))
    failures=sum((d['optimised_scenarios'].Status!='ok').sum() for d in data.values())
    r.paragraph(f"The grid varies covariance model, estimation window, 10%/15%/20% position caps and four objectives: minimum risk without a return target; the historical mean return of current holdings; a target calculated after clipping each asset's daily returns at its training-period 1st/99th percentiles; and half the historical excess return over assumed GBP cash. {failures} mathematically infeasible portfolio targets are recorded separately. Solver failures stop the run; they are not silently omitted.")
    r.paragraph('Scenarios reuse the same quarters and are correlated. Their count is not a statistical sample size. Lower realised risk is shown alongside realised returns; equal estimated returns do not imply equal realised returns.')
    r.heading('Small, explicit portfolio changes')
    r.paragraph('Each proposal adds a 5% or 10% sleeve, funded pro rata from the configured risky holdings, keeping GBP cash unchanged. Singles receive the full sleeve; pairs split it equally. All 36 configured candidates are tested individually. Pair candidates are locked from the first training window before any evaluation quarter; they are not chosen from later test results. Existing bond holdings are included as controls requiring no new investment.')
    display=cross.head(12).reset_index()[['Change','Actual_Mean_Realised_Risk_Reduction_pp','Proxy_Mean_Realised_Risk_Reduction_pp',
        'Actual_Extra_Risk_Reduction_vs_CBGA.L_pp','Actual_Mean_Return_Difference_pp','Actual_Cumulative_Net_Return_Difference_25bps_pp']].round(3)
    display.columns=['5% sleeve','Actual risk saving (pp)','Proxy risk saving (pp)','Extra saving vs existing CBGA (pp)',
                     'Actual quarter return change (pp)','Actual net return change, 25bps (pp)']
    r.table(display)
    r.paragraph('Positive risk savings mean lower annualised volatility. Positive return changes mean a higher return than keeping the configured holdings. The CBGA comparison uses the same sleeve size and funding rule. Negative extra saving means adding the new investment reduced risk less than adding to that existing bond holding. Missing proxy scores mean that pair was not selected before evaluation in the proxy history.')
    controls=actual['change_summary'].query('Type == "Existing holding control" and Total_Allocation == .05')[['Change','Mean_Realised_Risk_Reduction_pp','Mean_Return_Difference_pp']]
    r.heading('No-new-holding controls',3);r.table(controls.round(3))
    pair_rows=[]
    for row in actual['change_summary'].query('Type == "Pair" and Total_Allocation == .05').itertuples():
        first,second=row.Change.split('+')
        benchmark=max(a.loc[first,'Mean_Realised_Risk_Reduction_pp'],a.loc[second,'Mean_Realised_Risk_Reduction_pp'])
        proxy_score=p.loc[row.Change,'Mean_Realised_Risk_Reduction_pp'] if row.Change in p.index else np.nan
        pair_rows.append({'Pair, 2.5% each':row.Change,'Actual risk saving (pp)':row.Mean_Realised_Risk_Reduction_pp,
                          'Extra vs better 5% single (pp)':row.Mean_Realised_Risk_Reduction_pp-benchmark,
                          'Proxy risk saving (pp)':proxy_score,'Actual quarter return change (pp)':row.Mean_Return_Difference_pp})
    r.paragraph('These controls distinguish a new diversification exposure from increasing an existing low-risk holding.')
    r.heading('Does adding a second investment help?',3)
    if pair_rows:
        pairs=pd.DataFrame(pair_rows).sort_values('Extra vs better 5% single (pp)',ascending=False)
        r.table(pairs.round(3))
        pairs.to_csv(output/'pair_comparison.csv',index=False)
        r.paragraph('Pairs use the same 5% total sleeve as singles. The extra-saving column compares each pair with the better of its two 5% single-investment alternatives on the same evaluation dates. That better-single comparator is chosen after observing results, so it is a descriptive comparison, not a preselected trading strategy.')
    else:r.paragraph('No positive-benefit pair candidates were selected from the initial training window.')
    size_rows=[]
    for change in cross.dropna(subset=['Actual_Mean_Realised_Risk_Reduction_pp']).head(5).index:
        group=actual['change_summary'].query('Change == @change').set_index('Total_Allocation')
        if .05 in group.index and .10 in group.index:
            size_rows.append({'Change':change,'Risk saving at 5% (pp)':group.loc[.05,'Mean_Realised_Risk_Reduction_pp'],
                              'Risk saving at 10% (pp)':group.loc[.10,'Mean_Realised_Risk_Reduction_pp'],
                              'Net return change at 5%, 25bps (pp)':group.loc[.05,'Cumulative_Net_Return_Difference_25bps_pp'],
                              'Net return change at 10%, 25bps (pp)':group.loc[.10,'Cumulative_Net_Return_Difference_25bps_pp']})
    r.heading('Allocation-size sensitivity',3);r.table(pd.DataFrame(size_rows).round(3))
    r.paragraph('These 5% and 10% scenarios are illustrative position sizes, not selected allocations. The position-cap grid above concerns full optimisation; the small-change scenarios use explicit weights, whose largest investment weight stays below the baseline 20% cap.')
    r.paragraph('The comparison does not establish that every risk reduction is free of a return trade-off.')
    r.heading('Overlap, costs and implementation')
    daily=pd.read_csv(baseline/'actual'/'gbp_daily_returns.csv',index_col='Date',parse_dates=True)
    corr=daily.drop(columns='CASH_GBP').corr()
    overlaps=[]
    for aa,bb in [('CVX','XOM'),('IAU','GLD'),('TIP','IEF'),('SHY','TIP')]:
        if aa in corr and bb in corr: overlaps.append({'Investments':aa+' / '+bb,'Daily GBP return correlation':round(corr.loc[aa,bb],3)})
    r.table(pd.DataFrame(overlaps))
    r.paragraph('Proposals rebalance quarterly and drift between rebalances. Cost scenarios charge 0, 10 or 25 bps per side of traded notional, including cash, with turnover measured from the prior quarter’s drifted weights. The initial proposal is funded from current holdings; the unchanged comparator has zero initial trade. Ongoing fund fees already reflected in historical prices/NAVs are not subtracted a second time. Commission schedules, FX spreads, taxes, investor eligibility and Umushroom dealing access remain unverified.')
    r.paragraph('For implementation, review economically distinct exposures together, verify the exact available instrument and currency/hedge, then replace scenario costs with actual dealing costs. A public fund listing does not prove access in the user’s account. This report makes no automatic purchase or allocation decision.')
    r.heading('Charts')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,label in enumerate(['actual','proxy']):
        f=forecast.query('Dataset == @label').pivot(index='Lookback',columns='Estimator',values='Mean_Absolute_Error_pp')
        f.plot.bar(ax=axes[i],rot=0,color=['#234e78','#80a8b8'])
        axes[i].set(title=label.title()+' NAV-history risk forecast errors',xlabel='Training observations',ylabel='Mean absolute error (percentage points)')
        axes[i].grid(axis='y',alpha=.2);axes[i].set_axisbelow(True)
    fig.tight_layout();chart=output/'risk_forecast_errors.png';fig.savefig(chart,dpi=160,bbox_inches='tight');plt.close(fig)
    r.image(chart,'Risk-forecast errors for the same holdings; lower is better. Quarters are reused across models and windows.')
    fig,ax=plt.subplots(figsize=(10,5))
    selected=cross.dropna(subset=['Actual_Extra_Risk_Reduction_vs_CBGA.L_pp']).head(10).iloc[::-1]
    values=selected['Actual_Extra_Risk_Reduction_vs_CBGA.L_pp']
    ax.barh(selected.index,values,color=['#24756c' if x>0 else '#b76c53' for x in values])
    ax.axvline(0,color='#334155',linewidth=.8)
    ax.set(xlabel='Extra annualised risk saving over adding to existing CBGA (percentage points)',title='5% sleeve: incremental value of a new holding — actual NAV history')
    fig.tight_layout();chart=output/'incremental_diversification.png';fig.savefig(chart,dpi=160,bbox_inches='tight');plt.close(fig)
    r.image(chart,'A risk reduction versus the current portfolio may still be weaker than a change using an existing holding.')
    r.heading('Limits and decision direction')
    r.paragraph('The actual-NAV history is short. The universe and configured holdings were selected with present-day information; cached adjusted prices are current data vintages. Consequently these are retrospective sensitivity checks, not an unbiased historical strategy backtest. Current holdings, rebalanced in prior quarters, are a reference scenario rather than the actual historical account path. Rankings can change as data grows. The paired changes use training-only selection, but the report’s final ranking uses realised outcomes and must not be treated as a forward-tested strategy.')
    r.paragraph('The next allocation decision should use the changes that survive both histories, deliver incremental benefit over existing-holding controls, and have acceptable return and cost trade-offs. If none meets all of those conditions, retaining the current number of holdings is a valid research conclusion. More evaluation periods and confirmed account access are needed before turning this list into executable allocations.')
    r.heading('Reproduction and supporting files')
    r.paragraph('Run: .venv-portfolio/bin/python run_research.py --rebuild-baseline --output-dir reports/research-latest. This rebuilds both frontiers from cached prices, runs sensitivity checks, and generates this report. To use an existing frozen baseline instead, replace --rebuild-baseline with --baseline-dir reports/baseline-2026-10-30. To regenerate a report without calculations, use --report-only with the same baseline and output directories. Price refresh remains explicit through AR-Y_portfolio.py --refresh-data; changing END is required to extend its date window. Rebuild may download missing cache entries but does not refresh existing ones.')
    for title,file in [('All ranked 5% changes','ranked_changes.csv'),('Covariance accuracy summary','forecast_accuracy_summary.csv'),
                       ('Actual history: all scenarios','actual/optimised_scenarios.csv'),('Actual history: proposed weights','actual/proposed_weights.csv'),
                       ('Proxy history: all scenarios','proxy/optimised_scenarios.csv'),('Run inputs and code hashes','research_manifest.json')]:r.link(title,file)
    r.heading('Public sources')
    r.link('Exact BlackRock share class and NAV export','https://www.blackrock.com/uk/individual/products/326426/blackrock-euro-government-enhanced-index-fund')
    r.link('Finanzfluss fund-data page checked for older history','https://www.finanzfluss.de/informer/fonds/ie000vydx630/')
    r.link('Morningstar exact share class (historical retrieval denied)','https://global.morningstar.com/es/inversiones/fondos/0P0001ONS6/cotizacion')
    r.save(output)
    return cross
