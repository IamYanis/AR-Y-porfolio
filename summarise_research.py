"""Combine actual-NAV baseline and explicitly labelled ETF proxy sensitivity."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio_validation import markdown_table
from fund_nav import ISIN, NAME


def summarise(root):
    root=Path(root)
    actual=root/'actual';proxy=root/'proxy'
    scores=[]
    for label,folder in [('Actual',actual),('Proxy',proxy)]:
        frame=pd.read_csv(folder/'candidate_stability.csv').set_index('Asset')
        columns=['Mean_Risk_Reduction_pp','Worst_Risk_Reduction_pp','Helpful','Mean_Weight']
        frame=frame[columns].add_prefix(label+'_')
        scores.append(frame)
    review=scores[0].join(scores[1])
    matched=pd.read_csv(actual/'umushroom_diversification_ranking.csv').set_index('Asset')
    review=review.join(matched[['Add_One_Risk_Reduction_pp','Removal_Risk_Increase_pp']])
    returns=pd.read_csv(actual/'gbp_daily_returns.csv',index_col='Date',parse_dates=True)
    register=pd.read_csv(actual/'holdings_register.csv')
    current=[a for a in register.Asset if a!='CASH_GBP']
    candidates=list(review.index)
    corr=returns.drop(columns='CASH_GBP').corr()
    review['Most_Correlated_Existing']=review.index.map(lambda a:corr.loc[a,current].idxmax())
    review['Correlation_With_Existing']=review.index.map(lambda a:corr.loc[a,current].max())
    review['Most_Correlated_Candidate']=review.index.map(lambda a:corr.loc[a,[c for c in candidates if c!=a]].idxmax())
    review['Correlation_With_Candidate']=review.index.map(lambda a:corr.loc[a,[c for c in candidates if c!=a]].max())
    # Research filter, not a statistical confidence test or investment decision.
    review['Review_Priority']=np.where(
        (review.Actual_Mean_Risk_Reduction_pp>.01)&(review.Actual_Helpful>=2/3)&
        (review.Proxy_Mean_Risk_Reduction_pp>.01)&(review.Proxy_Helpful>=.75),
        'Review across both histories','Mixed or limited evidence')
    review=review.sort_values('Actual_Mean_Risk_Reduction_pp',ascending=False)
    review.to_csv(root/'candidate_review.csv',index_label='Asset')
    # Quantify proxy tracking difference only where actual NAVs exist.
    proxy_returns=pd.read_csv(proxy/'gbp_daily_returns.csv',index_col='Date',parse_dates=True)
    aligned=pd.concat([returns[ISIN].rename('Actual_NAV_GBP'),proxy_returns[ISIN].rename('EUNH_Proxy_GBP')],axis=1).dropna()
    aligned.to_csv(root/'bond_proxy_overlap_returns.csv',index_label='Date')
    tracking=pd.DataFrame([{
        'Actual_Fund':NAME,'ISIN':ISIN,'Proxy':'EUNH.DE',
        'First_Return':str(aligned.index.min().date()),'Last_Return':str(aligned.index.max().date()),
        'Observations':len(aligned),'Daily_Return_Correlation':aligned.corr().iloc[0,1],
        'Annualised_Tracking_Error':(aligned.iloc[:,0]-aligned.iloc[:,1]).std()*np.sqrt(252),
        'Actual_Cumulative_Return':(1+aligned.iloc[:,0]).prod()-1,
        'Proxy_Cumulative_Return':(1+aligned.iloc[:,1]).prod()-1}])
    tracking.to_csv(root/'bond_proxy_tracking.csv',index=False)
    actual_manifest=json.loads((actual/'run_manifest.json').read_text())
    proxy_manifest=json.loads((proxy/'run_manifest.json').read_text())
    matched_current=pd.read_csv(actual/'matched_current_return.csv')
    actual_summary=pd.read_csv(actual/'covariance_summary.csv')
    cash=float(actual_manifest['holdings_percent']['CASH_GBP'])/100
    focus=actual_summary[(abs(actual_summary.Cash_Weight-cash)<1e-9)&(actual_summary.Estimator=='OAS')]
    chosen=review[review.Review_Priority=='Review across both histories']
    shortlisted=chosen.reset_index()[['Asset','Actual_Mean_Risk_Reduction_pp','Proxy_Mean_Risk_Reduction_pp','Proxy_Helpful','Add_One_Risk_Reduction_pp','Most_Correlated_Candidate']]
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for filename,label,color in [('umushroom_original_frontier.csv','Existing universe','#c46b16'),
                                  ('umushroom_expanded_frontier.csv','Expanded universe','#184b92')]:
        f=pd.read_csv(actual/filename)
        target=float(matched_current.Return.iloc[0])
        f=f[f.Expected_Return <= max(target*1.7,.15)]
        axes[0].plot(100*f.Volatility,100*f.Expected_Return,label=label,color=color,linewidth=2)
    for i,row in matched_current.iterrows():
        axes[0].scatter(100*row.Volatility,100*row.Return,marker=['*','o','D'][i],s=[140,65,65][i],label=row.Portfolio,zorder=4)
    axes[0].set(xlabel='Annualised estimated volatility (%)',ylabel='Annualised historical mean return (%)',
                title='Actual-NAV baseline: matched return')
    axes[0].legend(fontsize=7)
    out=pd.read_csv(actual/'rolling_matched_returns.csv')
    out=out[out.Estimator=='OAS'].set_index('Universe').loc[['Existing','Expanded']]
    x=np.arange(2)
    axes[1].bar(x-.17,100*out.Predicted_Volatility,.34,label='Estimated before quarter',color='#a9bcd4')
    axes[1].bar(x+.17,100*out.Realised_Volatility,.34,label='Realised during quarter',color='#184b92')
    axes[1].set_xticks(x,['Existing','Expanded'])
    axes[1].set(ylabel='Annualised volatility (%)',title='Equal cash and training return: July–September 2026')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Portfolio research: actual BlackRock share-class NAVs',fontsize=13)
    fig.tight_layout()
    fig.savefig(root/'frontier_comparison.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    matched_display=matched_current.copy()
    matched_display['Historical mean return (%)']=100*matched_display.pop('Return')
    matched_display['Estimated volatility (%)']=100*matched_display.pop('Volatility')
    matched_display=matched_display[['Portfolio','Historical mean return (%)','Estimated volatility (%)','Sharpe']]
    focus_display=focus[['Universe','Mean_Predicted_Volatility','Mean_Realised_Volatility',
                         'Cumulative_Return_Cost_0bps','Cumulative_Return_Cost_25bps']].copy()
    for col in focus_display.columns[1:]: focus_display[col]*=100
    focus_display.columns=['Universe','Estimated risk (%)','Realised risk (%)','Quarter return (%)','Quarter return at 25bps (%)']
    text=f'''# Research findings — {pd.Timestamp(actual_manifest['as_of_exclusive']).strftime('%d %B %Y')}

The primary baseline uses the actual 6.98% holding, **{NAME}**,
ISIN **{ISIN}**, with daily EUR NAVs downloaded directly from BlackRock.
EUNH.DE is retained only for a separate proxy sensitivity run; it is not a holding.
There are 53 investments, including 17 existing holdings, plus 35.31% GBP cash.

The actual NAV export has a 601-day gap ending 10 July 2025. The primary common
return sample is {actual_manifest['return_start']} to {actual_manifest['return_end']}
({actual_manifest['observations']} observations). That permits **one** complete
out-of-sample quarter: 6 July–30 September 2026. The proxy run uses
{proxy_manifest['return_start']} to {proxy_manifest['return_end']}
({proxy_manifest['observations']} observations) and four quarters. Excluding
30 October avoids partially completed market sessions.

## Matched historical return baseline

{markdown_table(matched_display.round(2))}

![Frontier and quarterly validation comparison](frontier_comparison.png)

Both optimised portfolios target the configured holdings' estimated return.
The change from current holdings to the existing-universe optimum is a
reallocation effect; only the further change to the expanded universe measures
the estimated universe expansion benefit. These are in-sample figures, and
cash weights may differ. They are not prospective return forecasts.

## Evaluation with cash held equal

Minimum-risk OAS portfolios at the current 35.31% cash allocation, during the
one available actual-NAV evaluation quarter:

{markdown_table(focus_display.round(2))}

These portfolios do not impose equal expected returns. See
`actual/rolling_matched_returns.csv` for the additional comparison at equal
training-period expected return and equal cash. In that OAS comparison, realised
annualised risk was 5.53% for the existing universe and 5.38% for the expanded
universe: a 0.15 percentage-point reduction, much smaller than the in-sample
improvement. Future realised returns can differ.

Shrinkage improves numerical conditioning but has not been established as the
more accurate risk estimator. `actual/fixed_holdings_forecasts.csv` compares
forecasts for the same holdings; `covariance_summary.csv` also evaluates each
estimator's optimised allocations. Sample covariance and OAS can give different
answers depending on which of these questions is being assessed.

## Research shortlist

{markdown_table(shortlisted.round(4))}

The review filter requires a positive average realised risk reduction greater
than 0.01 percentage points in both histories, improvement in at least two of
three actual-history cash scenarios, and improvement in at least 75% of proxy
scenarios. These thresholds are descriptive and were not pre-registered. The
cash scenarios share dates; they are not independent observations. All candidate
rows, including failed or mixed candidates, are in `candidate_review.csv`.

The columns ending in `_pp` are percentage-point reductions in annualised risk.
The add-one matched-return column comes from the primary in-sample baseline;
it can disagree with the mean-free minimum-risk checks. This disagreement is
useful evidence of dependence on return assumptions. In this run, CVX, PDBC and
XOM show benefits under both the mean-free checks and the matched-return
baseline. Their exposures overlap; CVX and XOM have approximately 0.85 daily
return correlation in the actual-history sample. SHY, TIP and IEF are strong
minimum-risk candidates here but offer effectively no add-one improvement at
the baseline's higher matched-return target. They address a different allocation
objective and should not be counted as three independent diversifiers. Candidate benefits are
not additive. High correlation and shared underlying exposures should guide
which alternatives are considered together.

This is a research shortlist, not a purchase list. Trading availability, investor
eligibility, exact share classes, commissions, FX spreads and tax have not been
verified. US-listed ETF candidates may need account-eligible alternatives. The
cost scenarios apply to the full minimum-risk portfolios, not each candidate's
add-one ranking. Current universe selection and adjusted-price data introduce
retrospective selection and data-vintage biases.

## Actual fund versus proxy

{markdown_table(tracking.round(6))}

These figures cover only overlapping observed return dates. The actual fund
uses an enhanced FTSE EMU government bond strategy; EUNH is a different ETF
tracking a Bloomberg euro government bond index, with a different fee and
valuation basis. High correlation does not establish equivalence.

## Files and reproduction

- `actual/BASELINE_REPORT.md`: primary methodology, covariance diagnostics and results.
- `proxy/BASELINE_REPORT.md`: longer, explicitly labelled proxy sensitivity.
- `frontier_comparison.png`: focused primary frontier and quarterly validation chart.
- `actual/umushroom_expanded_frontiers.png`: full baseline scatter plot.
- `candidate_review.csv`: cross-history ranking and correlation review.
- Each run includes source-file hashes, exact return inputs, weights and validation folds.

From the project folder:

```bash
MPLBACKEND=Agg .venv-portfolio/bin/python AR-Y_portfolio.py --no-show --validate --as-of 2026-10-30 --output-dir reports/baseline-2026-10-30/actual
MPLBACKEND=Agg .venv-portfolio/bin/python AR-Y_portfolio.py --no-show --validate --bond-series proxy --as-of 2026-10-30 --output-dir reports/baseline-2026-10-30/proxy
.venv-portfolio/bin/python summarise_research.py --report-dir reports/baseline-2026-10-30
```

## Primary sources

- [Actual fund, exact share class and benchmark](https://www.blackrock.com/uk/individual/products/326426/blackrock-euro-government-enhanced-index-fund)
- [Official NAV export](https://www.blackrock.com/uk/individual/products/326426/fund/1527484370694.ajax?fileType=xls&fileName=BlackRock-Euro-Government-Enhanced-Index-Fund-D-Acc-EUR_fund&dataType=fund)
- [PDBC product structure and costs](https://www.invesco.com/us/en/financial-products/etfs/invesco-optimum-yield-diversified-commodity-strategy-no-k-1-etf.html)
- [EUNH proxy identity](https://www.ishares.com/de/privatanleger/de/produkte/251740/ishares-euro-government-bond-ucits-et_31)
'''
    (root/'RESEARCH_FINDINGS.md').write_text(text)
    return review

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',required=True)
    args=parser.parse_args()
    print(summarise(args.report_dir).to_string())
