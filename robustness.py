"""Sensitivity checks and explicit, cash-neutral portfolio changes."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from portfolio_validation import fit_covariances, minimum_risk, hold_period, period_metrics

DAYS=252


def return_estimate(train, method):
    if method == 'winsorised_mean':
        low,high=train.quantile(.01),train.quantile(.99)
        return train.clip(lower=low,upper=high,axis=1).mean().to_numpy()*DAYS
    return train.mean().to_numpy()*DAYS


def proposed_weights(current, assets, additions, total_allocation, allow_held=False):
    """Fund a sleeve pro rata from existing risky holdings, preserving cash."""
    w=np.array(current,dtype=float,copy=True)
    cash=assets.index('CASH_GBP')
    risky_budget=1-w[cash]
    if not additions or len(set(additions)) != len(additions):
        raise ValueError('Supply distinct investment additions')
    if not 0 < total_allocation <= risky_budget:
        raise ValueError('New allocation must fit inside the existing risky budget')
    for a in additions:
        if a not in assets or a == 'CASH_GBP' or (w[assets.index(a)] > 1e-12 and not allow_held):
            raise ValueError('Additions must be unheld investments in the universe')
    mask=np.ones(len(w),dtype=bool);mask[cash]=False
    w[mask]*=(risky_budget-total_allocation)/risky_budget
    for a in additions: w[assets.index(a)]+=total_allocation/len(additions)
    if not np.isclose(w.sum(),1) or np.any(w<0):
        raise RuntimeError('Invalid funded proposal')
    return w


def evaluate_change(test,w,current,cov,previous=None,previous_base=None):
    daily,end=hold_period(test,w)
    base_daily,base_end=hold_period(test,current)
    traded=np.abs(w-(current if previous is None else previous)).sum()
    base_traded=np.abs(current-(current if previous_base is None else previous_base)).sum()
    risk=float(np.std(daily,ddof=1)*np.sqrt(DAYS))
    base_risk=float(np.std(base_daily,ddof=1)*np.sqrt(DAYS))
    ret=float(np.prod(1+daily)-1);base_ret=float(np.prod(1+base_daily)-1)
    row={'Predicted_Risk_Reduction_pp':100*(np.sqrt(current@cov@current)-np.sqrt(w@cov@w)),
         'Realised_Risk_Reduction_pp':100*(base_risk-risk),
         'Base_Realised_Risk':base_risk,'Change_Realised_Risk':risk,
         'Period_Return':ret,'Base_Period_Return':base_ret,
         'Period_Return_Difference_pp':100*(ret-base_ret),
         'Traded_Notional':float(traded),'Base_Traded_Notional':float(base_traded)}
    for bps in [0,10,25]:
        row[f'Net_Return_{bps}bps']=(1+ret)*(1-bps/10000*traded)-1
        row[f'Base_Net_Return_{bps}bps']=(1+base_ret)*(1-bps/10000*base_traded)-1
    return row,end,base_end


def scenario_grid(returns,holdings,candidates,output,lookbacks=(126,252),caps=(.10,.15,.20),test_days=63):
    assets=list(returns.columns)
    current=np.array([holdings[a]/100 for a in assets])
    if not np.isfinite(returns.to_numpy()).all() or not np.isclose(current.sum(),1):
        raise ValueError('Require finite return inputs and fully specified holdings')
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError('Return dates must be ordered and unique')
    cash=current[assets.index('CASH_GBP')]
    first=max(lookbacks)
    stops=list(range(first,len(returns)-test_days+1,test_days))
    if not stops: raise ValueError('Insufficient observations for a complete common evaluation quarter')
    optimised,calibration,practical,predictions,selection,weights=[],[],[],[],[],[]
    previous={};optimised_previous={}
    # Lock pair candidates before the FIRST evaluation period. Do not use the
    # prior full-sample candidate ranking to choose these pairs.
    initial=returns.iloc[:first]
    initial_cov=fit_covariances(initial)[0]['OAS']
    initial_scores=[]
    for a in candidates:
        w=proposed_weights(current,assets,(a,),.05)
        initial_scores.append((a,np.sqrt(current@initial_cov@current)-np.sqrt(w@initial_cov@w)))
    leaders=[a for a,benefit in sorted(initial_scores,key=lambda x:(-x[1],x[0])) if benefit>0][:4]
    pair_choices=list(combinations(leaders,2))
    controls=[a for a in ['CBGA.L','IE000VYDX630'] if a in assets and current[assets.index(a)]>0]
    changes=[(a,) for a in candidates]+pair_choices+[(a,) for a in controls]
    for choice in changes:
        for budget in [.05,.10]:
            w=proposed_weights(current,assets,choice,budget,allow_held=True)
            change='+'.join(sorted(choice))
            selection.append({'Change':change,'Total_Allocation':budget,'Type':'Existing holding control' if choice[0] in controls else ('Pair' if len(choice)==2 else 'Single'),
                              'Selected_Using':'Existing holding control' if choice[0] in controls else ('Initial training covariance only' if len(choice)==2 else 'All configured candidates'),
                              'Selection_Train_End':str(initial.index.max().date())})
            for a,value in zip(assets,w):
                weights.append({'Change':change,'Total_Allocation':budget,'Asset':a,'Weight':value,
                                'Current_Weight':current[assets.index(a)]})
    for fold,stop in enumerate(stops,1):
        test=returns.iloc[stop:stop+test_days]
        meta={'Fold':fold,'Test_Start':str(test.index.min().date()),'Test_End':str(test.index.max().date())}
        for lookback in lookbacks:
            train=returns.iloc[stop-lookback:stop]
            if train.index.max()>=test.index.min(): raise RuntimeError('Training/evaluation overlap')
            matrices,intensity=fit_covariances(train)
            spec=dict(meta,Lookback=lookback,Train_Start=str(train.index.min().date()),Train_End=str(train.index.max().date()))
            for estimator,cov in matrices.items():
                m=period_metrics(test,current,cov)
                calibration.append(dict(spec,Estimator=estimator,
                    Condition_Number=float(np.linalg.cond(cov[np.ix_([i for i,a in enumerate(assets) if a!='CASH_GBP'],[i for i,a in enumerate(assets) if a!='CASH_GBP'])])),
                    Absolute_Forecast_Error_pp=100*abs(m['Predicted_Volatility']-m['Realised_Volatility']),**m))
                for cap in caps:
                    new_box=[(cash,cash) if a=='CASH_GBP' else (0,cap) for a in assets]
                    old_box=[(0,0) if a in candidates else box for a,box in zip(assets,new_box)]
                    for objective in ['minimum_risk','historical_mean','winsorised_mean','half_historical_excess']:
                        mu=return_estimate(train,objective)
                        anchor=float(current@mu)
                        cash_return=float(mu[assets.index('CASH_GBP')])
                        target=None if objective=='minimum_risk' else (cash_return+.5*(anchor-cash_return) if objective=='half_historical_excess' else anchor)
                        for universe,box in [('Existing',old_box),('Expanded',new_box)]:
                            row=dict(spec,Estimator=estimator,Cap=cap,Objective=objective,Universe=universe,
                                     Target_Return=np.nan if target is None else target)
                            # Distinguish mathematically infeasible targets from solver failures.
                            if target is not None:
                                feasible=linprog(np.zeros(len(assets)),A_eq=np.vstack([np.ones(len(assets)),mu]),
                                                 b_eq=[1,target],bounds=box,method='highs')
                                if not feasible.success:
                                    optimised.append(dict(row,Status='infeasible_target'));continue
                            w=minimum_risk(cov,box,mu if target is not None else None,target)
                            metric=period_metrics(test,w,cov)
                            key=(lookback,estimator,cap,objective,universe)
                            traded=float(np.abs(w-optimised_previous.get(key,current)).sum())
                            _,end=hold_period(test,w);optimised_previous[key]=end
                            row.update(Status='ok',Traded_Notional=traded,**metric)
                            row['Net_Return_25bps']=(1+metric['Period_Return'])*(1-.0025*traded)-1
                            optimised.append(row)
                # Fixed change forecasts vary by estimator and window; do not
                # count these replicated forecasts as new realised test periods.
                for choice in changes:
                    for budget in [.05,.10]:
                        w=proposed_weights(current,assets,choice,budget,allow_held=True)
                        predictions.append(dict(spec,Estimator=estimator,Change='+'.join(sorted(choice)),Total_Allocation=budget,
                            Predicted_Risk_Reduction_pp=100*(np.sqrt(current@cov@current)-np.sqrt(w@cov@w)),
                            Historical_Return_Difference_pp=100*((w-current)@return_estimate(train,'historical_mean')),
                            Winsorised_Return_Difference_pp=100*((w-current)@return_estimate(train,'winsorised_mean'))))
            if lookback==first: # One realised evaluation per unique change and quarter.
                cov=matrices['OAS']
                for choice in changes:
                    for budget in [.05,.10]:
                        change='+'.join(sorted(choice));key=(change,budget)
                        w=proposed_weights(current,assets,choice,budget,allow_held=True)
                        prior,prior_base=previous.get(key,(None,None))
                        row,end,end_base=evaluate_change(test,w,current,cov,prior,prior_base)
                        previous[key]=(end,end_base)
                        practical.append(dict(meta,Change=change,Type='Existing holding control' if choice[0] in controls else ('Pair' if len(choice)==2 else 'Single'),Total_Allocation=budget,**row))
        print(f"Sensitivity fold {fold}/{len(stops)}: {meta['Test_Start']} to {meta['Test_End']}",flush=True)
    frames={'optimised_scenarios':pd.DataFrame(optimised),'covariance_calibration':pd.DataFrame(calibration),
            'practical_changes':pd.DataFrame(practical),'change_forecasts':pd.DataFrame(predictions),
            'change_selection':pd.DataFrame(selection),'proposed_weights':pd.DataFrame(weights)}
    successful=frames['optimised_scenarios'].query("Status == 'ok'")
    keys=['Fold','Lookback','Estimator','Cap','Objective']
    old=successful.query("Universe == 'Existing'").set_index(keys)
    new=successful.query("Universe == 'Expanded'").set_index(keys)
    paired=old[['Predicted_Volatility','Realised_Volatility','Period_Return']].join(
        new[['Predicted_Volatility','Realised_Volatility','Period_Return']],lsuffix='_Existing',rsuffix='_Expanded',how='inner').reset_index()
    paired['Predicted_Reduction_pp']=100*(paired.Predicted_Volatility_Existing-paired.Predicted_Volatility_Expanded)
    paired['Realised_Reduction_pp']=100*(paired.Realised_Volatility_Existing-paired.Realised_Volatility_Expanded)
    paired['Return_Difference_pp']=100*(paired.Period_Return_Expanded-paired.Period_Return_Existing)
    if (paired.Predicted_Reduction_pp<-.002).any(): raise RuntimeError('Expanded feasible set unexpectedly increases optimum risk')
    frames['universe_comparison']=paired
    grouped=[]
    for (change,budget),g in frames['practical_changes'].groupby(['Change','Total_Allocation']):
        forecast=frames['change_forecasts'].query('Change == @change and Total_Allocation == @budget')
        control_metrics={}
        for control in controls:
            reference=frames['practical_changes'].query('Change == @control and Total_Allocation == @budget').set_index('Fold')
            excess=g.set_index('Fold').Realised_Risk_Reduction_pp-reference.Realised_Risk_Reduction_pp
            control_metrics[f'Extra_Risk_Reduction_vs_{control}_pp']=excess.mean()
            control_metrics[f'Beats_{control}_Fraction']=(excess>.01).mean()
        grouped.append({'Change':change,'Type':g.Type.iloc[0],'Total_Allocation':budget,'Evaluation_Quarters':len(g),
            'Mean_Realised_Risk_Reduction_pp':g.Realised_Risk_Reduction_pp.mean(),
            'Worst_Realised_Risk_Reduction_pp':g.Realised_Risk_Reduction_pp.min(),
            'Helpful_Quarter_Fraction':(g.Realised_Risk_Reduction_pp>.01).mean(),
            'Positive_Forecast_Fraction':(forecast.Predicted_Risk_Reduction_pp>.01).mean(),
            'Mean_Return_Difference_pp':g.Period_Return_Difference_pp.mean(),
            'Total_Traded_Notional':g.Traded_Notional.sum(),**control_metrics,
            **{f'Cumulative_Net_Return_Difference_{bps}bps_pp':100*((1+g[f'Net_Return_{bps}bps']).prod()-(1+g[f'Base_Net_Return_{bps}bps']).prod()) for bps in [0,10,25]}})
    frames['change_summary']=pd.DataFrame(grouped).sort_values('Mean_Realised_Risk_Reduction_pp',ascending=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items(): frame.to_csv(output/(name+'.csv'),index=False)
    settings={'lookbacks':list(lookbacks),'caps':list(caps),'test_days':test_days,'first_evaluation_offset':first,
              'cash_weight':float(cash),'folds':len(stops),'unused_tail_observations':len(returns)-(stops[-1]+test_days),
              'pair_candidates_locked_before_evaluation':leaders,'funding':'Pro rata from configured risky holdings',
              'new_sleeve_allocations':[.05,.10], 'cost_bps_per_side':[0,10,25],
              'return_start':str(returns.index.min().date()),'return_end':str(returns.index.max().date())}
    (output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    return frames,settings
