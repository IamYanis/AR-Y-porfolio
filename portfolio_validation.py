"""Rolling research checks. Fit on past observations; evaluate on later ones."""
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, linprog, LinearConstraint, Bounds
from risk_models import shrink_covariance

DAYS = 252
FUND_ISIN = 'IE000VYDX630'
FUND_NAME = 'BlackRock Euro Government Enhanced Index Fund D EUR Acc'


def fit_covariances(train):
    risky = train.drop(columns='CASH_GBP')
    oas, intensity = shrink_covariance(risky)
    matrices = {}
    for name, value in [('Sample', risky.cov().to_numpy()), ('OAS', oas)]:
        c = pd.DataFrame(0., index=train.columns, columns=train.columns)
        c.loc[risky.columns, risky.columns] = DAYS * value
        matrices[name] = c.to_numpy()
    return matrices, intensity


def minimum_risk(cov, bounds, mu=None, target=None):
    """Convex QP; eliminate fixed assets and certify first-order optimality."""
    n = len(bounds)
    a = np.ones((1, n))
    b = np.array([1.])
    if target is not None:
        a = np.vstack([a, mu])
        b = np.append(b, target)
    box = np.array(bounds, dtype=float)
    free = box[:,1] - box[:,0] > 1e-12
    fixed = box[:,0].copy()
    aa = a[:,free]
    bb = b - a[:,~free] @ fixed[~free]
    if not free.any():
        if np.max(abs(a@fixed-b)) > 1e-7:
            raise RuntimeError('Infeasible fixed weights')
        return fixed
    # Drop redundant rows after elimination (e.g. all cash endpoint).
    if np.linalg.matrix_rank(aa) < len(bb):
        if np.linalg.matrix_rank(np.column_stack([aa,bb])) > np.linalg.matrix_rank(aa):
            raise RuntimeError('Inconsistent return constraint')
        keep=[]
        for row in range(len(bb)):
            if np.linalg.matrix_rank(aa[keep+[row]]) > len(keep):
                keep.append(row)
        aa,bb=aa[keep],bb[keep]
    feasible = linprog(np.zeros(free.sum()), A_eq=aa, b_eq=bb,
                      bounds=box[free], method='highs')
    if not feasible.success:
        raise RuntimeError('Infeasible validation constraints: '+feasible.message)
    scale=1/max(float(np.max(np.diag(cov))),1e-8)
    q=cov[np.ix_(free,free)]*scale
    linear=cov[np.ix_(free,~free)]@fixed[~free]*scale
    objective=lambda z: .5*z@q@z + linear@z
    gradient=lambda z:q@z+linear
    result=minimize(lambda z:1000*objective(z),feasible.x,
                    jac=lambda z:1000*gradient(z),method='SLSQP',bounds=box[free],
                    constraints=[{'type':'eq','fun':lambda z:aa@z-bb,'jac':lambda z:aa}],
                    options={'ftol':1e-12,'maxiter':3000})
    def certified(z):
        if (np.max(abs(aa@z-bb)) > 1e-7 or
            np.any(z<box[free,0]-1e-8) or np.any(z>box[free,1]+1e-8)):
            return False
        # A small feasible-direction gradient gap certifies a convex optimum,
        # even when SLSQP reports a line-search precision warning.
        g=gradient(z)
        vertex=linprog(g,A_eq=aa,b_eq=bb,bounds=box[free],method='highs')
        return vertex.success and g@(z-vertex.x) < 1e-7
    if not certified(result.x):
        result=minimize(objective,feasible.x,jac=gradient,hess=lambda z:q,
                        method='trust-constr',bounds=Bounds(box[free,0],box[free,1]),
                        constraints=[LinearConstraint(aa,bb,bb)],
                        options={'gtol':1e-11,'xtol':1e-12,'maxiter':1500,
                                 'initial_barrier_parameter':1e-10,
                                 'initial_barrier_tolerance':1e-10,'barrier_tol':1e-12})
    if not certified(result.x):
        result=minimize(lambda z:1000*objective(z),result.x,
                        jac=lambda z:1000*gradient(z),method='SLSQP',bounds=box[free],
                        constraints=[{'type':'eq','fun':lambda z:aa@z-bb,'jac':lambda z:aa}],
                        options={'ftol':1e-12,'maxiter':3000})
    if not certified(result.x):
        raise RuntimeError(f'Uncertified optimum at return target {target}: {result.message}')
    w=fixed
    w[free]=result.x
    return w


def return_limits(mu, bounds):
    """Lowest and highest feasible arithmetic return under the same bounds."""
    limits = []
    for sign in (1, -1):
        result = linprog(sign * mu, A_eq=np.ones((1, len(mu))),
                         b_eq=[1.], bounds=bounds, method='highs')
        if not result.success:
            raise RuntimeError('Return-range optimisation failed: ' + result.message)
        limits.append(float(result.x @ mu))
    return limits


def compute_frontier(cov, mu, bounds, n_points=150, include_inefficient=False):
    """Minimum-risk boundary, with its dominated lower branch kept separate."""
    if n_points < 2:
        raise ValueError('A frontier grid requires at least two points')
    w_min = minimum_risk(cov, bounds)
    low, high = return_limits(mu, bounds)
    turning_return = float(w_min @ mu)

    def branch(start, stop):
        targets = (np.array([turning_return]) if abs(stop-start) < 1e-10
                   else np.linspace(start, stop, n_points))
        weights = np.array([minimum_risk(cov, bounds, mu, t) for t in targets])
        if np.max(abs(weights @ mu - targets)) > 1e-7:
            raise RuntimeError('Frontier weights miss their return targets')
        vols = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', weights, cov, weights), 0))
        return weights, weights @ mu, vols

    weights, means, vols = branch(turning_return, high)
    result = dict(weights=weights, returns=means, vols=vols, minvar_weights=w_min)
    if include_inefficient and turning_return-low > 1e-10:
        weights, means, vols = branch(low, turning_return)
    else:
        weights, means, vols = np.empty((0, len(mu))), np.array([]), np.array([])
    result.update(inefficient_weights=weights, inefficient_returns=means,
                  inefficient_vols=vols)
    return result


def maximum_sharpe(cov, mu, bounds, frontier, risk_free):
    """Select from the efficient branch and refine its best grid interval.

    With positive excess returns Sharpe's upper level sets are convex, so the
    efficient-branch maximum is unimodal. Retain grid points and endpoints too.
    If all excess returns are nonpositive, report the best grid point only;
    it is not necessarily the maximum Sharpe over inefficient portfolios.
    """
    vols = frontier['vols']
    ratios = np.full(len(vols), -np.inf)
    valid = vols > 1e-10
    ratios[valid] = (frontier['returns'][valid]-risk_free) / vols[valid]
    if not np.isfinite(ratios).any():
        raise RuntimeError('No positive-risk portfolio has a defined Sharpe ratio')
    best = int(np.argmax(ratios))
    choices = [frontier['weights'][best]]
    left, right = max(0, best-1), min(len(vols)-1, best+1)
    low, high = frontier['returns'][[left, right]]

    def negative_sharpe(target):
        w = minimum_risk(cov, bounds, mu, target)
        vol = np.sqrt(max(float(w @ cov @ w), 0))
        if vol <= 1e-10:
            return np.inf
        return -(float(w @ mu)-risk_free) / vol

    if ratios[best] > 0 and high-low > 1e-10:
        result = minimize_scalar(negative_sharpe, bounds=(low, high),
                                 method='bounded', options={'xatol': 1e-10})
        if not result.success:
            raise RuntimeError('Sharpe refinement failed: ' + result.message)
        choices.append(minimum_risk(cov, bounds, mu, result.x))
    return max(choices, key=lambda w: (float(w @ mu)-risk_free) /
               np.sqrt(float(w @ cov @ w)))


def feasible_cloud(bounds, count=100000, seed=42):
    """Illustrative feasible allocations, not a forecast or a complete region."""
    box = np.asarray(bounds, dtype=float)
    free = box[:, 1]-box[:, 0] > 1e-12
    remaining = 1-box[:, 0].sum()
    if remaining < -1e-10 or box[:, 1].sum() < 1-1e-10:
        raise ValueError('Infeasible cloud bounds')
    weights = np.tile(box[:, 0], (count, 1))
    if free.any():
        weights[:, free] += np.random.default_rng(seed).dirichlet(
            np.ones(free.sum()), size=count) * max(remaining, 0)
    valid = ((abs(weights.sum(axis=1)-1) < 1e-10) &
             (weights <= box[:, 1]+1e-10).all(axis=1))
    return weights[valid]


def exclusion_weights(cov, mu, bounds, asset_index, target):
    """None denotes proven infeasibility; numerical solver failures propagate."""
    box = list(bounds)
    box[asset_index] = (0, 0)
    feasible = linprog(np.zeros(len(mu)), A_eq=np.vstack([np.ones(len(mu)), mu]),
                       b_eq=[1., target], bounds=box, method='highs')
    if feasible.status == 2:
        return None
    if not feasible.success:
        raise RuntimeError('Exclusion feasibility check failed: ' + feasible.message)
    return minimum_risk(cov, box, mu, target)


def hold_period(test, weights):
    """Buy and hold within a quarter; weights drift, no daily free rebalancing."""
    asset_growth = np.cumprod(1 + test.to_numpy(), axis=0)
    wealth = asset_growth @ weights
    daily = wealth / np.r_[1., wealth[:-1]] - 1
    end_weights = asset_growth[-1] * weights / wealth[-1]
    return daily, end_weights


def period_metrics(test, weights, cov):
    daily, _ = hold_period(test, weights)
    return {'Predicted_Volatility': float(np.sqrt(max(weights@cov@weights,0))),
            'Realised_Volatility': float(np.std(daily,ddof=1)*np.sqrt(DAYS)),
            'Period_Return': float(np.prod(1+daily)-1)}


def run_validation(returns, holdings, candidates, cap, output, train_days=252, test_days=63):
    if train_days < 2 or test_days < 2:
        raise ValueError('Training and evaluation windows require at least two observations')
    assets = list(returns.columns)
    current = np.array([holdings[a]/100 for a in assets])
    cash = assets.index('CASH_GBP')
    candidate_set = set(candidates)
    actual_cash = current[cash]
    cash_levels = sorted(set([0., .20, actual_cash]))
    rows, candidate_rows, fixed_rows, allocation_rows, matched_rows = [], [], [], [], []
    history = {}
    # Omit incomplete last quarters rather than mix evaluation horizons.
    for fold, stop in enumerate(range(train_days, len(returns)-test_days+1, test_days),1):
        train = returns.iloc[stop-train_days:stop]
        test = returns.iloc[stop:stop+test_days]
        if train.index.max() >= test.index.min():
            raise RuntimeError('Training overlaps evaluation')
        matrices, intensity = fit_covariances(train)
        common = {'Fold':fold, 'Train_Start':str(train.index.min().date()),
                  'Train_End':str(train.index.max().date()),
                  'Test_Start':str(test.index.min().date()), 'Test_End':str(test.index.max().date())}
        for estimator, cov in matrices.items():
            # Same fixed holdings under both estimators isolates forecast quality.
            fixed_rows.append(dict(common, Estimator=estimator,
                                   **period_metrics(test,current,cov)))
        for cash_level in cash_levels:
            bounds = [(cash_level,cash_level) if a=='CASH_GBP' else (0,cap) for a in assets]
            old_bounds = [(0,0) if a in candidate_set else b for a,b in zip(assets,bounds)]
            for estimator,cov in matrices.items():
                for universe, box in [('Existing',old_bounds),('Expanded',bounds)]:
                    w = minimum_risk(cov,box)
                    metrics = period_metrics(test,w,cov)
                    # Per-side traded notional, including cash. No exit liquidation.
                    key=(cash_level,estimator,universe)
                    previous = history.get(key,current)
                    traded = float(np.abs(w-previous).sum())
                    daily, ending = hold_period(test,w)
                    history[key] = ending
                    row=dict(common, Cash_Weight=cash_level, Estimator=estimator,
                             Universe=universe, Shrinkage=intensity if estimator=='OAS' else 0.,
                             Traded_Notional=traded, **metrics)
                    for bps in [0,10,25]:
                        row[f'Period_Return_Cost_{bps}bps']=(1+metrics['Period_Return'])*(1-traded*bps/10000)-1
                    rows.append(row)
                    for a,v in zip(assets,w):
                        allocation_rows.append(dict(common,Cash_Weight=cash_level,
                                                   Estimator=estimator,Universe=universe,Asset=a,Weight=v))
            # Add one asset to the existing universe, without estimating means.
            cov=matrices['OAS']
            base=minimum_risk(cov,old_bounds)
            base_metrics=period_metrics(test,base,cov)
            for candidate in candidates:
                box=list(old_bounds)
                i=assets.index(candidate)
                box[i]=bounds[i]
                w=minimum_risk(cov,box)
                m=period_metrics(test,w,cov)
                candidate_rows.append(dict(common,Cash_Weight=cash_level,Asset=candidate,
                    Candidate_Weight=w[i],
                    Predicted_Risk_Reduction_pp=100*(base_metrics['Predicted_Volatility']-m['Predicted_Volatility']),
                    Realised_Risk_Reduction_pp=100*(base_metrics['Realised_Volatility']-m['Realised_Volatility']),
                    Period_Return_Difference_pp=100*(m['Period_Return']-base_metrics['Period_Return'])))
        mu=train.mean().to_numpy()*DAYS
        target=float(current@mu)
        anchor_bounds=[(actual_cash,actual_cash) if a=='CASH_GBP' else (0,cap) for a in assets]
        anchor_old=[(0,0) if a in candidate_set else b for a,b in zip(assets,anchor_bounds)]
        for estimator,cov in matrices.items():
            for universe,box in [('Existing',anchor_old),('Expanded',anchor_bounds)]:
                w=minimum_risk(cov,box,mu,target)
                matched_rows.append(dict(common,Estimator=estimator,Universe=universe,
                    Training_Target_Return=target, Cash_Weight=actual_cash,
                    **period_metrics(test,w,cov)))
        print(f"Validated fold {fold}: {common['Test_Start']} to {common['Test_End']}", flush=True)
    if not rows:
        raise ValueError('Insufficient data for a complete rolling validation fold')
    rolling=pd.DataFrame(rows)
    detail=pd.DataFrame(candidate_rows)
    fixed=pd.DataFrame(fixed_rows)
    fixed['Absolute_Forecast_Error_pp']=100*abs(fixed.Predicted_Volatility-fixed.Realised_Volatility)
    rolling['Absolute_Forecast_Error_pp']=100*abs(rolling.Predicted_Volatility-rolling.Realised_Volatility)
    def summarise(g):
        return pd.Series({'Folds':len(g),
            'Mean_Predicted_Volatility':g.Predicted_Volatility.mean(),
            'Mean_Realised_Volatility':g.Realised_Volatility.mean(),
            'Mean_Absolute_Forecast_Error_pp':g.Absolute_Forecast_Error_pp.mean(),
            'Total_Traded_Notional':g.Traded_Notional.sum(),
            **{f'Cumulative_Return_Cost_{bps}bps':(1+g[f'Period_Return_Cost_{bps}bps']).prod()-1 for bps in [0,10,25]}})
    summary=pd.DataFrame([dict(zip(['Cash_Weight','Estimator','Universe'],key), **summarise(g).to_dict()) for key,g in rolling.groupby(['Cash_Weight','Estimator','Universe'])])
    stability=detail.groupby('Asset').agg(
        Scenarios=('Fold','size'),
        Mean_Risk_Reduction_pp=('Realised_Risk_Reduction_pp','mean'),
        Worst_Risk_Reduction_pp=('Realised_Risk_Reduction_pp','min'),
        Mean_Predicted_Reduction_pp=('Predicted_Risk_Reduction_pp','mean'),
        Mean_Weight=('Candidate_Weight','mean'),
        Mean_Return_Difference_pp=('Period_Return_Difference_pp','mean')).reset_index()
    frequencies=detail.assign(Helpful=detail.Realised_Risk_Reduction_pp > .01,
                              Selected=detail.Candidate_Weight > .005).groupby('Asset')[['Helpful','Selected']].mean()
    stability=stability.merge(frequencies,on='Asset').sort_values('Mean_Risk_Reduction_pp',ascending=False)
    outputs={'rolling_validation':rolling,'covariance_summary':summary,
             'fixed_holdings_forecasts':fixed,'candidate_validation':detail,
             'candidate_stability':stability,'validation_weights':pd.DataFrame(allocation_rows),
             'rolling_matched_returns':pd.DataFrame(matched_rows)}
    for name,frame in outputs.items():
        frame.to_csv(Path(output)/f'{name}.csv',index=False)
    outputs['train_days']=train_days
    outputs['test_days']=test_days
    outputs['unused_tail_days']=(len(returns)-train_days)%test_days
    return outputs


def markdown_table(frame):
    # Avoid adding a tabulate dependency.
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join(
        '| '+' | '.join(str(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def write_report(output, returns, holdings, candidates, comparison, matched, ranking,
                 cov, sample_cov, shrinkage, validation, as_of, data_dir, bond_series):
    output=Path(output)
    # Save hashes and source metadata for auditability, without changing cache files.
    files=[]
    for file in sorted(Path(data_dir).parent.rglob('*')):
        if file.suffix in ['.csv','.json','.xml']:
            files.append({'file':str(file.relative_to(Path(data_dir).parent)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    manifest={'as_of_exclusive':as_of,'return_start':str(returns.index.min().date()),
              'return_end':str(returns.index.max().date()),'observations':len(returns),
              'python':sys.version,'bond_series':bond_series,'platform':platform.platform(),'numpy':np.__version__,
              'pandas':pd.__version__,'cache_files':files,'holdings_percent':holdings,
              'candidate_assets':list(candidates),'oas_shrinkage':float(shrinkage),
              'max_position':.20,'cash_yield':.03,'trading_days':252,
              'optimisation_cash_weight':holdings['CASH_GBP']/100,
              'negligible_weight_threshold':.001,
              'random_seed':42,'train_observations':252,'test_observations':63}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    text=f'''# Portfolio research baseline

As-of cutoff (exclusive): {as_of}. Return observations: {len(returns)},
{returns.index.min().date()} to {returns.index.max().date()}. All returns in GBP.
53 investments (17 existing and 36 candidates), plus GBP cash.

## Actual holdings and calculation series

The actual 6.98% bond holding is **{FUND_NAME}**, ISIN **{FUND_ISIN}**.
Calculation series for this run: **{bond_series}**.
See `holdings_register.csv` for its calculation series and source status. A proxy,
if used, is not the actual fund and its estimated returns/risks are approximate.
Other holdings and percentages are those configured by the user; no broker
account reconciliation has been performed. Cash is the residual to 100%.

## Baseline

`matched_current_return.csv` directly compares the configured portfolio with
both optimised universes at its estimated return. `matched_current_weights.csv`
records those allocations. All baseline optimisations fix GBP cash to the
actual allocation; the remaining {100-holdings['CASH_GBP']:.2f}% is invested with a 20% cap per
security, long-only weights, and a total allocation of 100%.

OAS shrinkage: {shrinkage:.2%}. Both universes use the same fitted covariance
and common return sample. Cash covariance is zero by assumption, with a fixed
3% annual compounded yield converted to a daily return. Reported historical
returns are arithmetic annualisations, not forecasts or realised CAGR. Sharpe
ratios use the same arithmetic annualisation for the daily assumed cash return;
zero-risk cash portfolios have undefined Sharpe ratios.

{markdown_table(comparison.T.round(5).reset_index().rename(columns={'index':'Portfolio'}))}

At the current estimated return, optimising existing holdings reduces estimated
volatility by {100*(comparison['Current']['Volatility']-comparison['Original Matched Return']['Volatility']):.2f}
percentage points. Expanding the universe reduces it by a further
{100*(comparison['Original Matched Return']['Volatility']-comparison['Expanded Matched Return']['Volatility']):.2f}
percentage points. Both keep the same cash allocation. These are **in-sample**
comparisons, and their return targets depend on noisy historical means.

## Two portfolio views and weights

![Classical risky-only portfolios](umushroom_classical_frontiers.png)

Plot A excludes cash entirely and keeps the 20% security cap. Solid curves are
efficient boundaries; dotted lower curves are inefficient. The clouds contain
feasible sampled allocations, not the whole region, an unconstrained analytic
bullet, or simulated future returns. Both universes use the same sample and
the same fitted OAS covariance matrix.

![Actual portfolio optimisation](umushroom_expanded_frontiers.png)

Plot B fixes cash at {holdings['CASH_GBP']:.2f}%. Triangles mark minimum volatility
at the current portfolio's estimated return. Minimum variance and maximum Sharpe
are separate objectives: their estimated returns need not match the current
portfolio. Grey standalone-security reference points are outside the 20% cap.
Maximum Sharpe is selected from the efficient branch and refined between nearby
return-grid points when excess returns are positive. If all excess returns are
nonpositive, only the best efficient-grid Sharpe is reported; this need not be
the maximum over dominated portfolios. Zero-risk Sharpe is undefined.

`umushroom_expanded_weights.csv` compares current weights with all six optimised
portfolios, in percentages, and includes changes in percentage points.
`umushroom_portfolio_comparison.csv` retains decimal return/volatility units for
compatibility; the console shows percentage units and dimensionless Sharpe.

The connected green points scale PDBC, JNJ, KO, CVX and AAPL to 0%, 25%, 50%,
75% and 100% of **this run's** expanded matched-return weights. Each purchase
reduces all existing risky holdings proportionally; cash stays fixed. These
portfolios are not reoptimised and their estimated returns are allowed to change.
100% here means the five suggested allocations, not 100% of portfolio wealth,
and does not reproduce the separately marked fully optimised expanded portfolio.
[Fractional scenario comparison](umushroom_fractional_allocations.csv) records
annualised return/volatility in percentages, dimensionless Sharpe, cash and the
five allocations. As with the frontiers, these are gross in-sample estimates,
not cost-adjusted backtests or forecasts.

## Potential additions and existing positions

`umushroom_diversification_ranking.csv` ranks each external addition by its
marginal volatility reduction after reoptimising at the **same current estimated
return and cash weight**. It also records allocations in the full expanded
matched-return portfolio and the risk cost of removing that candidate.
Add-one effects are not additive.

{markdown_table(ranking.head(10).round(4))}

`umushroom_existing_positions.csv` shows current and matched-return optimised
weights, changes in percentage points, and the risk increase from excluding each
existing security and reoptimising each universe. Positive exclusion costs mean
the asset helps under this model; infeasible exclusions are labelled, with no
invented numerical penalty. Cash is fixed and is not an exclusion candidate.
Negligible means at most 0.1% of total portfolio value. A zero or negligible
optimal weight is not proof of a bad investment and is not a sell instruction.

`umushroom_matched_frontiers.csv` retains the comparison across the full common
feasible return range and explicitly labels each efficient/inefficient branch.
The original legacy frontier and allocation exports remain available.

## Historical validation

'''
    if validation is None:
        text+='Not run. Use `--validate`.\n'
    else:
        summary=validation['covariance_summary']
        fixed=validation['fixed_holdings_forecasts'].groupby('Estimator').Absolute_Forecast_Error_pp.mean()
        text+=f'''Rolling windows: {validation['train_days']} training observations followed by
{validation['test_days']} non-overlapping evaluation observations. Training
windows overlap; evaluation periods do not. {validation['unused_tail_days']}
trailing observations are excluded from validation because they do not form a
complete quarter; they remain in the baseline.

Each estimator is refitted using training data only. Minimum-risk portfolios
use identical fixed cash budgets (0%, 20%, and the current allocation), long-only
weights, and a 20% cap per investment. This isolates covariance and avoids
historical mean optimisation. Holdings drift within each quarter and rebalance
at the next boundary. Bounds apply at rebalance, not continuously.

Fixed current-holdings mean absolute volatility forecast error (percentage points):
{fixed.round(4).to_string()}

{markdown_table(summary.round(5))}

`rolling_matched_returns.csv` additionally compares portfolios at the same
training-period estimated return (the configured holdings' return), with cash
fixed to the current allocation. Evaluation-period returns are measured rather
than assumed equal. These matched-return evaluation figures are gross of costs.

Costs in the minimum-risk table are scenario assumptions of 0/10/25 bps per side of traded notional,
including cash; actual commissions, FX spreads, tax and fund dealing costs are
not modelled. Initial turnover starts from current holdings. No final liquidation
is charged. These comparisons do not promise equal realised future returns.

Candidate stability below adds each candidate separately to the existing
universe using OAS, without a return target. Helpful means a subsequent risk
reduction greater than 0.01 percentage points; selected means weight above 0.5%.
The scenarios share dates and are correlated, so counts are not independent tests.

{markdown_table(validation['candidate_stability'].head(12).round(5))}

'''
    text+='''## Interpretation and limits

The universe and present holdings were chosen with current information. These
are retrospective robustness checks, not an unbiased historical strategy test;
survivorship and selection bias remain. The history is short and cannot establish stability across a full market cycle.
The official share-class download has a long gap (November 2023 to July 2025);
actual-NAV runs use only the common recent history. No proxy values are spliced
into the actual series. An explicitly labelled proxy run provides a longer
sensitivity check, not a substitute for the actual holding. Current cached
adjusted prices are not point-in-time data vintages. Same-date international
closes and FX fixes are not simultaneous. Holiday prices are carried for at most
three observations before GBP conversion; `data_quality.csv` records missing
and filled dates. This convention can affect short-horizon correlations.

A high allocation or historical Sharpe alone is insufficient to recommend an
investment. The candidate shortlist requires review of overlap, trading access,
share-class suitability and actual costs. US-listed candidate ETFs have not been
verified as available in Umushroom or for the account's investor jurisdiction.
IAU and GLD overlap in gold exposure; sector funds overlap with individual
stocks. Candidate add-one effects are not additive.

## Sources

- [Actual BlackRock fund and share class](https://www.blackrock.com/uk/individual/products/326426/blackrock-euro-government-enhanced-index-fund)
- [EUNH ETF proxy identity](https://www.ishares.com/de/privatanleger/de/produkte/251740/ishares-euro-government-bond-ucits-et_31)
- [CBGA identity and GBP hedge](https://www.blackrock.com/uk/individual/products/337228/ishares-treasury-bond-3-7yr-ucits-etf)
- [DWS listing currencies](https://etf.dws.com/en-gb/AssetDownload/Index/461b6a3b-bbaa-4b84-a11e-fb9fb15145a6/Overall-Factsheet.pdf)
- [OAS formulation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html) (the implementation uses the documented large-p simplification)

Cache source metadata and hashes are recorded in `run_manifest.json`. The CSV
reports contain underlying numbers; `umushroom_expanded_frontiers.png` plots the
baseline. `candidate_validation.csv` includes every fold and cash scenario.
'''
    (output/'BASELINE_REPORT.md').write_text(text)
