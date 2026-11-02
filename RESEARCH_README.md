# Automatic portfolio research report

From the AR-Y_portfolio folder, run:

```bash
.venv-portfolio/bin/python run_research.py --rebuild-baseline
```

This calculates both baseline frontiers from the local price/NAV cache, runs the
robustness checks, and creates `reports/research-latest/SUMMARY_REPORT.html` and
`SUMMARY_REPORT.md`. HTML includes its charts and can be viewed offline or printed
to PDF from a browser. CSV links refer to supporting files alongside the report.
No allocation is executed and recorded holdings are not changed.

To retain a dated snapshot, provide a different output folder:

```bash
.venv-portfolio/bin/python run_research.py --rebuild-baseline --as-of 2026-10-30 --output-dir reports/research-2026-10-30
```

Rebuild writes both baseline runs under the output folder's `baseline/` directory.
The cutoff date is exclusive. Without `--as-of`, the baseline uses today's London
date and excludes that day's potentially incomplete market sessions. Rebuild
reuses existing cached prices; it can download missing cache entries. To update
existing prices, use `AR-Y_portfolio.py --refresh-data` explicitly. Advance END
in that script to extend its configured date window. Refresh the proxy run as
well if the EUNH proxy's cache needs updating.

To analyse a previously saved baseline without rebuilding it:

```bash
.venv-portfolio/bin/python run_research.py --baseline-dir reports/baseline-2026-10-30 --output-dir reports/research-latest
```

To regenerate the narrative/charts from saved sensitivity results:

```bash
.venv-portfolio/bin/python run_research.py --report-only --output-dir reports/research-latest
```

The runner uses the output folder's baseline if present. Otherwise provide the
same `--baseline-dir` used for the original run. Report-only checks that baseline
inputs and sensitivity CSV/JSON files have not changed. If they changed, rerun
calculations. Normal runs overwrite files in their selected output folder.

## What is tested

- Sample covariance and OAS, with 126 and 252 past observations, using identical
  evaluation dates. Each evaluation quarter contains 63 observations.
- Position caps of 10%, 15%, and 20%, with the configured cash fraction fixed.
- Minimum risk, matched historical return, matched winsorised historical return,
  and half the historical excess return over the assumed cash yield. Winsorised
  means clip at training-period 1st/99th percentiles. Targets are scenario inputs,
  not forecasts. Infeasible targets are recorded; numerical failures stop the run.
- All configured new candidates individually, in 5% and 10% sleeves funded pro
  rata from existing risky holdings. Weights drift between quarterly rebalances.
- Equal-weight pairs selected from the four candidates with the best positive
  estimated risk reduction in the initial training window. This selection is
  locked before any evaluation date. Pair sets can differ between actual and
  proxy histories. The report does not give untested pairs proxy scores.
- Adding to existing CBGA and actual BlackRock holdings as controls requiring no
  new investment. This separates new diversification from simple reallocation.
- Per-side cost scenarios of 0/10/25 bps of traded notional, including cash.

## Important files

`SUMMARY_REPORT.html` is the main readable report; `SUMMARY_REPORT.md` is its
text version. `ranked_changes.csv` compares 5% proposals across actual and proxy
histories; it includes return differences and incremental risk savings over
existing CBGA. `pair_comparison.csv` compares the selected pairs with their
single-investment alternatives. `summary_metrics.json` contains the main numbers.

Under `actual/` and `proxy/`, each scenario's weights, realised changes, forecasts,
optimised portfolios, selection dates and settings are retained. `research_manifest.json`
records input/output hashes and calculation code hashes. `report_manifest.json`
records report generation time and code hash. The baseline directory contains
exact daily returns, covariance/frontier outputs and holdings registers.

The generated report does not hardcode result values. It reads the saved outputs,
formats the findings, and embeds generated charts. Changing the baseline requires
recalculating the checks before regenerating the report.

## Source review and evidence limits

`source_review/requests.json` records the public-data review performed on
30 October 2026. A fresh official BlackRock export retained the 601-day gap in
exact-share-class NAVs. A Finanzfluss public page did not supply a verified
complete daily series; Morningstar denied retrieval. The gap is not interpolated
or filled with another fund. This review is stored evidence, not an automatic
network request on every report run.

The current actual-NAV baseline supports one evaluation quarter, and its proxy
sensitivity supports four. Model scenarios share dates and are not independent
tests. Current holdings and today's selected universe introduce retrospective
selection bias. The report describes evidence and trade-offs; it does not verify
account dealing access, investor eligibility, commissions, FX spreads or tax.

## Tests

```bash
.venv-portfolio/bin/python -m unittest test_price_cache test_fund_nav test_portfolio_validation test_robustness test_research_report
```
