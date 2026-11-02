# Portfolio data and research

The recorded 6.98% bond holding is BlackRock Euro Government Enhanced Index Fund
D EUR Acc, ISIN IE000VYDX630. Actual holdings are distinct from calculation series.

## Normal analysis

From this project directory:

```bash
.venv-portfolio/bin/python AR-Y_portfolio.py
```

Asset and FX prices are cached in `data/prices/`. Missing series are downloaded
once; cached prices are reused without a network request. CSV prices remain in
original quote currency. Companion JSON files record requests and download dates.
GBP conversion happens during analysis. Cache paths are relative to the script.

The actual bond fund uses BlackRock's public share-class NAV download in
`data/nav/`. Its `.xml` file is the original XLS/XML Spreadsheet response; `.csv`
is the readable NAV extract; `.json` records the exact source URL, timestamp and
hash. The parser checks the ISIN and EUR share-class currency. Accumulating NAV
changes include retained fund income. The source contains a long historical gap;
missing NAVs are not replaced with proxy values. The common sample consequently
begins in July 2025 with the current download.

## Refresh

```bash
.venv-portfolio/bin/python AR-Y_portfolio.py --refresh-data
```

Refresh replaces the full configured historical window for Yahoo adjusted prices
and re-downloads the official BlackRock NAV export. This picks up corporate-action
adjustments. Advance `END` in the script to include newer dates; it is exclusive.
A new START/END window creates separate Yahoo cache files. No data expires or
refreshes automatically. The run prints its observed common date range.

Failed or invalid downloads preserve the previous raw cache and stop the run.
An interrupted multi-asset refresh can leave mixed update dates: rerun refresh
to complete it. Yahoo CSV and raw BlackRock XML replacements are atomic; their
metadata and derived NAV CSV are separate writes. Source files and hashes appear
in every run manifest.

## Validation and reproducible reports

```bash
MPLBACKEND=Agg .venv-portfolio/bin/python AR-Y_portfolio.py --no-show --validate --as-of 2026-10-30 --output-dir reports/baseline-2026-10-30/actual
MPLBACKEND=Agg .venv-portfolio/bin/python AR-Y_portfolio.py --no-show --validate --bond-series proxy --as-of 2026-10-30 --output-dir reports/baseline-2026-10-30/proxy
MPLBACKEND=Agg .venv-portfolio/bin/python summarise_research.py --report-dir reports/baseline-2026-10-30
```

`--as-of` excludes the specified date and later to avoid incomplete daily sessions.
It defaults to today's date in Europe/London. `--bond-series actual` is the default;
`proxy` explicitly represents the actual fund with the different EUNH.DE ETF.
Both modes keep the actual fund's name, ISIN and 6.98% holding in the register.
They must use different output folders if you want to retain both runs.

`--output-dir` defaults to `reports/latest`; use a new folder to retain a snapshot.
`--no-show` saves charts without opening a window. Existing root-level charts and
CSVs from earlier script versions are legacy outputs; new reports are saved to
the selected output directory.

Read `RESEARCH_FINDINGS.md` in the combined report directory, then each run's
`BASELINE_REPORT.md`. Validation estimates on 252 past observations and evaluates
on the following 63 observations, with no overlap between evaluation quarters.
Sample covariance and OAS are evaluated separately, with fixed cash budgets and
buy-and-hold weights within each quarter. The matched-return check also fixes
the estimated training return. Candidate checks add one investment at a time.
The shortlist is retrospective research, with limited history and no trading
availability or account reconciliation implied.

Run local tests:

```bash
.venv-portfolio/bin/python -m unittest test_price_cache test_fund_nav test_portfolio_validation
```

## Full robustness summary

For a single command that rebuilds the baseline from cache, runs the additional
sensitivity checks and generates an HTML/Markdown report:

```bash
.venv-portfolio/bin/python run_research.py --rebuild-baseline
```

Read `RESEARCH_README.md` for report-only regeneration, dated outputs and the
small-change comparisons. The generated report is `reports/research-latest/SUMMARY_REPORT.html`.

## Markowitz views and weight optimisation

The main script now produces two different charts:

- `umushroom_classical_frontiers.png`: risky securities only, with zero cash,
  feasible allocation samples, minimum-variance points, and separately styled
  efficient and inefficient branches for both universes.
- `umushroom_expanded_frontiers.png`: actual-portfolio comparison with GBP cash
  fixed at the configured holding (currently 35.31%). It marks the current
  portfolio, both minimum-variance portfolios, both efficient-branch maximum
  Sharpes, and both minimum-volatility portfolios at the current estimated return.

All optimised portfolios are long-only, fully invested, and capped at 20% per
security. Cash is exempt from that cap. The current portfolio's weights, return
series, annualisation and covariance treatment are unchanged. Both universes
use the same common GBP history and the same fitted OAS estimator.

`umushroom_expanded_weights.csv` contains the six optimised allocations alongside
current weights, with percentage-point changes. `umushroom_diversification_ranking.csv`
ranks add-one marginal volatility savings at the current estimated return and
fixed cash. `umushroom_existing_positions.csv` evaluates exclusion of each
existing security in both universes at that same target. Infeasible exclusions
are labelled; numerical solver failures stop the calculation. Negligible weights
mean at most 0.1% of total portfolio value, not an automatic sell decision.

`BASELINE_REPORT.md` contains the concise comparison and embeds both plots.
Existing supporting CSV names remain available, including decimal-unit metrics;
the console and plots display returns and volatility as annualised percentages.
Matched-return CSV rows now label efficient and inefficient branches separately.
The feasible clouds illustrate sampled allocations, not the complete feasible
region or a forecast. Maximum Sharpe is refined along the efficient branch when
positive excess returns exist; otherwise it is the best efficient-grid point.

Use a new `--output-dir` to preserve earlier research. Historical means are noisy
in-sample estimates. The main script rejects future `--as-of` dates and excludes
the specified date and later; cached adjusted prices are still current vintages,
not point-in-time data. No frozen paper-test definitions or observations change.

Focused tests in `test_portfolio_validation.py` cover branch separation, full
investment and caps, nested-universe risk, analytic maximum Sharpe, fixed-cash
clouds, infeasible exclusions, propagated solver failures and future cutoffs.
Cached actual/proxy dry runs at the exclusive 2026-10-30 cutoff also verified
that current-portfolio metrics and historical return CSVs exactly match the
previous baselines. The optimisation assumptions changed; the observations did not.

## Fractional suggested allocations

Plot B also connects five scenarios: 0%, 25%, 50%, 75% and 100% of the PDBC,
JNJ, KO, CVX and AAPL weights in the **current run's expanded matched-return
solution** (`Expanded Matched Return` in `umushroom_expanded_weights.csv`). The
weights are taken directly from that calculated solution, not hardcoded or read
from a possibly stale prior CSV. All existing risky holdings fund the purchases
proportionally; GBP cash stays at its current allocation. Other external assets
remain unheld. Every scenario is checked for full investment, long-only bounds,
the 20% security cap and unchanged cash.

`umushroom_fractional_allocations.csv` contains five rows of actual calculated
annualised return/volatility in percent, Sharpe, cash and the five asset weights.
The 0% row is exactly the current portfolio. The 100% row introduces all five
suggested weights but retains the existing holdings' relative proportions; it
is **not** the fully optimised expanded portfolio, which is marked separately.
Fractional scenarios are not constrained to have equal estimated returns or
monotonically improving risk/Sharpe. These are gross in-sample comparisons using
the same GBP means, covariance and metrics as the other portfolios.

Focused tests in `test_robustness.py` check exact 0% identity, all five target
fractions, pro rata funding, fixed cash, position caps, changing target weights,
zero targets and invalid unequal allocations.
