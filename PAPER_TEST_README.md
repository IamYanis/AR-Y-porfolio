# Forward paper portfolio test

The registered trial lives in:

`paper_tests/forward-2026-10-30/`

Open its report:

`paper_tests/forward-2026-10-30/reports/latest/SUMMARY_REPORT.html`

The Markdown version is in the same folder. The initial report correctly shows
“Awaiting entry valuation”: no historical research return is counted as forward
performance.

## Registered comparisons

Each portfolio starts with an illustrative £100,000:

1. Current configured Umushroom weights.
2. A 5% sleeve allocated to existing CBGA.L.
3. A 5% sleeve allocated to CVX.
4. A 5% sleeve split equally between CVX and PDBC.

The sleeves are funded pro rata from current risky holdings. All portfolios have
35.31% initial target GBP cash. Funding the existing-bond control also scales its
original position before adding the sleeve; its net weight increase is therefore
less than five percentage points. Frozen target weights are in `frozen_weights.csv`.

Selection uses the last 252 completed actual-NAV baseline return observations,
ending 29 October 2026. The single minimises estimated OAS risk; its companion
minimises pair risk subject to absolute training correlation <= 0.75. The pair
keeps the same total sleeve as the single. It is a comparison to test, not an
assumption that more holdings must improve results. Both OAS and sample risk
estimates are frozen before evaluation, without later refits or asset reselection.

The actual BlackRock fund, ISIN IE000VYDX630, remains in every portfolio with the
appropriate funded weight. EUNH.DE is not used as a forward substitute.

## Entry and ongoing rules

The earliest entry is 2 November 2026. Actual entry is the first common observed
closing valuation on or after that date; performance starts after that close.
Only completed dates before the update's exclusive `--as-of` cutoff are used.
The default cutoff is today's London date, excluding incomplete current sessions.
No artificial entry prices or backdated performance are generated.

Positions drift between quarterly rebalances to their frozen targets. Rebalance
occurs at the first observed joint close on or after each three-month anniversary
of actual entry. Monthly reports and completed-quarter comparisons show returns,
forecast errors, turnover and 0/10/25 bps per-side cost scenarios. Cash accrues at
the frozen 3% assumption with elapsed calendar days / 365.25. Fees are represented
as a portfolio-level net-wealth adjustment; this is not a share-execution simulator.

The report publishes realised volatility only after 20 recorded return intervals.
It annualises using observed intervals and elapsed calendar time, because strict
common valuations omit days when an active investment or NAV is unavailable.
The original covariance estimates use the historical baseline's 252-day convention;
publication timing, non-synchronous closes and irregular valuations limit precision.

The next monthly and quarterly review dates are provisional until actual entry.
A boundary's closing valuation may be available only on a later update, particularly
for the fund NAV. Reports flag stale endpoints and valuation gaps above seven days.

## Monthly update

From the project directory:

```bash
.venv-portfolio/bin/python paper_portfolio.py update --refresh-data
```

This explicitly downloads adjusted closing prices for active investments, FX and
the actual BlackRock NAV export, records new observations and generates a report.
The forward download window extends with the current date; it does not depend on
the historical script's fixed END setting. Each refresh retains a separate raw
source vintage under the trial's `data/snapshots/` folder. Failed downloads do not
replace the last successful snapshot or recorded outcomes.

No background task, notification or calendar connection has been installed. Run
the command monthly, and at quarterly review. You can run it more often without
changing the frozen portfolio choices.

To regenerate reports without downloading:

```bash
.venv-portfolio/bin/python paper_portfolio.py report
```

To process the most recent local snapshot without downloading:

```bash
.venv-portfolio/bin/python paper_portfolio.py update
```

Commands select the latest registered trial by default. To target one explicitly:

```bash
.venv-portfolio/bin/python paper_portfolio.py report --trial-dir paper_tests/forward-2026-10-30
```

Optional import of an external observed GBP price-index file:

```bash
.venv-portfolio/bin/python paper_portfolio.py update --prices-file /path/to/gbp_prices.csv
```

Use a Date column and one column per active investment, including IE000VYDX630
for the actual accumulating fund. Equity/ETF prices must include distributions,
FX must already be applied consistently, and the file must include the previous
recorded valuation to link new returns. Quote levels may have arbitrary constant
units per asset; changes must represent GBP total returns. No proxy can be passed
under the EUNH ticker as a substitute for the actual-fund column. Imported values
are user-provided data; the tool cannot independently certify a relabelled series.

## Preserving the forward test

`protocol.json` and `training_returns.csv` are hash-checked. Register refuses an
existing trial directory or an initial baseline more than seven days stale.
Use a new version after rebuilding a current baseline if you change weights,
model rules or forecasts:

```bash
.venv-portfolio/bin/python paper_portfolio.py init --trial-dir paper_tests/a-new-version --baseline-dir reports/research-latest/baseline/actual
```

The observation ledger records first-seen interval returns, not a mixture of
price levels from different adjustment vintages. A uniform historical price
rescaling therefore cannot create an artificial new return. Later source-return
revisions are logged in `source_revision_log.csv`; they do not overwrite recorded
outcomes. Late backfilled dates do not split existing intervals. Corrupt hashes
stop updates rather than silently recomputing history.

Report archives retain each generated report with summary, portfolio paths,
quarterly comparisons, allocation/rebalance events, status, and input/code hashes.
`reports/latest/` is a convenience copy. The CSV ledger and its hash sidecar are
separate writes; an interrupted commit can require recovery from preserved source
snapshots. Do not edit or delete trial records to alter performance.

Account trading access, actual commissions, FX spreads, taxes and liquidity are
not verified. These are hypothetical comparisons, not orders or executable advice.
A few forward quarters cannot establish performance across a full market cycle.

Run all tests:

```bash
.venv-portfolio/bin/python -m unittest test_price_cache test_fund_nav test_portfolio_validation test_robustness test_research_report test_paper_test
```
