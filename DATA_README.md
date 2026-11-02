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
