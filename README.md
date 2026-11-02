# AR-Y Portfolio

Python research tools for comparing the current Umushroom portfolio with an
expanded investment universe, estimating covariance, testing diversification
changes, and generating HTML and Markdown reports.

The actual BlackRock Euro Government Enhanced Index Fund D EUR Acc holding
(ISIN **IE000VYDX630**) is kept distinct from the **EUNH.DE** calculation proxy.
Historical rankings are retrospective research, not validated forecasts of
future returns. These tools do not execute transactions.

## Setup

The existing environment uses Python 3.9.6. `requirements.txt` records its direct
dependency versions; transitive dependencies are not locked.

```bash
python3 -m venv .venv-portfolio
source .venv-portfolio/bin/activate
python -m pip install -r requirements.txt
```

## Run the analysis

```bash
python AR-Y_portfolio.py --no-show
python run_research.py --rebuild-baseline
```

The research runner writes the readable report to:

`reports/research-latest/SUMMARY_REPORT.html`

For the main script's two Markowitz plots, weight comparison, addition ranking,
existing-position review and concise `BASELINE_REPORT.md`, see `reports/latest/`.
The actual-portfolio optimisations retain the configured GBP cash weight;
the separate classical chart excludes cash. Details are in [DATA_README.md](DATA_README.md).

It also writes a Markdown version and supporting CSVs. Asset prices, FX rates
and the actual fund's NAV are downloaded when missing and cached locally.
Existing cached observations are refreshed only when explicitly requested.
Configured historical date windows remain in the scripts: publishing this
repository does not advance those windows or update any research results.

The entry points `tactical_screener.py`, `extend_tactical_returns.py` and
`optimise_tactical_portfolio.py` contain additional tactical research tools.

## Documentation

- [Data sources, caches and actual-fund handling](DATA_README.md)
- [Research reports, sensitivity checks and limitations](RESEARCH_README.md)
- [Registered forward paper test and update instructions](PAPER_TEST_README.md)

## Tests

From the project directory with the virtual environment activated:

```bash
MPLBACKEND=Agg python -m unittest discover -p 'test_*.py'
```

## Repository contents

Source code, tests, documentation, source-review request metadata and the
registered paper test's frozen protocol, selection tables and training returns
are versioned. Do not edit a registered protocol to change its evaluation rules.

Local environments, price/NAV caches, downloaded source pages, generated reports
and legacy root-level charts and CSV outputs are excluded by `.gitignore`.
They remain on the original computer. A fresh clone must obtain its own data
and regenerate reports; current provider data may differ from the original
research vintage. The frozen paper-test definition alone does not constitute
observed forward performance.
