# Private Banking Portfolio Risk & Allocation Analyser

[![Python checks](https://github.com/vpaditya22/private-banking-portfolio-analyser/actions/workflows/python-checks.yml/badge.svg)](https://github.com/vpaditya22/private-banking-portfolio-analyser/actions/workflows/python-checks.yml)

A small **Python + Streamlit** portfolio-analysis application exploring how asset allocation affects illustrative portfolio risk and return metrics. Built as a learning project relevant to portfolio analytics and private-banking conversations.

## What it demonstrates

- Portfolio-weight validation: allocations must cover the four asset classes and sum to 100%.
- Reproducible synthetic monthly returns for global equities, high-quality bonds, gold, and cash.
- Annualized return, annualized volatility, maximum drawdown, and a Sharpe-style ratio.
- Interactive allocation and illustrative wealth-path charts.
- Comparison of three example allocations against a user-defined allocation.
- User-entered one-period stress scenarios and downloadable CSV comparison.
- Unit tests for core calculations and input validation.

## Preview

Run the app locally to explore the interactive dashboard. A live hosted demo is not configured yet.

## Run locally

Python 3.10+ is recommended.

```bash
git clone https://github.com/vpaditya22/private-banking-portfolio-analyser.git
cd private-banking-portfolio-analyser
python -m venv .venv
```

Activate the environment:

**Windows**
```powershell
.venv\\Scripts\\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies, run checks, and start the dashboard:

```bash
python -m pip install -r requirements.txt
pytest -q
python -m compileall -q app.py src tests
streamlit run app.py
```

## Methodology

- **Synthetic data:** generated using fixed random seeds so that each run is reproducible.
- **Annualized return:** compounds the periodic returns to an annual rate.
- **Annualized volatility:** sample standard deviation of monthly returns multiplied by the square root of 12.
- **Maximum drawdown:** largest peak-to-trough decline in the illustrative wealth index.
- **Sharpe-style ratio:** annualized mean excess monthly return divided by monthly volatility, using the risk-free rate entered in the app.
- **Stress test:** weighted sum of the selected asset-class shocks for one hypothetical period.

For practical guidance on interpreting these outputs and their limitations, see the [Portfolio Metrics Interpretation Guide](docs/PORTFOLIO_METRICS_GUIDE.md).

## Limitations and responsible use

This is an educational coding project, **not an investment tool**. It uses no live prices, actual historical performance, client records, or proprietary data. Synthetic outputs must not be presented as market evidence or used to recommend investments. The stress test is a simple weighted scenario and does not model correlations, rebalancing, fees, taxes, currency effects, liquidity, or client-specific liabilities. The Sharpe-style ratio is a simplified estimate and should not be interpreted as an institutional performance measure.

## Possible next steps

- Connect a documented public market-data source and add data-quality checks.
- Add correlation analysis and scenario comparisons using clearly identified historical periods.
- Add deployment, accessibility checks, and more edge-case tests.

## Disclaimer

All allocations, returns, and scenarios are hypothetical. This project is for education and demonstration only and does not constitute financial, legal, or tax advice.
