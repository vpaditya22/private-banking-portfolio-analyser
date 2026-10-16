# Private Banking Client Portfolio Intelligence

A client-focused Streamlit dashboard added as a separate subproject alongside the existing portfolio analyser.

## Features
- Three fictional client profiles, objectives, risk profiles and liquidity targets
- Annualized return, volatility, Sharpe ratio and maximum drawdown
- Benchmark comparison
- Current versus target allocation and illustrative rebalancing calculations
- Asset-level risk metrics and return correlation matrix
- Uniform and asset-specific hypothetical stress scenarios
- Downloadable client review summary and CSV reports
- CSV upload for dated historical prices
- Synthetic demo data, so no API key is required

## Run locally

From the repository root, create and activate a virtual environment, then install dependencies:

Windows PowerShell:
    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install and run:
    python -m pip install -r client_portfolio_intelligence/requirements.txt
    streamlit run client_portfolio_intelligence/app.py

Run tests from the repository root:
    pytest -q client_portfolio_intelligence/tests

## CSV format

The CSV must include a date column followed by ticker columns matching the selected client's holdings. Example columns: date, SPY, AGG, GLD, VNQ, QQQ.

## Limitations

All sample clients and generated prices are fictional. The benchmark may not match a client's mandate. The risk indicator is educational, not a regulated suitability assessment. Rebalancing ignores taxes, fees, liquidity restrictions and market impact. Stress scenarios are not forecasts. Do not upload identifiable client data to an unapproved environment. This is not investment, tax or legal advice.
