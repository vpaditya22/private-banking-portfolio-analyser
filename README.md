# Private Banking Portfolio Risk & Allocation Analyser

A Python and Streamlit portfolio-analysis demo focused on portfolio allocation, risk metrics, and hypothetical stress testing.

## Features
- Interactive allocation sliders for global equities, high-quality bonds, gold, and cash
- Annualized return, annualized volatility, maximum drawdown, and Sharpe-style ratio
- Illustrative portfolio path and allocation visualizations
- Comparison of example allocation profiles
- User-defined one-period stress scenarios
- CSV export of allocation comparisons
- Unit tests for key calculations

## Important data note
This project uses reproducible **synthetic monthly returns**, not actual market history or client data. Results are educational examples only and are not investment advice or a recommendation.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies, run tests, and launch the app:

```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## Metrics and limitations
Annualized return compounds the generated monthly returns. Annualized volatility uses monthly standard deviation multiplied by the square root of 12. Maximum drawdown measures the largest peak-to-trough decline. The Sharpe-style ratio uses an assumed risk-free rate. Stress testing uses a weighted sum of user-entered shocks and does not model correlations, fees, taxes, currency effects, or client-specific liabilities.

All allocations and scenarios are hypothetical. This tool is not financial, legal, or tax advice.
