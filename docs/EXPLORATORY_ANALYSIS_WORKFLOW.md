# Reproducible Return Exploration Workflow

Use this workflow when you want to inspect the synthetic return series outside the Streamlit dashboard, for example in a notebook or as part of an MSDSM exploratory-data-analysis exercise.

## 1. Generate a repeatable sample

Run from the repository root after installing `requirements.txt`:

```python
from src.portfolio import DEFAULT_WEIGHTS, portfolio_returns, synthetic_monthly_returns

asset_returns = synthetic_monthly_returns(months=60, seed=2026)
portfolio = portfolio_returns(asset_returns, DEFAULT_WEIGHTS)
```

The fixed seed makes the same synthetic sample reproducible. Change the seed deliberately when exploring how results vary across generated samples; do not describe these samples as actual market history.

## 2. Inspect data quality and summary statistics

```python
print(asset_returns.shape)
print(asset_returns.isna().sum())
print(asset_returns.describe().T)
```

Check that the number of observations matches the chosen horizon, the expected asset-class columns are present, and missing values are understood before calculating metrics.

## 3. Explore relationships

```python
correlation_matrix = asset_returns.corr()
print(correlation_matrix.round(3))
```

The correlation matrix describes relationships within this generated sample only. The synthetic generator does not model a calibrated joint market process, so these correlations should not be used as estimates of real-world diversification benefits.

## 4. Export a working dataset

```python
export = asset_returns.copy()
export["Portfolio return"] = portfolio
export.to_csv("synthetic_monthly_returns.csv", index_label="Month")
```

Keep the month index in the export so the return sequence can be reconstructed and drawdowns can be calculated in the correct order.

## 5. Record the assumptions

For any chart, table, or report, record the seed, number of months, allocation weights, and metric definitions. Label all outputs as synthetic and illustrative. Do not combine synthetic observations with real historical data without clearly distinguishing their sources.

## Interpretation guardrails

- A generated dataset is useful for testing code and practicing EDA, not validating an investment strategy.
- Summary statistics can change with the seed and sample length.
- Correlation does not imply stable diversification in real markets.
- Compare portfolios only when the same sample and metric conventions are used.
- Do not use this exercise to make client-specific investment recommendations.
