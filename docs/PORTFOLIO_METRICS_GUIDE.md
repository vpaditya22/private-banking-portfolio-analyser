# Portfolio Metrics: Interpretation Guide

This guide explains how to read the dashboard's illustrative metrics. The application uses synthetic returns, so the results demonstrate calculations rather than describe real markets or predict future performance.

## Annualized return

Annualized return expresses the compounded change in the illustrative portfolio over a one-year equivalent. Compounding means the result can differ from simply multiplying an average monthly return by twelve. It is sensitive to the sequence and magnitude of periodic returns.

## Annualized volatility

Annualized volatility scales the sample standard deviation of monthly returns by the square root of 12. It describes variability, not the direction of returns. Volatility does not capture every kind of risk, including liquidity, credit, model, or operational risk.

## Maximum drawdown

Maximum drawdown is the largest percentage decline from a prior wealth-index peak to a subsequent trough in the selected return path. It is path-dependent: two portfolios with similar annualized returns can experience different drawdowns.

## Sharpe-style ratio

The dashboard's simplified ratio compares annualized mean excess monthly return with monthly volatility. Because the numerator and denominator may use different annualization conventions, treat this as an illustrative comparison only—not a standardized institutional Sharpe ratio. A ratio based on synthetic returns should not be compared directly with reported fund performance.

## Stress scenario

The stress result is the weighted sum of user-entered asset-class shocks for one hypothetical period. It assumes the selected weights remain fixed and does not model correlations, changing prices within the period, rebalancing, fees, taxes, currency effects, liquidity, or client liabilities. It is not a forecast or a worst-case estimate.

## Responsible interpretation checklist

- Confirm that the allocation weights sum to 100% and cover every required asset class.
- Compare metrics over the same return horizon and under the same assumptions.
- Read return, volatility, and drawdown together instead of relying on one number.
- Treat synthetic results as a software demonstration, not market evidence.
- Do not use this application to make client-specific investment recommendations.
