# EDA Review Checklist

Use this checklist before sharing charts or summary tables produced from the project's synthetic monthly returns.

## Dataset integrity

- [ ] Record the number of months and the random seed used to generate the sample.
- [ ] Confirm the four expected asset-class columns are present and the index is in chronological order.
- [ ] Check missing values and non-finite observations before computing summary statistics.
- [ ] Confirm all periodic returns are greater than -100%; a loss beyond that threshold is not a valid simple return.
- [ ] Record the exact allocation weights and confirm they sum to 100%.

## Calculation checks

- [ ] Recompute at least one weighted portfolio-return row manually to confirm the weight mapping.
- [ ] Verify cumulative return from the product of one plus each periodic portfolio return.
- [ ] Inspect the wealth path and peak-to-trough drawdown calculation on a small known sequence.
- [ ] Keep annualization conventions consistent when comparing return and volatility metrics.
- [ ] Treat the dashboard's Sharpe-style ratio as an illustrative metric, not a standardized performance statistic.

## Communication checks

- [ ] Label charts, tables, and exported files as synthetic and illustrative.
- [ ] State the seed, sample length, allocation, and any assumptions that materially affect results.
- [ ] Do not interpret sample correlations as calibrated real-market relationships.
- [ ] Avoid implying that generated results forecast returns or establish that an allocation is suitable for a client.

Passing this checklist improves reproducibility and reduces avoidable analysis mistakes; it does not validate an investment strategy or turn synthetic data into market evidence.
