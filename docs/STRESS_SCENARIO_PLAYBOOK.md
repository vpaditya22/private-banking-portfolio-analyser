# Stress Scenario Playbook

This guide provides a repeatable way to explore the dashboard's one-period stress-test feature. All shocks below are hypothetical teaching examples, not forecasts or investment recommendations.

## How to run a scenario

1. Start the Streamlit app and choose an allocation that totals 100%.
2. Record the starting portfolio value and allocation weights.
3. Enter one shock per asset class in the stress-test section.
4. Record the displayed portfolio impact and illustrative post-shock value.
5. Change one assumption at a time to compare scenarios, keeping the starting value and allocation fixed.

The app calculates the portfolio impact as the sum of each asset weight multiplied by its entered shock. For example, a portfolio with 60% equities, 30% bonds, 5% gold, and 5% cash under shocks of -20%, +3%, +8%, and 0% has an illustrative impact of:

`(0.60 × -20%) + (0.30 × 3%) + (0.05 × 8%) + (0.05 × 0%) = -10.1%`

For a $1,000,000 starting value, that corresponds to a hypothetical $101,000 decrease and a post-shock value of $899,000.

## Suggested educational scenarios

| Scenario | Global equities | High-quality bonds | Gold | Cash |
|---|---:|---:|---:|---:|
| Equity sell-off | -20% | +3% | +8% | 0% |
| Inflation surprise | -10% | -8% | +12% | 0% |
| Broad risk-off shock | -15% | +1% | +5% | 0% |
| Positive risk-on shock | +12% | -2% | -4% | 0% |

These scenarios are illustrative and are not calibrated to a specific historical event. Actual assets may behave differently, and relationships between assets can change sharply during market stress.

## Interpretation checklist

- Confirm the allocation weights sum to 100% before interpreting results.
- Distinguish percentage impact from dollar impact; dollar impact depends on starting portfolio value.
- Compare the same scenario across different allocations to understand allocation sensitivity.
- Do not infer probabilities from the scenario table.
- Treat this as a single-period arithmetic estimate, not a forecast of a multi-period portfolio path.

## Important limitations

This simple weighted shock model does not model correlations, liquidity, market impact, rebalancing, fees, taxes, currency effects, cash flows, or client-specific objectives and liabilities. It uses no live market data or client records. The dashboard is an educational project and must not be used as a standalone basis for investment decisions.
