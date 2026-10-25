# Periodic Return Validation

The portfolio calculations use simple periodic returns. For a return observation to be valid, it must be a finite number strictly greater than -100%.

## Why the boundary matters

A simple return of -100% means the asset value has fallen to zero. Returns below -100% are not valid simple returns. Infinite values also cannot be used in a portfolio calculation. Allowing these values into cumulative growth or drawdown calculations can produce misleading metrics or undefined results.

## Validation behavior

- Portfolio-return aggregation requires every expected asset-class observation to be present and finite.
- Asset-class returns at or below -100% are rejected before weighted aggregation.
- Performance metrics reject non-finite periodic returns and returns at or below -100%.
- The standalone maximum-drawdown and performance-metrics calculations reject missing observations rather than silently filling them with zero or dropping them. Resolve incomplete data explicitly before calculating metrics.

## Quick verification

After installing the project dependencies, run:

```bash
pytest -q
python -m compileall -q app.py src tests
```

These checks validate the software's handling of synthetic inputs. They do not validate an investment strategy or make the generated return series representative of actual markets.
