"""Core calculations for the illustrative portfolio analyser."""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSET_CLASSES = ["Global equities", "High-quality bonds", "Gold", "Cash"]

DEFAULT_WEIGHTS = {
    "Global equities": 0.50,
    "High-quality bonds": 0.30,
    "Gold": 0.10,
    "Cash": 0.10,
}

SAMPLE_ALLOCATIONS = {
    "Balanced illustrative": DEFAULT_WEIGHTS,
    "Growth illustrative": {
        "Global equities": 0.65,
        "High-quality bonds": 0.20,
        "Gold": 0.10,
        "Cash": 0.05,
    },
    "Capital-preservation illustrative": {
        "Global equities": 0.25,
        "High-quality bonds": 0.40,
        "Gold": 0.10,
        "Cash": 0.25,
    },
}


def validate_weights(weights: dict[str, float], tolerance: float = 1e-6) -> None:
    """Raise ValueError unless weights cover the asset classes and sum to 1."""
    if set(weights) != set(ASSET_CLASSES):
        raise ValueError(f"Weights must include exactly: {', '.join(ASSET_CLASSES)}")
    if any((not np.isfinite(float(v))) or float(v) < 0 for v in weights.values()):
        raise ValueError("Weights must be finite and non-negative.")
    if abs(sum(float(v) for v in weights.values()) - 1.0) > tolerance:
        raise ValueError("Portfolio weights must sum to 100%.")


def allocation_concentration_metrics(weights: dict[str, float]) -> dict[str, float | str]:
    """Summarize weight concentration without implying asset-class correlation.

    The Herfindahl-Hirschman index is the sum of squared portfolio weights.
    Its reciprocal is the effective number of equally weighted asset classes
    that would produce the same concentration.
    """
    validate_weights(weights)
    ordered_weights = np.array([float(weights[asset]) for asset in ASSET_CLASSES], dtype=float)
    concentration_index = float(np.square(ordered_weights).sum())
    largest_index = int(np.argmax(ordered_weights))
    return {
        "Herfindahl-Hirschman index": concentration_index,
        "Effective number of asset classes": 1.0 / concentration_index,
        "Largest asset class": ASSET_CLASSES[largest_index],
        "Largest allocation": float(ordered_weights[largest_index]),
    }



def asset_return_correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlations for the required asset-class return series.

    All four asset classes must have at least two finite observations and
    non-zero sample variance. Correlation is descriptive, not a forecast.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("Asset returns must be provided as a pandas DataFrame.")

    duplicate_assets = [
        asset for asset in ASSET_CLASSES
        if int((asset_returns.columns == asset).sum()) > 1
    ]
    if duplicate_assets:
        raise ValueError(
            f"Duplicate asset return columns: {', '.join(duplicate_assets)}"
        )

    missing = set(ASSET_CLASSES) - set(asset_returns.columns)
    if missing:
        raise ValueError(f"Missing asset return columns: {', '.join(sorted(missing))}")

    clean = _validate_periodic_returns(asset_returns[ASSET_CLASSES])
    if len(clean) < 2:
        raise ValueError("At least two return observations are required for correlation.")

    zero_variance = clean.columns[clean.std(ddof=1).eq(0)].tolist()
    if zero_variance:
        raise ValueError(
            "Correlation is undefined for zero-variance asset classes: "
            f"{', '.join(zero_variance)}"
        )

    correlation = clean.corr(method="pearson").reindex(
        index=ASSET_CLASSES,
        columns=ASSET_CLASSES,
    )
    if not np.isfinite(correlation.to_numpy(dtype=float)).all():
        raise ValueError("Correlation matrix contains undefined values.")
    return correlation

def _validate_periodic_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reject invalid simple returns before they can distort portfolio metrics."""
    numeric = returns.astype(float)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Periodic returns must be finite; resolve missing or infinite observations first.")
    if (values <= -1).any():
        raise ValueError("Periodic returns must be greater than -100%.")
    return numeric


def synthetic_monthly_returns(months: int = 60, seed: int = 2026) -> pd.DataFrame:
    """Return deterministic illustrative monthly returns; not real market history."""
    if months < 2:
        raise ValueError("At least two months are required.")
    rng = np.random.default_rng(seed)
    params = {
        "Global equities": (0.0070, 0.045),
        "High-quality bonds": (0.0025, 0.015),
        "Gold": (0.0030, 0.035),
        "Cash": (0.0015, 0.0010),
    }
    data = {
        asset: rng.normal(mean, vol, months)
        for asset, (mean, vol) in params.items()
    }
    return pd.DataFrame(data, index=pd.date_range("2021-01-31", periods=months, freq="ME"))


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    validate_weights(weights)
    missing = set(ASSET_CLASSES) - set(asset_returns.columns)
    if missing:
        raise ValueError(f"Missing asset return columns: {', '.join(sorted(missing))}")
    clean_asset_returns = _validate_periodic_returns(asset_returns[ASSET_CLASSES])
    w = pd.Series(weights, dtype=float).reindex(clean_asset_returns.columns)
    return clean_asset_returns.mul(w, axis=1).sum(axis=1).rename("Portfolio return")


def _max_drawdown_from_values(values: np.ndarray) -> float:
    """Calculate drawdown from a NumPy array without compounding overflow."""
    log_wealth = np.cumsum(np.log1p(values))
    high_water_marks = np.maximum.accumulate(np.concatenate(([0.0], log_wealth)))[1:]
    drawdowns = np.expm1(log_wealth - high_water_marks)
    return float(drawdowns.min())


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, including losses from initial capital."""
    clean = _validate_periodic_returns(returns)
    if clean.empty:
        raise ValueError("At least one return observation is required.")
    return _max_drawdown_from_values(clean.to_numpy(dtype=float))


def rolling_risk_metrics(returns: pd.Series, window: int = 12) -> pd.DataFrame:
    """Calculate trailing annualized volatility and maximum drawdown.

    The first window - 1 rows contain NaN because a complete lookback
    period is not yet available. Drawdown includes the initial capital baseline
    at the start of each rolling window.
    """
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 2
    ):
        raise ValueError("Rolling window must be an integer of at least two periods.")

    clean = _validate_periodic_returns(returns)
    if clean.empty:
        raise ValueError("At least one return observation is required.")

    rolling_volatility = clean.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(12)
    rolling_drawdown = clean.rolling(window=window, min_periods=window).apply(
        _max_drawdown_from_values,
        raw=True,
    )
    return pd.DataFrame(
        {
            "Rolling annualized volatility": rolling_volatility,
            "Rolling maximum drawdown": rolling_drawdown,
        },
        index=clean.index,
    )


def performance_metrics(returns: pd.Series, annual_risk_free_rate: float = 0.03) -> dict[str, float]:
    """Compute historical-style summary metrics from a periodic return series."""
    clean = _validate_periodic_returns(returns)
    if len(clean) < 2:
        raise ValueError("At least two return observations are required.")
    if not np.isfinite(float(annual_risk_free_rate)) or annual_risk_free_rate <= -1:
        raise ValueError("Annual risk-free rate must be finite and greater than -100%.")
    periods_per_year = 12
    total_growth = float((1 + clean).prod())
    annualized_return = total_growth ** (periods_per_year / len(clean)) - 1
    annualized_volatility = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    monthly_rf = (1 + annual_risk_free_rate) ** (1 / periods_per_year) - 1
    excess = clean - monthly_rf
    sharpe = float((excess.mean() / clean.std(ddof=1)) * np.sqrt(periods_per_year)) if clean.std(ddof=1) else 0.0
    return {
        "Annualized return": float(annualized_return),
        "Annualized volatility": annualized_volatility,
        "Maximum drawdown": max_drawdown(clean),
        "Sharpe-style ratio": sharpe,
        "Cumulative return": total_growth - 1,
    }


def apply_one_period_shock(weights: dict[str, float], shocks: dict[str, float]) -> float:
    """Compute illustrative one-period portfolio impact from asset-class shocks."""
    validate_weights(weights)
    if set(shocks) != set(ASSET_CLASSES):
        raise ValueError(f"Shocks must include exactly: {', '.join(ASSET_CLASSES)}")
    if any(not np.isfinite(float(v)) or float(v) < -1 for v in shocks.values()):
        raise ValueError("Shock returns must be finite and cannot be below -100%.")
    return float(sum(float(weights[a]) * float(shocks[a]) for a in ASSET_CLASSES))


def allocation_summary(weights: dict[str, float], asset_returns: pd.DataFrame, annual_rf: float = 0.03) -> dict[str, float]:
    return performance_metrics(portfolio_returns(asset_returns, weights), annual_rf)

def portfolio_contributions(asset_returns: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Return each asset class's weighted periodic contribution to portfolio return.

    The contribution columns sum to the portfolio return for each period. Values
    are decimal returns (for example, 0.01 means one percentage point).
    """
    validate_weights(weights)
    missing = set(ASSET_CLASSES) - set(asset_returns.columns)
    if missing:
        raise ValueError(f"Missing asset return columns: {', '.join(sorted(missing))}")
    clean_asset_returns = _validate_periodic_returns(asset_returns[ASSET_CLASSES])
    weight_series = pd.Series(weights, dtype=float).reindex(ASSET_CLASSES)
    contributions = clean_asset_returns.mul(weight_series, axis=1)
    contributions.columns.name = "Asset class"
    return contributions

def portfolio_wealth_path(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Return starting capital followed by each period's compounded portfolio value.

    The first element is the initial value before any return is applied. Subsequent
    elements are end-of-period values, so the output always has one more point than
    the input return series.
    """
    clean = _validate_periodic_returns(returns)
    if clean.empty:
        raise ValueError("At least one return observation is required.")

    try:
        initial_value = float(starting_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Starting portfolio value must be finite and greater than zero.") from exc
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Starting portfolio value must be finite and greater than zero.")

    # Finite inputs can still overflow when compounded or scaled by starting capital.
    with np.errstate(over="ignore", invalid="ignore"):
        period_growth = (1.0 + clean).cumprod().to_numpy(dtype=float)
        values = initial_value * np.concatenate(([1.0], period_growth))
    if not np.isfinite(values).all():
        raise ValueError("Portfolio wealth path exceeds the numeric range; reduce returns or starting value.")

    return pd.Series(
        values,
        index=pd.RangeIndex(len(values), name="Period"),
        name="Portfolio value",
    )
