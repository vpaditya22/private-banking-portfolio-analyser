import numpy as np
import pandas as pd

TRADING_DAYS = 252

def normalize_weights(weights):
    weights = pd.Series(weights, dtype=float).dropna()
    if weights.empty or not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Weights must contain finite numeric values.")
    if (weights < 0).any():
        raise ValueError("Negative weights are not supported.")
    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return weights / total

def build_portfolio_returns(prices, weights):
    weights = normalize_weights(weights)
    prices = prices.loc[:, weights.index].copy()
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Prices must be positive and contain no missing observations.")
    returns = prices.pct_change().dropna()
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Price data produced invalid returns.")
    return returns.mul(weights, axis=1).sum(axis=1)

def calculate_metrics(returns, risk_free_rate=0.04):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        raise ValueError("At least two valid return observations are required.")
    if (returns <= -1).any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain invalid values.")
    annual_return = float(np.prod(1 + returns) ** (TRADING_DAYS / len(returns)) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((annual_return - risk_free_rate) / volatility) if volatility > 0 else 0.0
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {"annual_return": annual_return, "volatility": volatility, "sharpe": sharpe, "max_drawdown": float(drawdown.min())}

def portfolio_metrics(prices, weights, risk_free_rate=0.04):
    return calculate_metrics(build_portfolio_returns(prices, weights), risk_free_rate)

def benchmark_metrics(price_series, risk_free_rate=0.04):
    prices = pd.Series(price_series).dropna()
    if (prices <= 0).any():
        raise ValueError("Benchmark prices must be positive.")
    return calculate_metrics(prices.pct_change().dropna(), risk_free_rate)

def asset_metrics(prices):
    return pd.DataFrame({ticker: calculate_metrics(prices[ticker].pct_change().dropna()) for ticker in prices.columns}).T

def stress_scenario(weights, shocks):
    weights = normalize_weights(weights)
    if np.isscalar(shocks):
        shocks = pd.Series(float(shocks), index=weights.index)
    else:
        shocks = pd.Series(shocks, dtype=float).reindex(weights.index).fillna(0)
    if not np.isfinite(shocks.to_numpy()).all() or (shocks <= -1).any():
        raise ValueError("Shock values must be finite and greater than -100%.")
    impacts = weights * shocks
    return {"portfolio_impact": float(impacts.sum()), "asset_impacts": impacts}

def rebalancing_table(current_weights, target_weights, portfolio_value):
    current = normalize_weights(current_weights)
    target = normalize_weights(target_weights).reindex(current.index).fillna(0)
    target = target / target.sum() if target.sum() > 0 else current.copy()
    return pd.DataFrame({
        "current_weight": current, "target_weight": target,
        "current_value": current * portfolio_value,
        "target_value": target * portfolio_value,
        "trade_value": (target - current) * portfolio_value,
    }).rename_axis("ticker")

def risk_label(volatility, profile):
    thresholds = {"Conservative": .10, "Moderate": .18, "Growth": .28}
    threshold = thresholds.get(profile, .18)
    if volatility <= threshold * .75:
        return "Lower"
    if volatility <= threshold * 1.25:
        return "Moderate"
    return "Elevated"
