import numpy as np
import pandas as pd

CLIENTS = {
    "Maya Sharma": {"segment": "High-net-worth", "risk_profile": "Moderate", "horizon": "7–10 years", "liquidity_target": 0.15, "objective": "Grow long-term wealth while maintaining diversification and a meaningful liquidity reserve."},
    "Arjun Mehta": {"segment": "Emerging affluent", "risk_profile": "Conservative", "horizon": "3–5 years", "liquidity_target": 0.25, "objective": "Preserve capital, pursue measured growth, and retain flexibility for planned expenses."},
    "Neha Kapoor": {"segment": "High-net-worth", "risk_profile": "Growth", "horizon": "10+ years", "liquidity_target": 0.10, "objective": "Pursue long-term capital appreciation with diversified exposure and periodic risk reviews."},
}
DEFAULT_HOLDINGS = {
    "Maya Sharma": pd.Series({"SPY": .35, "AGG": .30, "GLD": .10, "VNQ": .10, "QQQ": .15}),
    "Arjun Mehta": pd.Series({"SPY": .20, "AGG": .50, "GLD": .15, "VNQ": .05, "QQQ": .10}),
    "Neha Kapoor": pd.Series({"SPY": .40, "AGG": .15, "GLD": .05, "VNQ": .10, "QQQ": .30}),
}
TARGET_ALLOCATIONS = {name: weights.copy() for name, weights in DEFAULT_HOLDINGS.items()}

def make_sample_prices(years=3, seed=42):
    """Generate reproducible synthetic daily prices, not actual market data."""
    observations = 252 * years
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=observations)
    rng = np.random.default_rng(seed)
    parameters = {
        "SPY": (.00035, .0105), "AGG": (.00010, .0030),
        "GLD": (.00022, .0080), "VNQ": (.00025, .0120),
        "QQQ": (.00042, .0130),
    }
    data = {}
    for ticker, (drift, volatility) in parameters.items():
        data[ticker] = 100 * np.cumprod(1 + rng.normal(drift, volatility, observations))
    return pd.DataFrame(data, index=dates).rename_axis("date")
