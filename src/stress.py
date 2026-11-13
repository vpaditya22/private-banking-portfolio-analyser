"""Explain one-period portfolio shocks at the asset-class level.

These are hypothetical, static-weight scenarios, not forecasts.
"""
from __future__ import annotations

from math import isfinite

ASSET_CLASSES = ("Global equities", "High-quality bonds", "Gold", "Cash")


def stress_attribution(weights: dict[str, float], shocks: dict[str, float], starting_value: float) -> list[dict[str, float | str]]:
    """Return asset-level percentage-point and currency contributions.

    Contributions sum to the weighted one-period portfolio shock. Each monetary
    impact is based on the *initial* portfolio value (no rebalancing or fees).
    """
    if set(weights) != set(ASSET_CLASSES) or set(shocks) != set(ASSET_CLASSES):
        raise ValueError("Weights and shocks must contain exactly the four asset classes.")
    try:
        capital = float(starting_value)
        w = {asset: float(weights[asset]) for asset in ASSET_CLASSES}
        s = {asset: float(shocks[asset]) for asset in ASSET_CLASSES}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Inputs must be numeric.") from exc
    if not isfinite(capital) or capital <= 0:
        raise ValueError("Starting value must be finite and positive.")
    if any(not isfinite(x) or x < 0 for x in w.values()) or not isfinite(sum(w.values())) or abs(sum(w.values()) - 1) > 1e-6:
        raise ValueError("Weights must be finite, non-negative, and sum to 100%.")
    if any(not isfinite(x) or x < -1 for x in s.values()):
        raise ValueError("Shocks must be finite and cannot represent losses below 100%.")
    rows = []
    for asset in ASSET_CLASSES:
        contribution = w[asset] * s[asset]
        currency_impact = capital * contribution
        if not isfinite(contribution) or not isfinite(currency_impact):
            raise ValueError("Scenario impact exceeds the numeric range.")
        rows.append({
            "Asset class": asset,
            "Portfolio weight": w[asset],
            "Asset shock": s[asset],
            "Portfolio return contribution": contribution,
            "Value impact": currency_impact,
        })
    if not isfinite(sum(row["Value impact"] for row in rows)):
        raise ValueError("Total scenario impact exceeds the numeric range.")
    return rows
