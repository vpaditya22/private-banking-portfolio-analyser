import pandas as pd
import pytest

from client_portfolio_intelligence.src.analytics import (
    normalize_weights, stress_scenario, rebalancing_table, portfolio_metrics,
)
from client_portfolio_intelligence.src.sample_data import make_sample_prices, DEFAULT_HOLDINGS


def test_weights_normalize():
    assert normalize_weights({"A": 2, "B": 2}).sum() == pytest.approx(1.0)


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        normalize_weights({"A": 1, "B": -0.1})


def test_stress_scenario():
    result = stress_scenario(
        pd.Series({"A": 0.60, "B": 0.40}),
        pd.Series({"A": -0.10, "B": -0.20}),
    )
    assert result["portfolio_impact"] == pytest.approx(-0.14)


def test_rebalancing_trades_sum_to_zero():
    table = rebalancing_table({"A": 0.80, "B": 0.20}, {"A": 0.50, "B": 0.50}, 10000)
    assert table["trade_value"].sum() == pytest.approx(0.0)


def test_portfolio_metrics_are_finite():
    prices = make_sample_prices(years=1)
    metrics = portfolio_metrics(prices, DEFAULT_HOLDINGS["Maya Sharma"])
    assert all(pd.notna(value) for value in metrics.values())
