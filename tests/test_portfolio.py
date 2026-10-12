import pandas as pd
import pytest

from src.portfolio import (
    ASSET_CLASSES,
    apply_one_period_shock,
    max_drawdown,
    performance_metrics,
    portfolio_returns,
    synthetic_monthly_returns,
    validate_weights,
)

VALID = {
    "Global equities": 0.50,
    "High-quality bonds": 0.30,
    "Gold": 0.10,
    "Cash": 0.10,
}


def test_valid_weights():
    validate_weights(VALID)


def test_weights_must_sum_to_one():
    bad = dict(VALID)
    bad["Cash"] = 0.05
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_negative_weights_rejected():
    bad = dict(VALID)
    bad["Gold"] = -0.1
    bad["Global equities"] = 0.7
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_weights_reject_nan():
    bad = dict(VALID)
    bad["Gold"] = float("nan")
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_weights_require_exact_asset_classes():
    with pytest.raises(ValueError):
        validate_weights({"Global equities": 1.0})


def test_synthetic_data_is_reproducible():
    pd.testing.assert_frame_equal(synthetic_monthly_returns(), synthetic_monthly_returns())


def test_synthetic_data_has_requested_length():
    assert len(synthetic_monthly_returns(months=24)) == 24


def test_synthetic_data_requires_at_least_two_months():
    with pytest.raises(ValueError):
        synthetic_monthly_returns(months=1)


def test_portfolio_returns_are_weighted_sum():
    df = pd.DataFrame({a: [0.10, 0.0] for a in ASSET_CLASSES})
    result = portfolio_returns(df, VALID)
    assert list(result.round(8)) == [0.10, 0.0]


def test_missing_asset_return_column_rejected():
    df = pd.DataFrame({"Global equities": [0.01]})
    with pytest.raises(ValueError):
        portfolio_returns(df, VALID)


def test_max_drawdown_negative_after_loss():
    returns = pd.Series([0.10, -0.20, 0.05])
    assert max_drawdown(returns) < 0


def test_stress_shock_weighted_sum():
    shocks = {a: 0.0 for a in ASSET_CLASSES}
    shocks["Global equities"] = -0.20
    assert apply_one_period_shock(VALID, shocks) == pytest.approx(-0.10)


def test_stress_shock_rejects_loss_below_100_percent():
    shocks = {a: 0.0 for a in ASSET_CLASSES}
    shocks["Gold"] = -1.01
    with pytest.raises(ValueError):
        apply_one_period_shock(VALID, shocks)


def test_metrics_return_expected_keys():
    metrics = performance_metrics(pd.Series([0.01, -0.02, 0.015, 0.0]))
    assert {
        "Annualized return",
        "Annualized volatility",
        "Maximum drawdown",
        "Sharpe-style ratio",
        "Cumulative return",
    } <= set(metrics)


def test_metrics_reject_single_observation():
    with pytest.raises(ValueError):
        performance_metrics(pd.Series([0.01]))
