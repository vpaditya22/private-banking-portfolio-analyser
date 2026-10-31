import pandas as pd
import pytest

from src.portfolio import (
    ASSET_CLASSES,
    apply_one_period_shock,
    max_drawdown,
    performance_metrics,
    portfolio_returns,
    portfolio_contributions,
    portfolio_wealth_path,
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


@pytest.mark.parametrize(
    ("periodic_returns", "expected"),
    [
        ([-0.10, 0.05], -0.10),
        ([-0.10, 0.20], -0.10),
        ([0.10, -0.20], -0.20),
        ([0.05, 0.03], 0.0),
    ],
)
def test_max_drawdown_uses_initial_capital_as_high_water_mark(periodic_returns, expected):
    assert max_drawdown(pd.Series(periodic_returns)) == pytest.approx(expected)


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


@pytest.mark.parametrize("risk_free_rate", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_metrics_reject_invalid_annual_risk_free_rate(risk_free_rate):
    with pytest.raises(ValueError, match="Annual risk-free rate"):
        performance_metrics(pd.Series([0.01, -0.02, 0.015, 0.0]), risk_free_rate)


@pytest.mark.parametrize("bad_returns", [
    [0.01, float("nan"), 0.02],
    [0.01, float("inf"), 0.02],
])
def test_max_drawdown_rejects_missing_or_non_finite_returns(bad_returns):
    with pytest.raises(ValueError, match="Periodic returns must be finite"):
        max_drawdown(pd.Series(bad_returns))


@pytest.mark.parametrize("bad_returns", [
    [0.01, float("nan"), 0.02],
    [0.01, float("inf"), 0.02],
])
def test_metrics_reject_missing_or_non_finite_returns(bad_returns):
    with pytest.raises(ValueError, match="Periodic returns must be finite"):
        performance_metrics(pd.Series(bad_returns))


def test_metrics_compound_cumulative_return():
    metrics = performance_metrics(pd.Series([0.10, 0.10]), annual_risk_free_rate=0.0)
    assert metrics["Cumulative return"] == pytest.approx(0.21)


def test_portfolio_contributions_sum_to_portfolio_returns():
    df = pd.DataFrame({
        "Global equities": [0.10, -0.02],
        "High-quality bonds": [0.02, 0.01],
        "Gold": [0.04, 0.03],
        "Cash": [0.001, 0.001],
    })
    contributions = portfolio_contributions(df, VALID)
    expected = portfolio_returns(df, VALID)
    pd.testing.assert_series_equal(contributions.sum(axis=1).rename("Portfolio return"), expected)


def test_portfolio_contributions_reject_missing_asset_column():
    df = pd.DataFrame({"Global equities": [0.01]})
    with pytest.raises(ValueError, match="Missing asset return columns"):
        portfolio_contributions(df, VALID)


def test_portfolio_contributions_preserve_period_index_and_asset_order():
    index = pd.to_datetime(["2025-01-31", "2025-02-28"])
    df = pd.DataFrame({a: [0.01, 0.02] for a in ASSET_CLASSES}, index=index)
    contributions = portfolio_contributions(df, VALID)
    assert contributions.index.equals(index)
    assert list(contributions.columns) == ASSET_CLASSES


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_portfolio_contributions_reject_non_finite_returns(bad_value):
    df = pd.DataFrame({a: [0.01, 0.02] for a in ASSET_CLASSES})
    df.loc[1, "Gold"] = bad_value
    with pytest.raises(ValueError, match="Periodic returns must be finite"):
        portfolio_contributions(df, VALID)



def test_portfolio_wealth_path_includes_starting_capital_and_compounds_returns():
    returns = pd.Series(
        [0.10, -0.20],
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )
    path = portfolio_wealth_path(returns, starting_value=1000)

    assert list(path) == pytest.approx([1000, 1100, 880])
    assert path.index.equals(pd.RangeIndex(3, name="Period"))
    assert path.name == "Portfolio value"


def test_portfolio_wealth_path_rejects_empty_returns():
    with pytest.raises(ValueError, match="At least one return observation"):
        portfolio_wealth_path(pd.Series(dtype=float))


@pytest.mark.parametrize("starting_value", [0, -100, float("nan"), float("inf"), float("-inf")])
def test_portfolio_wealth_path_rejects_invalid_starting_value(starting_value):
    with pytest.raises(ValueError, match="Starting portfolio value"):
        portfolio_wealth_path(pd.Series([0.01, 0.02]), starting_value)


@pytest.mark.parametrize("bad_return", [float("nan"), float("inf"), -1.0])
def test_portfolio_wealth_path_rejects_invalid_returns(bad_return):
    with pytest.raises(ValueError):
        portfolio_wealth_path(pd.Series([0.01, bad_return]))
