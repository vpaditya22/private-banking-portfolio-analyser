import pandas as pd
import pytest

from src.portfolio import (
    ASSET_CLASSES,
    allocation_concentration_metrics,
    asset_return_correlation_matrix,
    apply_one_period_shock,
    max_drawdown,
    performance_metrics,
    portfolio_returns,
    portfolio_contributions,
    portfolio_wealth_path,
    rolling_risk_metrics,
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


def test_allocation_concentration_metrics_for_equal_weights():
    equal_weights = {asset: 0.25 for asset in ASSET_CLASSES}

    metrics = allocation_concentration_metrics(equal_weights)

    assert metrics["Herfindahl-Hirschman index"] == pytest.approx(0.25)
    assert metrics["Effective number of asset classes"] == pytest.approx(4.0)
    assert metrics["Largest asset class"] == "Global equities"
    assert metrics["Largest allocation"] == pytest.approx(0.25)


def test_allocation_concentration_metrics_for_unequal_weights():
    metrics = allocation_concentration_metrics(VALID)

    assert metrics["Herfindahl-Hirschman index"] == pytest.approx(0.36)
    assert metrics["Effective number of asset classes"] == pytest.approx(1 / 0.36)
    assert metrics["Largest asset class"] == "Global equities"
    assert metrics["Largest allocation"] == pytest.approx(0.50)


def test_allocation_concentration_metrics_reject_invalid_weights():
    invalid_weights = dict(VALID)
    invalid_weights["Cash"] = 0.05

    with pytest.raises(ValueError, match="sum to 100%"):
        allocation_concentration_metrics(invalid_weights)



def test_asset_return_correlation_matrix_is_symmetric_and_uses_asset_order():
    returns = pd.DataFrame({
        "Cash": [0.015, 0.025, 0.020, 0.030],
        "Gold": [0.020, 0.040, 0.060, 0.080],
        "Global equities": [0.010, 0.020, 0.030, 0.040],
        "High-quality bonds": [0.040, 0.030, 0.020, 0.010],
    })

    correlation = asset_return_correlation_matrix(returns)

    assert list(correlation.index) == ASSET_CLASSES
    assert list(correlation.columns) == ASSET_CLASSES
    pd.testing.assert_frame_equal(correlation, correlation.T)
    assert correlation.to_numpy().diagonal() == pytest.approx([1.0] * len(ASSET_CLASSES))
    assert correlation.loc["Global equities", "High-quality bonds"] == pytest.approx(-1.0)
    assert correlation.loc["Global equities", "Gold"] == pytest.approx(1.0)


def test_asset_return_correlation_matrix_rejects_missing_columns():
    returns = pd.DataFrame({"Global equities": [0.01, 0.02]})

    with pytest.raises(ValueError, match="Missing asset return columns"):
        asset_return_correlation_matrix(returns)


def test_asset_return_correlation_matrix_rejects_duplicate_asset_columns():
    returns = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04, 0.05], [0.02, 0.03, 0.04, 0.05, 0.06]],
        columns=[
            "Global equities",
            "Global equities",
            "High-quality bonds",
            "Gold",
            "Cash",
        ],
    )

    with pytest.raises(ValueError, match="Duplicate asset return columns"):
        asset_return_correlation_matrix(returns)


def test_asset_return_correlation_matrix_requires_two_observations():
    returns = pd.DataFrame({asset: [0.01] for asset in ASSET_CLASSES})

    with pytest.raises(ValueError, match="At least two return observations"):
        asset_return_correlation_matrix(returns)


def test_asset_return_correlation_matrix_rejects_zero_variance_assets():
    returns = pd.DataFrame({asset: [0.01, 0.02, 0.03] for asset in ASSET_CLASSES})
    returns["Cash"] = 0.01

    with pytest.raises(ValueError, match="zero-variance asset classes: Cash"):
        asset_return_correlation_matrix(returns)


@pytest.mark.parametrize("bad_return", [float("nan"), float("inf"), -1.0])
def test_asset_return_correlation_matrix_rejects_invalid_returns(bad_return):
    returns = pd.DataFrame({asset: [0.01, 0.02, 0.03] for asset in ASSET_CLASSES})
    returns.loc[1, "Gold"] = bad_return

    with pytest.raises(ValueError, match="Periodic returns"):
        asset_return_correlation_matrix(returns)

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


def test_max_drawdown_remains_stable_when_compounded_wealth_overflows():
    returns = pd.Series([1e308, 1e308, -0.50])
    assert max_drawdown(returns) == pytest.approx(-0.50)


def test_rolling_risk_metrics_preserve_index_and_wait_for_full_window():
    index = pd.date_range("2025-01-31", periods=3, freq="ME")
    returns = pd.Series([0.10, -0.20, 0.05], index=index)

    result = rolling_risk_metrics(returns, window=2)

    assert result.index.equals(index)
    assert list(result.columns) == [
        "Rolling annualized volatility",
        "Rolling maximum drawdown",
    ]
    assert result.iloc[0].isna().all()
    expected_volatility = pd.Series([0.10, -0.20]).std(ddof=1) * (12 ** 0.5)
    assert result.iloc[1]["Rolling annualized volatility"] == pytest.approx(expected_volatility)
    assert result.iloc[1]["Rolling maximum drawdown"] == pytest.approx(-0.20)
    assert result.iloc[2]["Rolling maximum drawdown"] == pytest.approx(-0.20)


def test_rolling_risk_metrics_returns_nan_when_history_is_shorter_than_window():
    result = rolling_risk_metrics(pd.Series([0.01, 0.02]), window=3)
    assert result.isna().all().all()


@pytest.mark.parametrize("window", [1, 0, -1, 1.5, True])
def test_rolling_risk_metrics_reject_invalid_window(window):
    with pytest.raises(ValueError, match="Rolling window"):
        rolling_risk_metrics(pd.Series([0.01, 0.02]), window=window)


def test_rolling_risk_metrics_reject_non_finite_returns():
    with pytest.raises(ValueError, match="Periodic returns must be finite"):
        rolling_risk_metrics(pd.Series([0.01, float("nan")]), window=2)


def test_rolling_risk_metrics_reject_empty_returns():
    with pytest.raises(ValueError, match="At least one return observation"):
        rolling_risk_metrics(pd.Series(dtype=float), window=2)


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


def test_portfolio_wealth_path_rejects_compounding_overflow():
    with pytest.raises(ValueError, match="exceeds the numeric range"):
        portfolio_wealth_path(pd.Series([1e308, 1e308]))


def test_portfolio_wealth_path_rejects_starting_value_scaling_overflow():
    with pytest.raises(ValueError, match="exceeds the numeric range"):
        portfolio_wealth_path(pd.Series([1.0]), starting_value=1e308)
