import numpy as np
import pandas as pd
import pytest

from src.portfolio import downside_risk_metrics, performance_metrics


def test_downside_risk_metrics_annualize_target_shortfalls():
    returns = pd.Series([-0.10, 0.02, -0.05, 0.04])

    metrics = downside_risk_metrics(returns, annual_target_return=0.0)

    expected_deviation = np.sqrt((0.10**2 + 0.05**2) / 4) * np.sqrt(12)
    expected_excess_return = returns.mean() * 12
    assert metrics["Annualized downside deviation"] == pytest.approx(expected_deviation)
    assert metrics["Sortino ratio"] == pytest.approx(expected_excess_return / expected_deviation)


def test_downside_risk_metrics_convert_effective_annual_target_to_monthly():
    returns = pd.Series([0.005, 0.015, -0.01, 0.02])
    annual_target = 0.12
    monthly_target = (1 + annual_target) ** (1 / 12) - 1
    shortfalls = np.minimum(returns.to_numpy() - monthly_target, 0.0)

    metrics = downside_risk_metrics(returns, annual_target_return=annual_target)

    expected = np.sqrt(np.mean(shortfalls**2)) * np.sqrt(12)
    assert metrics["Annualized downside deviation"] == pytest.approx(expected)


def test_downside_risk_metrics_returns_nan_sortino_when_no_period_is_below_target():
    metrics = downside_risk_metrics(pd.Series([0.02, 0.03, 0.04]), annual_target_return=0.0)

    assert metrics["Annualized downside deviation"] == 0.0
    assert np.isnan(metrics["Sortino ratio"])


def test_downside_risk_metrics_rejects_empty_returns():
    with pytest.raises(ValueError, match="At least one return observation"):
        downside_risk_metrics(pd.Series(dtype=float))


@pytest.mark.parametrize("target", [-1.0, -1.1, float("nan"), float("inf"), float("-inf"), "not-a-rate"])
def test_downside_risk_metrics_rejects_invalid_annual_target(target):
    with pytest.raises(ValueError, match="Annual target return"):
        downside_risk_metrics(pd.Series([-0.02, 0.01]), annual_target_return=target)


@pytest.mark.parametrize("bad_return", [float("nan"), float("inf"), -1.0])
def test_downside_risk_metrics_rejects_invalid_periodic_returns(bad_return):
    with pytest.raises(ValueError, match="Periodic returns"):
        downside_risk_metrics(pd.Series([-0.02, bad_return, 0.03]))


def test_performance_metrics_include_sortino_and_downside_deviation():
    metrics = performance_metrics(
        pd.Series([-0.04, 0.02, -0.01, 0.03]),
        annual_risk_free_rate=0.03,
        annual_target_return=0.02,
    )

    assert "Annualized downside deviation" in metrics
    assert "Sortino ratio" in metrics
    assert metrics["Annualized downside deviation"] > 0
    assert np.isfinite(metrics["Sortino ratio"])
