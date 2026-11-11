import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    ASSET_CLASSES,
    portfolio_risk_contributions,
)

WEIGHTS = {
    "Global equities": 0.50,
    "High-quality bonds": 0.30,
    "Gold": 0.10,
    "Cash": 0.10,
}


def test_risk_contributions_reconcile_to_annualized_portfolio_volatility():
    series = [0.01, -0.02, 0.03, 0.00, 0.02]
    returns = pd.DataFrame({asset: series for asset in ASSET_CLASSES})

    result = portfolio_risk_contributions(returns, WEIGHTS)

    portfolio_returns = pd.Series(series)
    expected_volatility = portfolio_returns.std(ddof=1) * np.sqrt(12)
    assert result["Component contribution to annualized volatility"].sum() == pytest.approx(
        expected_volatility
    )
    assert result["Share of total volatility"].sum() == pytest.approx(1.0)
    assert list(result.index) == ASSET_CLASSES


def test_risk_contribution_shares_match_weights_for_identical_asset_returns():
    series = [0.01, -0.02, 0.03, 0.00, 0.02]
    returns = pd.DataFrame({asset: series for asset in ASSET_CLASSES})

    result = portfolio_risk_contributions(returns, WEIGHTS)

    assert result["Share of total volatility"].to_numpy() == pytest.approx(
        [WEIGHTS[asset] for asset in ASSET_CLASSES]
    )


def test_risk_contributions_can_be_negative_for_a_diversifying_asset():
    returns = pd.DataFrame({
        "Global equities": [0.10, -0.10, 0.10, -0.10],
        "High-quality bonds": [-0.10, 0.10, -0.10, 0.10],
        "Gold": [0.0, 0.0, 0.0, 0.0],
        "Cash": [0.0, 0.0, 0.0, 0.0],
    })
    weights = {
        "Global equities": 0.60,
        "High-quality bonds": 0.30,
        "Gold": 0.10,
        "Cash": 0.0,
    }

    result = portfolio_risk_contributions(returns, weights)

    assert result.loc["High-quality bonds", "Component contribution to annualized volatility"] < 0
    assert result["Share of total volatility"].sum() == pytest.approx(1.0)


def test_risk_contributions_respect_custom_periods_per_year():
    series = [0.01, -0.02, 0.03, 0.00, 0.02]
    returns = pd.DataFrame({asset: series for asset in ASSET_CLASSES})

    monthly = portfolio_risk_contributions(returns, WEIGHTS, periods_per_year=12)
    quarterly = portfolio_risk_contributions(returns, WEIGHTS, periods_per_year=4)

    assert quarterly["Component contribution to annualized volatility"].sum() == pytest.approx(
        monthly["Component contribution to annualized volatility"].sum() / np.sqrt(3)
    )


def test_risk_contributions_reject_missing_asset_columns():
    with pytest.raises(ValueError, match="Missing asset return columns"):
        portfolio_risk_contributions(pd.DataFrame({"Global equities": [0.01, 0.02]}), WEIGHTS)


def test_risk_contributions_reject_duplicate_asset_columns():
    returns = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04, 0.05], [0.02, 0.01, 0.04, 0.03, 0.06]],
        columns=[
            "Global equities",
            "Global equities",
            "High-quality bonds",
            "Gold",
            "Cash",
        ],
    )
    with pytest.raises(ValueError, match="Duplicate asset return columns"):
        portfolio_risk_contributions(returns, WEIGHTS)


def test_risk_contributions_reject_insufficient_observations():
    returns = pd.DataFrame({asset: [0.01] for asset in ASSET_CLASSES})
    with pytest.raises(ValueError, match="At least two return observations"):
        portfolio_risk_contributions(returns, WEIGHTS)


def test_risk_contributions_reject_zero_volatility_portfolio():
    returns = pd.DataFrame({
        "Global equities": [0.10, -0.10, 0.05],
        "High-quality bonds": [-0.10, 0.10, -0.05],
        "Gold": [0.0, 0.0, 0.0],
        "Cash": [0.0, 0.0, 0.0],
    })
    offsetting_weights = {
        "Global equities": 0.50,
        "High-quality bonds": 0.50,
        "Gold": 0.0,
        "Cash": 0.0,
    }
    with pytest.raises(ValueError, match="Portfolio volatility must be positive"):
        portfolio_risk_contributions(returns, offsetting_weights)


@pytest.mark.parametrize("bad_return", [float("nan"), float("inf"), -1.0])
def test_risk_contributions_reject_invalid_returns(bad_return):
    returns = pd.DataFrame({asset: [0.01, 0.02, 0.03] for asset in ASSET_CLASSES})
    returns.loc[1, "Gold"] = bad_return
    with pytest.raises(ValueError, match="Periodic returns"):
        portfolio_risk_contributions(returns, WEIGHTS)


@pytest.mark.parametrize("periods_per_year", [0, -1, 1.5, True])
def test_risk_contributions_reject_invalid_annualization_frequency(periods_per_year):
    returns = pd.DataFrame({asset: [0.01, 0.02, 0.03] for asset in ASSET_CLASSES})
    with pytest.raises(ValueError, match="Periods per year"):
        portfolio_risk_contributions(returns, WEIGHTS, periods_per_year)


def test_risk_contributions_reject_invalid_weights():
    invalid_weights = dict(WEIGHTS)
    invalid_weights["Cash"] = 0.05
    returns = pd.DataFrame({asset: [0.01, 0.02, 0.03] for asset in ASSET_CLASSES})
    with pytest.raises(ValueError, match="sum to 100%"):
        portfolio_risk_contributions(returns, invalid_weights)
