import pandas as pd
import pytest

from src.portfolio import (
    ASSET_CLASSES,
    DEFAULT_WEIGHTS,
    portfolio_contributions,
    portfolio_returns,
)


@pytest.mark.parametrize("calculate", [portfolio_returns, portfolio_contributions])
def test_portfolio_calculations_reject_duplicate_asset_columns(calculate):
    columns = ASSET_CLASSES + ["Global equities"]
    returns = pd.DataFrame([[0.01, 0.02, 0.03, 0.001, 0.50]], columns=columns)

    with pytest.raises(ValueError, match="Duplicate asset return columns: Global equities"):
        calculate(returns, DEFAULT_WEIGHTS)
