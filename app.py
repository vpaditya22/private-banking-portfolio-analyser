import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.portfolio import (
    ASSET_CLASSES,
    DEFAULT_WEIGHTS,
    SAMPLE_ALLOCATIONS,
    apply_one_period_shock,
    performance_metrics,
    portfolio_returns,
    portfolio_contributions,
    portfolio_wealth_path,
    synthetic_monthly_returns,
)

st.set_page_config(page_title="Portfolio Risk & Allocation Analyser", page_icon="📊", layout="wide")
st.title("Private Banking Portfolio Risk & Allocation Analyser")
st.caption("Educational portfolio-analysis demo · Synthetic returns · Not investment advice")

with st.sidebar:
    st.header("Illustrative portfolio")
    starting_value = st.number_input("Starting portfolio value ($)", min_value=1000, max_value=100_000_000, value=1_000_000, step=25_000)
    st.caption("Enter allocation weights. They must add up to 100%.")
    weights = {}
    for asset in ASSET_CLASSES:
        weights[asset] = st.slider(asset, min_value=0, max_value=100, value=int(DEFAULT_WEIGHTS[asset] * 100), step=5) / 100
    weight_sum = sum(weights.values())
    st.write(f"**Total allocation:** {weight_sum:.0%}")
    annual_rf = st.number_input("Assumed annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.25) / 100

if not np.isclose(weight_sum, 1.0):
    st.error("Your allocation must total exactly 100% before the analysis can be interpreted.")
    st.stop()

returns = synthetic_monthly_returns()
port_returns = portfolio_returns(returns, weights)
metrics = performance_metrics(port_returns, annual_rf)

st.info("The return series is generated from seeded synthetic distributions to make this demo reproducible. It is not actual historical performance.")
cols = st.columns(4)
for col, (label, value, fmt) in zip(cols, [
    ("Annualized return", metrics["Annualized return"], "{:.2%}"),
    ("Annualized volatility", metrics["Annualized volatility"], "{:.2%}"),
    ("Maximum drawdown", metrics["Maximum drawdown"], "{:.2%}"),
    ("Sharpe-style ratio", metrics["Sharpe-style ratio"], "{:.2f}"),
]):
    col.metric(label, fmt.format(value))

left, right = st.columns(2)
with left:
    allocation_df = pd.DataFrame({"Asset class": ASSET_CLASSES, "Weight": [weights[a] for a in ASSET_CLASSES]})
    st.plotly_chart(px.pie(allocation_df, names="Asset class", values="Weight", title="Portfolio allocation"), use_container_width=True)
with right:
    wealth = portfolio_wealth_path(port_returns, starting_value)
    wealth_df = pd.DataFrame({
        "Period": ["Initial"] + returns.index.strftime("%b %Y").tolist(),
        "Illustrative portfolio value": wealth.to_numpy(),
    })
    st.plotly_chart(
        px.line(
            wealth_df,
            x="Period",
            y="Illustrative portfolio value",
            title="Illustrative portfolio path (including starting capital)",
            labels={
                "Period": "Period",
                "Illustrative portfolio value": "Portfolio value ($)",
            },
        ),
        use_container_width=True,
    )


st.subheader("Asset-level return contribution")
st.caption(
    "Each contribution is the asset-class return multiplied by its portfolio weight. "
    "Contributions add up to the portfolio return within each synthetic month. "
    "Average monthly contributions are arithmetic averages, not compounded returns."
)
monthly_contributions = portfolio_contributions(returns, weights)
average_contribution = monthly_contributions.mean().reindex(ASSET_CLASSES)
last_period_contribution = monthly_contributions.iloc[-1].reindex(ASSET_CLASSES)
contribution_summary = pd.DataFrame({
    "Asset class": ASSET_CLASSES,
    "Average monthly contribution": [average_contribution[a] for a in ASSET_CLASSES],
    "Last synthetic month contribution": [last_period_contribution[a] for a in ASSET_CLASSES],
})
contribution_left, contribution_right = st.columns([3, 2])
with contribution_left:
    st.plotly_chart(
        px.bar(
            contribution_summary,
            x="Asset class",
            y="Average monthly contribution",
            title="Average monthly contribution by asset class",
            labels={"Average monthly contribution": "Contribution (monthly return)"},
        ),
        use_container_width=True,
    )
with contribution_right:
    st.dataframe(
        contribution_summary.style.format({
            "Average monthly contribution": "{:.3%}",
            "Last synthetic month contribution": "{:.3%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
st.download_button(
    "Download monthly contribution CSV",
    monthly_contributions.to_csv(index=True, index_label="Synthetic month").encode("utf-8"),
    "monthly_asset_contributions.csv",
    "text/csv",
)

st.subheader("Allocation comparison")
comparison = []
for name, allocation in SAMPLE_ALLOCATIONS.items():
    m = performance_metrics(portfolio_returns(returns, allocation), annual_rf)
    comparison.append({"Illustrative allocation": name, **m})
comparison.append({"Illustrative allocation": "Your allocation", **metrics})
comparison_df = pd.DataFrame(comparison)
st.dataframe(
    comparison_df.style.format({
        "Annualized return": "{:.2%}",
        "Annualized volatility": "{:.2%}",
        "Maximum drawdown": "{:.2%}",
        "Cumulative return": "{:.2%}",
        "Sharpe-style ratio": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)
st.download_button("Download comparison CSV", comparison_df.to_csv(index=False).encode("utf-8"), "portfolio_comparison.csv", "text/csv")

st.subheader("Hypothetical market stress test")
st.write("Adjust one-period shocks for each asset class. These are user-entered scenarios, not forecasts.")
shock_cols = st.columns(4)
default_shocks = {
    "Global equities": -0.20,
    "High-quality bonds": 0.03,
    "Gold": 0.08,
    "Cash": 0.00,
}
shocks = {}
for col, asset in zip(shock_cols, ASSET_CLASSES):
    shocks[asset] = col.number_input(f"{asset} shock (%)", min_value=-100.0, max_value=100.0, value=default_shocks[asset]*100, step=1.0) / 100
impact = apply_one_period_shock(weights, shocks)
delta = starting_value * impact
st.metric("Illustrative portfolio impact", f"{impact:.2%}", f"${delta:,.0f}")
st.write(f"Illustrative value after this one-period shock: **${starting_value + delta:,.0f}**")

with st.expander("Methodology and limitations"):
    st.markdown("""
    - Monthly asset-class returns are synthetic and seeded for reproducibility.
    - Annualized return compounds the generated monthly returns; volatility uses monthly standard deviation × √12.
    - Maximum drawdown measures the largest peak-to-trough decline in the illustrative wealth index.
    - The Sharpe-style ratio depends on the assumed risk-free rate.
    - Stress testing applies a simple weighted sum of one-period shocks; it does not model correlations, path dependence, fees, taxes, currency effects or client-specific liabilities.
    - A real advisory process would require client-specific objectives, time horizon, liquidity requirements, liabilities, risk tolerance and appropriate professional review.
    """)
st.caption("All portfolios and scenarios are hypothetical. This tool is not financial, legal or tax advice.")
