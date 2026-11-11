import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.portfolio import (
    ASSET_CLASSES,
    allocation_concentration_metrics,
    asset_return_correlation_matrix,
    DEFAULT_WEIGHTS,
    SAMPLE_ALLOCATIONS,
    apply_one_period_shock,
    performance_metrics,
    portfolio_returns,
    portfolio_contributions,
    portfolio_risk_contributions,
    portfolio_wealth_path,
    rolling_risk_metrics,
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
    annual_target_return = st.number_input(
        "Annual minimum acceptable return for Sortino (%)",
        min_value=-50.0,
        max_value=100.0,
        value=3.0,
        step=0.5,
    ) / 100

if not np.isclose(weight_sum, 1.0):
    st.error("Your allocation must total exactly 100% before the analysis can be interpreted.")
    st.stop()

returns = synthetic_monthly_returns()
port_returns = portfolio_returns(returns, weights)
metrics = performance_metrics(port_returns, annual_rf, annual_target_return)

st.info("The return series is generated from seeded synthetic distributions to make this demo reproducible. It is not actual historical performance.")
cols = st.columns(4)
for col, (label, value, fmt) in zip(cols, [
    ("Annualized return", metrics["Annualized return"], "{:.2%}"),
    ("Annualized volatility", metrics["Annualized volatility"], "{:.2%}"),
    ("Maximum drawdown", metrics["Maximum drawdown"], "{:.2%}"),
    ("Sharpe-style ratio", metrics["Sharpe-style ratio"], "{:.2f}"),
]):
    col.metric(label, fmt.format(value))

st.subheader("Historical downside risk (95%)")
st.caption(
    "Empirical loss estimates from the synthetic monthly sample. VaR is the "
    "5th-percentile loss threshold; CVaR is the average loss in observations "
    "at or below that threshold. With only 60 observations, these tail estimates "
    "are highly sample-sensitive and are not forecasts or risk limits."
)
tail_cols = st.columns(2)
tail_cols[0].metric(
    "Historical VaR (95%)",
    f'{metrics["Historical VaR (95%)"]:.2%}',
    help="Positive loss at the empirical 5th-percentile monthly return threshold.",
)
tail_cols[1].metric(
    "Historical CVaR (95%)",
    f'{metrics["Historical CVaR (95%)"]:.2%}',
    help="Average positive loss among observations at or below the historical VaR return threshold.",
)

st.subheader("Target-based downside risk")
st.caption(
    f"Sortino analysis uses an annual minimum acceptable return of {annual_target_return:.2%}, "
    "converted to an effective monthly target. Downside deviation includes only "
    "shortfalls below that target and is annualized by the square root of 12."
)
sortino_cols = st.columns(2)
sortino_value = metrics["Sortino ratio"]
sortino_display = f"{sortino_value:.2f}" if np.isfinite(sortino_value) else "N/A"
sortino_cols[0].metric(
    "Annualized downside deviation",
    f'{metrics["Annualized downside deviation"]:.2%}',
    help="Root-mean-square monthly shortfall below the selected target, annualized by √12.",
)
sortino_cols[1].metric(
    "Sortino ratio",
    sortino_display,
    help="Arithmetic annualized excess return above the selected target divided by annualized downside deviation. N/A means no downside observations were present.",
)

concentration = allocation_concentration_metrics(weights)
st.subheader("Allocation concentration")
concentration_cols = st.columns(3)
concentration_cols[0].metric(
    "Herfindahl-Hirschman index",
    f'{concentration["Herfindahl-Hirschman index"]:.3f}',
    help="Sum of squared allocation weights. Lower values indicate more evenly distributed weights.",
)
concentration_cols[1].metric(
    "Effective number of asset classes",
    f'{concentration["Effective number of asset classes"]:.2f}',
    help="The reciprocal of the concentration index; it expresses the allocation as an equivalent number of equally weighted asset classes.",
)
concentration_cols[2].metric(
    "Largest allocation",
    f'{concentration["Largest asset class"]} ({concentration["Largest allocation"]:.0%})',
)
st.caption(
    "Weight concentration is not a standalone risk measure and does not account for "
    "correlations, underlying holdings, or liquidity."
)

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


correlation = asset_return_correlation_matrix(returns)
st.subheader("Asset-class return correlation")
st.caption(
    "Pearson correlations across the same 60 seeded synthetic monthly observations. "
    "These describe this illustrative sample only and are not forecasts or evidence "
    "of live-market diversification."
)
correlation_chart = px.imshow(
    correlation,
    text_auto=".2f",
    zmin=-1.0,
    zmax=1.0,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Synthetic monthly asset-class return correlations",
)
correlation_chart.update_layout(coloraxis_colorbar_title="Correlation")
st.plotly_chart(correlation_chart, use_container_width=True)


rolling_risk = rolling_risk_metrics(port_returns, window=12)
rolling_risk_df = rolling_risk.rename_axis("Period").reset_index()

st.subheader("Trailing 12-month risk")
st.caption(
    "Each point uses the latest 12 synthetic monthly observations. "
    "The first 11 months are omitted until a complete lookback window is available."
)
risk_volatility_col, risk_drawdown_col = st.columns(2)
with risk_volatility_col:
    volatility_chart = px.line(
        rolling_risk_df,
        x="Period",
        y="Rolling annualized volatility",
        title="Rolling annualized volatility",
        labels={"Rolling annualized volatility": "Annualized volatility"},
    )
    volatility_chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(volatility_chart, use_container_width=True)
with risk_drawdown_col:
    drawdown_chart = px.line(
        rolling_risk_df,
        x="Period",
        y="Rolling maximum drawdown",
        title="Rolling maximum drawdown",
        labels={"Rolling maximum drawdown": "Maximum drawdown"},
    )
    drawdown_chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(drawdown_chart, use_container_width=True)


risk_contributions = portfolio_risk_contributions(returns, weights)
risk_contribution_df = risk_contributions.reset_index()

st.subheader("Asset-level risk contribution")
st.caption(
    "Euler decomposition allocates annualized portfolio volatility across asset classes "
    "using the sample covariance matrix. Component contributions add to total portfolio "
    "volatility. A negative contribution means the asset reduced measured portfolio risk "
    "in this synthetic sample; it is not a guaranteed hedge."
)
risk_chart = px.bar(
    risk_contribution_df,
    x="Share of total volatility",
    y="Asset class",
    orientation="h",
    title="Asset-class share of annualized portfolio volatility",
    labels={"Share of total volatility": "Share of total volatility"},
)
risk_chart.update_xaxes(tickformat=".0%")
st.plotly_chart(risk_chart, use_container_width=True)
st.dataframe(
    risk_contribution_df.style.format({
        "Portfolio weight": "{:.1%}",
        "Marginal contribution to annualized volatility": "{:.2%}",
        "Component contribution to annualized volatility": "{:.2%}",
        "Share of total volatility": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
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
    m = performance_metrics(portfolio_returns(returns, allocation), annual_rf, annual_target_return)
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
        "Historical VaR (95%)": "{:.2%}",
        "Historical CVaR (95%)": "{:.2%}",
        "Annualized downside deviation": "{:.2%}",
        "Sortino ratio": "{:.2f}",
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
    - Historical 95% VaR is the empirical 5th-percentile monthly loss threshold; CVaR is the average loss among observations at or below that threshold. With 60 synthetic observations, both estimates are highly sample-sensitive.
    - Sortino ratio compares arithmetic annualized excess return above the selected minimum acceptable return with annualized downside deviation. If no monthly observations fall below the target, the ratio is undefined and shown as N/A.
    - The Sharpe-style ratio depends on the assumed risk-free rate.
    - Stress testing applies a simple weighted sum of one-period shocks; it does not model correlations, path dependence, fees, taxes, currency effects or client-specific liabilities.
    - A real advisory process would require client-specific objectives, time horizon, liquidity requirements, liabilities, risk tolerance and appropriate professional review.
    """)
st.caption("All portfolios and scenarios are hypothetical. This tool is not financial, legal or tax advice.")
