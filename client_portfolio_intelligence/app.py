from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client_portfolio_intelligence.src.analytics import (
    normalize_weights, portfolio_metrics, asset_metrics,
    build_portfolio_returns, stress_scenario, rebalancing_table,
    benchmark_metrics, risk_label,
)
from client_portfolio_intelligence.src.sample_data import (
    make_sample_prices, CLIENTS, DEFAULT_HOLDINGS, TARGET_ALLOCATIONS,
)

st.set_page_config(page_title="Private Banking Client Portfolio Intelligence", page_icon="💼", layout="wide")
st.title("Private Banking Client Portfolio Intelligence")
st.caption("Portfolio analytics prototype for client reviews, allocation conversations and risk oversight.")
st.warning("Educational demo only. Client profiles and default prices are fictional. Not investment advice.")

with st.sidebar:
    st.header("Workspace")
    data_mode = st.radio("Market data", ["Synthetic demo data", "Upload price history"])
    uploaded_prices = st.file_uploader("Prices CSV", type=["csv"]) if data_mode == "Upload price history" else None
    client_name = st.selectbox("Select client", list(CLIENTS.keys()))
    client = CLIENTS[client_name]
    st.divider()
    risk_free_rate = st.number_input("Annual risk-free rate (%)", 0.0, 20.0, 4.0, 0.25) / 100
    stress_pct = st.slider("Uniform market shock (%)", -60, 10, -20, 5) / 100

if uploaded_prices is not None:
    raw = pd.read_csv(uploaded_prices)
    date_col = next((c for c in raw.columns if c.lower().strip() == "date"), None)
    if date_col is None:
        st.error("Your price CSV must contain a date column.")
        st.stop()
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    prices = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").ffill().dropna()
    prices.columns = [str(c).upper().strip() for c in prices.columns]
    if len(prices) < 3:
        st.error("Upload at least three valid price observations.")
        st.stop()
else:
    prices = make_sample_prices()

holdings = DEFAULT_HOLDINGS[client_name].copy()
holdings = holdings.loc[[ticker for ticker in holdings.index if ticker in prices.columns]]
if holdings.empty:
    st.error("No holdings match your price CSV columns. Check ticker names.")
    st.stop()
holdings = normalize_weights(holdings)
targets = TARGET_ALLOCATIONS[client_name].reindex(holdings.index).fillna(0)
targets = normalize_weights(targets) if targets.sum() > 0 else holdings.copy()

returns = build_portfolio_returns(prices, holdings)
metrics = portfolio_metrics(prices[holdings.index], holdings, risk_free_rate)
benchmark_ticker = "SPY" if "SPY" in prices.columns else prices.columns[0]
benchmark = benchmark_metrics(prices[benchmark_ticker], risk_free_rate)
asset = asset_metrics(prices[holdings.index])

st.subheader(f"Client overview — {client_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Client segment", client["segment"])
c2.metric("Risk profile", client["risk_profile"])
c3.metric("Investment horizon", client["horizon"])
c4.metric("Liquidity reserve target", f"{client['liquidity_target']:.0%}")
st.write(client["objective"])

st.subheader("Portfolio scorecard")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Annualized return", f"{metrics['annual_return']:.2%}")
k2.metric("Annualized volatility", f"{metrics['volatility']:.2%}")
k3.metric("Sharpe ratio", f"{metrics['sharpe']:.2f}")
k4.metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")
k5.metric("Risk indicator", risk_label(metrics["volatility"], client["risk_profile"]))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Performance", "Allocation & rebalancing", "Risk & suitability",
    "Scenario lab", "Client review", "Data & exports",
])

with tab1:
    st.subheader("Portfolio value trajectory")
    growth = (1 + returns).cumprod()
    st.plotly_chart(px.line(x=growth.index, y=growth.values, labels={"x": "Date", "y": "Growth of $1"}, title="Hypothetical growth of $1 invested"), use_container_width=True)
    left, right = st.columns(2)
    left.metric(f"{benchmark_ticker} annualized return", f"{benchmark['annual_return']:.2%}")
    right.metric("Portfolio minus benchmark", f"{metrics['annual_return'] - benchmark['annual_return']:.2%}")
    st.caption("The selected benchmark may not match the client's investment mandate.")
    comparison = pd.DataFrame({
        "Metric": ["Annualized return", "Annualized volatility", "Sharpe ratio", "Maximum drawdown"],
        "Client portfolio": [metrics["annual_return"], metrics["volatility"], metrics["sharpe"], metrics["max_drawdown"]],
        f"Benchmark ({benchmark_ticker})": [benchmark["annual_return"], benchmark["volatility"], benchmark["sharpe"], benchmark["max_drawdown"]],
    })
    st.dataframe(comparison, use_container_width=True)

with tab2:
    st.subheader("Current versus target allocation")
    allocation = pd.DataFrame({"ticker": holdings.index, "current_weight": holdings.values, "target_weight": targets.reindex(holdings.index).values})
    left, right = st.columns(2)
    left.plotly_chart(px.pie(allocation, names="ticker", values="current_weight", hole=0.45, title="Current allocation"), use_container_width=True)
    long_alloc = allocation.melt(id_vars="ticker", var_name="allocation", value_name="weight", value_vars=["current_weight", "target_weight"])
    right.plotly_chart(px.bar(long_alloc, x="ticker", y="weight", color="allocation", barmode="group", title="Current vs target allocation"), use_container_width=True)
    st.subheader("Illustrative rebalancing trades")
    portfolio_value = st.number_input("Illustrative portfolio value ($)", min_value=1000, max_value=1000000000, value=1000000, step=25000)
    rebalance = rebalancing_table(holdings, targets, portfolio_value)
    st.dataframe(rebalance.style.format({
        "current_weight": "{:.2%}", "target_weight": "{:.2%}",
        "current_value": "$ {:,.0f}", "target_value": "$ {:,.0f}", "trade_value": "$ {:,.0f}",
    }), use_container_width=True)
    st.caption("Illustrative trades exclude taxes, fees, liquidity restrictions and market impact.")

with tab3:
    st.subheader("Asset-level risk")
    st.dataframe(asset.style.format({"annual_return": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.2f}", "max_drawdown": "{:.2%}"}), use_container_width=True)
    corr = prices[holdings.index].pct_change().dropna().corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Asset return correlation matrix"), use_container_width=True)
    st.write(f"**Risk profile:** {client['risk_profile']}")
    st.write(f"**Objective:** {client['objective']}")
    st.write(f"**Horizon:** {client['horizon']}")
    st.write(f"**Liquidity target:** {client['liquidity_target']:.0%}")
    st.info("Confirm objectives, ability to bear losses, liquidity needs, tax circumstances and restrictions before discussing allocation changes.")
    st.caption("This educational indicator is not a regulated suitability assessment.")

with tab4:
    st.subheader("Uniform hypothetical market shock")
    result = stress_scenario(holdings, stress_pct)
    st.metric("Estimated portfolio impact", f"{result['portfolio_impact']:.2%}")
    st.dataframe(result["asset_impacts"].rename("Weighted impact").to_frame().style.format("{:.2%}"), use_container_width=True)
    st.caption("A hypothetical scenario, not a forecast.")
    st.subheader("Asset-specific scenario")
    shock_inputs = {}
    cols = st.columns(min(3, len(holdings)))
    for i, ticker in enumerate(holdings.index):
        with cols[i % len(cols)]:
            shock_inputs[ticker] = st.slider(f"{ticker} shock (%)", -70, 30, int(stress_pct * 100), 5, key=f"shock_{ticker}") / 100
    custom = stress_scenario(holdings, pd.Series(shock_inputs))
    st.metric("Asset-specific scenario impact", f"{custom['portfolio_impact']:.2%}")
    st.dataframe(custom["asset_impacts"].rename("Weighted impact").to_frame().style.format("{:.2%}"), use_container_width=True)

with tab5:
    st.subheader("Client review summary")
    review = f"""# Private Banking Portfolio Review

**Client:** {client_name}

**Segment:** {client['segment']}

**Risk profile:** {client['risk_profile']}

**Investment horizon:** {client['horizon']}

## Objective
{client['objective']}

## Portfolio scorecard
- Annualized return: {metrics['annual_return']:.2%}
- Annualized volatility: {metrics['volatility']:.2%}
- Sharpe ratio: {metrics['sharpe']:.2f}
- Maximum drawdown: {metrics['max_drawdown']:.2%}
- Benchmark ({benchmark_ticker}) annualized return: {benchmark['annual_return']:.2%}

## Review checklist
- Reconfirm objectives, risk tolerance, capacity for loss and time horizon.
- Validate liquidity needs and investment restrictions.
- Discuss benchmark relevance and drawdown tolerance.
- Review fees, taxes and implementation constraints.

*Educational prototype only. Not investment advice.*
"""
    st.markdown(review)
    st.download_button("Download client review", review.encode("utf-8"), f"{client_name.lower().replace(' ', '_')}_review.md", "text/markdown")

with tab6:
    st.subheader("Data preview and exports")
    st.dataframe(prices.tail(25), use_container_width=True)
    st.download_button("Download price history CSV", prices.to_csv().encode("utf-8"), "price_history.csv", "text/csv")
    st.download_button("Download portfolio metrics CSV", pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8"), "portfolio_metrics.csv", "text/csv")
    st.download_button("Download allocation CSV", allocation.to_csv(index=False).encode("utf-8"), "client_allocation.csv", "text/csv")
    st.download_button("Download asset metrics CSV", asset.to_csv().encode("utf-8"), "asset_metrics.csv", "text/csv")
    st.caption("Do not upload confidential client data to an unapproved environment.")
