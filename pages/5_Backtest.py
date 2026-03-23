import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting.engine import BacktestEngine
from backtesting.strategies import get_strategy

st.set_page_config(page_title="Backtesting", page_icon="📈", layout="wide")
st.title("📈 Backtesting Engine")
st.caption("3 strategies · Metriques completes · Comparaison Buy & Hold")

ASSETS = {
    "CAC 40": "^FCHI",
    "S&P 500": "^GSPC",
    "LVMH": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Sanofi": "SAN.PA",
    "BNP Paribas": "BNP.PA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Amazon": "AMZN",
    "JPMorgan": "JPM",
    "XLK (Tech ETF)": "XLK",
    "XLF (Finance ETF)": "XLF",
    "XLE (Energie ETF)": "XLE",
    "XLV (Sante ETF)": "XLV",
}

PAIRS = {
    "LVMH / Kering": ("MC.PA", "KER.PA"),
    "TotalEnergies / BP": ("TTE.PA", "BP.L"),
    "Apple / Microsoft": ("AAPL", "MSFT"),
    "JPMorgan / Goldman": ("JPM", "GS"),
    "Coca-Cola / PepsiCo": ("KO", "PEP"),
    "XLK / XLV": ("XLK", "XLV"),
}

@st.cache_data(ttl=3600)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def plot_equity_curve(equity, benchmark_equity, bh_equity, title):
    fig = go.Figure()

    # Normalise à 100 pour comparaison
    norm_strat = equity / equity.iloc[0] * 100
    norm_bm    = benchmark_equity / benchmark_equity.iloc[0] * 100
    norm_bh    = bh_equity / bh_equity.iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=equity.index, y=norm_strat,
        name="Strategie", line=dict(color="#00d4aa", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=norm_bh,
        name="Buy & Hold", line=dict(color="#f5a623", width=1.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=norm_bm,
        name="S&P 500", line=dict(color="#6c8ebf", width=1.5, dash="dot")
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="white", opacity=0.3)
    fig.update_layout(
        title=title, template="plotly_dark",
        yaxis_title="Performance (base 100)",
        height=400, margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified"
    )
    return fig

def plot_drawdown(equity):
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", name="Drawdown",
        line=dict(color="#ff6b6b", width=1),
        fillcolor="rgba(255,107,107,0.15)"
    ))
    fig.update_layout(
        title="Drawdown", template="plotly_dark",
        yaxis_title="Drawdown %",
        height=250, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_monthly_returns(returns):
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    colors  = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in monthly.values]
    fig = go.Figure(go.Bar(
        x=monthly.index, y=monthly.values,
        marker_color=colors, name="Rendement mensuel"
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
    fig.update_layout(
        title="Rendements mensuels", template="plotly_dark",
        yaxis_title="%", height=280,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_signals(data, signals_df):
    close = data["Close"].squeeze()
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        name="Prix", line=dict(color="white", width=1.5)
    ))
    if not signals_df.empty:
        buys  = signals_df[signals_df["Signal"] == "BUY"]
        sells = signals_df[signals_df["Signal"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["Date"], y=buys["Prix"],
                mode="markers", name="Achat",
                marker=dict(color="#00d4aa", size=10, symbol="triangle-up")
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["Date"], y=sells["Prix"],
                mode="markers", name="Vente",
                marker=dict(color="#ff6b6b", size=10, symbol="triangle-down")
            ))
    fig.update_layout(
        title="Signaux de trading",
        template="plotly_dark", height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def display_metrics(metrics, bh_return, final_capital, initial_capital):
    st.markdown("#### Metriques de performance")
    col1, col2, col3, col4 = st.columns(4)

    strat_return = metrics["Rendement annualisé %"]
    sharpe       = metrics["Sharpe Ratio"]
    mdd          = metrics["Max Drawdown %"]
    win          = metrics["Win Rate %"]

    with col1:
        color = "#00d4aa" if strat_return >= 0 else "#ff6b6b"
        st.markdown(f"""
        <div style='background:#111620;border:1px solid #1e2736;
                    border-radius:8px;padding:16px;text-align:center'>
            <div style='font-size:10px;color:#9aa3b5;letter-spacing:0.1em;margin-bottom:6px'>
                RENDEMENT ANNUALISE
            </div>
            <div style='font-size:28px;font-weight:800;color:{color}'>
                {strat_return:+.1f}%
            </div>
            <div style='font-size:10px;color:#5a6478;margin-top:4px'>
                Buy & Hold : {bh_return:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = "#00d4aa" if sharpe >= 1 else "#f5a623" if sharpe >= 0.5 else "#ff6b6b"
        st.markdown(f"""
        <div style='background:#111620;border:1px solid #1e2736;
                    border-radius:8px;padding:16px;text-align:center'>
            <div style='font-size:10px;color:#9aa3b5;letter-spacing:0.1em;margin-bottom:6px'>
                SHARPE RATIO
            </div>
            <div style='font-size:28px;font-weight:800;color:{color}'>
                {sharpe:.2f}
            </div>
            <div style='font-size:10px;color:#5a6478;margin-top:4px'>
                Sortino : {metrics["Sortino Ratio"]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        color = "#00d4aa" if mdd > -10 else "#f5a623" if mdd > -20 else "#ff6b6b"
        st.markdown(f"""
        <div style='background:#111620;border:1px solid #1e2736;
                    border-radius:8px;padding:16px;text-align:center'>
            <div style='font-size:10px;color:#9aa3b5;letter-spacing:0.1em;margin-bottom:6px'>
                MAX DRAWDOWN
            </div>
            <div style='font-size:28px;font-weight:800;color:{color}'>
                {mdd:.1f}%
            </div>
            <div style='font-size:10px;color:#5a6478;margin-top:4px'>
                Calmar : {metrics["Calmar Ratio"]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        color = "#00d4aa" if win >= 55 else "#f5a623" if win >= 45 else "#ff6b6b"
        st.markdown(f"""
        <div style='background:#111620;border:1px solid #1e2736;
                    border-radius:8px;padding:16px;text-align:center'>
            <div style='font-size:10px;color:#9aa3b5;letter-spacing:0.1em;margin-bottom:6px'>
                WIN RATE
            </div>
            <div style='font-size:28px;font-weight:800;color:{color}'>
                {win:.1f}%
            </div>
            <div style='font-size:10px;color:#5a6478;margin-top:4px'>
                {metrics["Nb Trades"]} trades · PF {metrics["Profit Factor"]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tableau complet
    with st.expander("Voir toutes les metriques"):
        df_metrics = pd.DataFrame([metrics]).T
        df_metrics.columns = ["Valeur"]
        st.dataframe(df_metrics, use_container_width=True)

# ── Interface ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 MA Cross & RSI",
    "📈 Momentum",
    "⚖️ Pairs Trading"
])

# ════════════════════════════════════════════════════════
# TAB 1 — MA CROSS & RSI
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Strategies Techniques — MA Cross & RSI Mean Reversion")

    col1, col2, col3 = st.columns(3)
    with col1:
        asset1    = st.selectbox("Actif", list(ASSETS.keys()), key="asset1")
        period1   = st.selectbox("Periode", ["2y","3y","5y"], index=1, key="period1")
        capital1  = st.number_input("Capital initial ($)", value=100000, step=10000, key="cap1")
    with col2:
        strategy1 = st.radio("Strategie", ["MA Cross", "RSI Mean Reversion"], key="strat1")
    with col3:
        if strategy1 == "MA Cross":
            short_w = st.slider("MA courte", 10, 100, 50, key="short1")
            long_w  = st.slider("MA longue", 50, 300, 200, key="long1")
            params1 = {"short_window": short_w, "long_window": long_w}
        else:
            rsi_per   = st.slider("Periode RSI", 7, 30, 14, key="rsi1")
            oversold  = st.slider("Seuil oversold", 20, 40, 30, key="os1")
            overbought= st.slider("Seuil overbought", 60, 80, 70, key="ob1")
            params1   = {"rsi_period": rsi_per, "oversold": oversold, "overbought": overbought}

    commission1 = st.slider("Commission %", 0.0, 0.5, 0.1, 0.05, key="comm1") / 100
    slippage1   = st.slider("Slippage %", 0.0, 0.5, 0.1, 0.05, key="slip1") / 100

    if st.button("Lancer le backtest", type="primary", key="bt1"):
        with st.spinner("Calcul en cours..."):
            ticker1  = ASSETS[asset1]
            data1    = load_data(ticker1, period1)
            bench1   = load_data("^GSPC", period1)["Close"].squeeze()

            if data1.empty:
                st.error("Impossible de charger les donnees.")
            else:
                strat1   = get_strategy(strategy1, params1)
                engine1  = BacktestEngine(
                    data1, strat1, capital1,
                    commission1, slippage1, bench1
                )
                results1 = engine1.run()

                st.divider()
                display_metrics(
                    results1["metrics"],
                    results1["bh_return"],
                    results1["final_capital"],
                    capital1
                )

                # Equity curve
                bh_equity1 = data1["Close"].squeeze() / float(data1["Close"].squeeze().iloc[0]) * capital1
                st.plotly_chart(
                    plot_equity_curve(
                        results1["equity_curve"], bench1 / float(bench1.iloc[0]) * capital1,
                        bh_equity1, f"{asset1} — {strat1.name}"
                    ), use_container_width=True
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_drawdown(results1["equity_curve"]), use_container_width=True)
                with col_b:
                    st.plotly_chart(plot_monthly_returns(results1["returns"]), use_container_width=True)

                if not results1["signals"].empty:
                    st.plotly_chart(plot_signals(data1, results1["signals"]), use_container_width=True)

                    st.markdown("#### Derniers trades")
                    st.dataframe(
                        results1["signals"].tail(20),
                        use_container_width=True, hide_index=True
                    )

# ════════════════════════════════════════════════════════
# TAB 2 — MOMENTUM
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Strategie Momentum")
    st.caption("Achete les tendances positives, sort quand le momentum devient negatif")

    col1, col2, col3 = st.columns(3)
    with col1:
        asset2   = st.selectbox("Actif", list(ASSETS.keys()), key="asset2")
        period2  = st.selectbox("Periode", ["2y","3y","5y"], index=1, key="period2")
        capital2 = st.number_input("Capital initial ($)", value=100000, step=10000, key="cap2")
    with col2:
        lookback  = st.slider("Lookback (jours)", 5, 60, 20, key="lb2")
        threshold = st.slider("Seuil entree %", 1, 10, 2, key="thr2") / 100
        params2   = {"lookback": lookback, "threshold": threshold}
    with col3:
        commission2 = st.slider("Commission %", 0.0, 0.5, 0.1, 0.05, key="comm2") / 100
        slippage2   = st.slider("Slippage %", 0.0, 0.5, 0.1, 0.05, key="slip2") / 100

    if st.button("Lancer le backtest", type="primary", key="bt2"):
        with st.spinner("Calcul en cours..."):
            ticker2  = ASSETS[asset2]
            data2    = load_data(ticker2, period2)
            bench2   = load_data("^GSPC", period2)["Close"].squeeze()

            if data2.empty:
                st.error("Impossible de charger les donnees.")
            else:
                strat2   = get_strategy("Momentum", params2)
                engine2  = BacktestEngine(
                    data2, strat2, capital2,
                    commission2, slippage2, bench2
                )
                results2 = engine2.run()

                st.divider()
                display_metrics(
                    results2["metrics"],
                    results2["bh_return"],
                    results2["final_capital"],
                    capital2
                )

                bh_equity2 = data2["Close"].squeeze() / float(data2["Close"].squeeze().iloc[0]) * capital2
                st.plotly_chart(
                    plot_equity_curve(
                        results2["equity_curve"],
                        bench2 / float(bench2.iloc[0]) * capital2,
                        bh_equity2,
                        f"{asset2} — Momentum {lookback}j"
                    ), use_container_width=True
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_drawdown(results2["equity_curve"]), use_container_width=True)
                with col_b:
                    st.plotly_chart(plot_monthly_returns(results2["returns"]), use_container_width=True)

                if not results2["signals"].empty:
                    st.plotly_chart(plot_signals(data2, results2["signals"]), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — PAIRS TRADING
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Pairs Trading — Market Neutral")
    st.caption("Exploite la correlation entre 2 actifs similaires via le z-score du spread")

    col1, col2, col3 = st.columns(3)
    with col1:
        pair_choice = st.selectbox("Paire", list(PAIRS.keys()), key="pair3")
        period3     = st.selectbox("Periode", ["2y","3y","5y"], index=1, key="period3")
        capital3    = st.number_input("Capital initial ($)", value=100000, step=10000, key="cap3")
    with col2:
        window3     = st.slider("Fenetre z-score (jours)", 10, 60, 30, key="win3")
        z_thresh    = st.slider("Seuil z-score", 1.0, 3.0, 2.0, 0.1, key="zt3")
        params3     = {"window": window3, "z_threshold": z_thresh}
    with col3:
        commission3 = st.slider("Commission %", 0.0, 0.5, 0.1, 0.05, key="comm3") / 100
        slippage3   = st.slider("Slippage %", 0.0, 0.5, 0.1, 0.05, key="slip3") / 100

    if st.button("Lancer le backtest", type="primary", key="bt3"):
        with st.spinner("Calcul en cours..."):
            ticker_a, ticker_b = PAIRS[pair_choice]
            data_a = load_data(ticker_a, period3)
            data_b = load_data(ticker_b, period3)
            bench3 = load_data("^GSPC", period3)["Close"].squeeze()

            if data_a.empty or data_b.empty:
                st.error("Impossible de charger les donnees.")
            else:
                close_a = data_a["Close"].squeeze()
                close_b = data_b["Close"].squeeze()

                # Aligne les deux series sur les memes dates
                aligned = pd.concat([close_a, close_b], axis=1).dropna()
                aligned.columns = ["A", "B"]

                # Calcul du hedge ratio par regression et du spread
                from numpy.linalg import lstsq
                X           = aligned["B"].values.reshape(-1, 1)
                y           = aligned["A"].values
                hedge_ratio = float(lstsq(X, y, rcond=None)[0][0])
                spread      = pd.Series(
                    np.log(aligned["A"]) - hedge_ratio * np.log(aligned["B"]),
                    index=aligned.index
                )

                # Affiche la correlation et le spread
                correlation = aligned["A"].corr(aligned["B"])
                st.metric("Correlation historique", f"{correlation:.3f}")

                fig_spread = go.Figure()
                zscore_spread = (spread - spread.rolling(window3).mean()) / spread.rolling(window3).std()
                fig_spread.add_trace(go.Scatter(
                    x=spread.index, y=zscore_spread.values,
                    name="Z-score spread", line=dict(color="#00d4aa", width=1.5)
                ))
                for level, color in [(z_thresh,"#ff6b6b"),(-z_thresh,"#00d4aa"),(0,"white")]:
                    fig_spread.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.5)
                fig_spread.update_layout(
                    title=f"Z-score du spread — {pair_choice}",
                    template="plotly_dark", height=280,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_spread, use_container_width=True)

                # Backtest sur l actif A avec le signal du spread
                strat3 = get_strategy("Pairs Trading", params3)
                strat3.set_spread(spread)

                # Reconstruit data_a aligne
                data_a_aligned = data_a.loc[aligned.index]
                engine3 = BacktestEngine(
                    data_a_aligned, strat3, capital3,
                    commission3, slippage3, bench3
                )
                results3 = engine3.run()

                st.divider()
                display_metrics(
                    results3["metrics"],
                    results3["bh_return"],
                    results3["final_capital"],
                    capital3
                )

                bh_equity3 = close_a.loc[aligned.index] / float(close_a.loc[aligned.index].iloc[0]) * capital3
                st.plotly_chart(
                    plot_equity_curve(
                        results3["equity_curve"],
                        bench3 / float(bench3.iloc[0]) * capital3,
                        bh_equity3,
                        f"{pair_choice} — Pairs Trading"
                    ), use_container_width=True
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_drawdown(results3["equity_curve"]), use_container_width=True)
                with col_b:
                    st.plotly_chart(plot_monthly_returns(results3["returns"]), use_container_width=True)
