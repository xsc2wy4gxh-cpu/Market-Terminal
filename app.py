import streamlit as st
from data.fetcher import get_snapshot, get_price_history, INDICES, COMMODITIES
from data.indicators import add_moving_averages, add_rsi, add_bollinger
from components.charts import candlestick_with_indicators

# ── Config page ──────────────────────────────────────────
st.set_page_config(
    page_title="Market Terminal",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: #111620;
        border: 1px solid #1e2736;
        border-radius: 4px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.title("📊 Market Terminal")
st.caption("Données via yfinance · Actualisation manuelle")

# ── Snapshot ─────────────────────────────────────────────
st.subheader("Vue d'ensemble")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Indices**")
    df_idx = get_snapshot(INDICES)
    for _, row in df_idx.iterrows():
        st.metric(
            label=row["Nom"],
            value=f"{row['Prix']:,.0f}" if row["Prix"] else "N/A",
            delta=f"{row['Var. %']:+.2f}%" if row["Var. %"] else None,
        )

with col2:
    st.markdown("**Matières premières**")
    df_com = get_snapshot(COMMODITIES)
    for _, row in df_com.iterrows():
        st.metric(
            label=row["Nom"],
            value=f"{row['Prix']:,.2f}" if row["Prix"] else "N/A",
            delta=f"{row['Var. %']:+.2f}%" if row["Var. %"] else None,
        )

# ── Chart détaillé ───────────────────────────────────────
st.divider()
st.subheader("Analyse graphique")

all_assets = {**INDICES, **COMMODITIES}
selected = st.selectbox("Choisir un actif", list(all_assets.keys()))
period   = st.radio("Période", ["1mo","3mo","6mo","1y","2y"],
                    index=2, horizontal=True)

df = get_price_history(all_assets[selected], period=period)
df = add_moving_averages(df)
df = add_rsi(df)
df = add_bollinger(df)

fig = candlestick_with_indicators(df, selected)
st.plotly_chart(fig, use_container_width=True)