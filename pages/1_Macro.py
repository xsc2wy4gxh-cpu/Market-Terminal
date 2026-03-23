import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Macro", page_icon="🏦", layout="wide")
st.title("🏦 Tableau de bord Macro")
st.caption("Taux US · Taux EU · Taux JP · VIX · Spreads")

RATES_US = {
    "2 ans":  "^IRX",
    "5 ans":  "^FVX",
    "10 ans": "^TNX",
    "30 ans": "^TYX",
}

FRED_EU = {
    "Allemagne 10 ans": "IRLTLT01DEM156N",
    "France 10 ans":    "IRLTLT01FRM156N",
    "Italie 10 ans":    "IRLTLT01ITM156N",
}

FRED_JP = {
    "Japon 10 ans": "IRLTLT01JPM156N",
}

FRED_10Y = {
    "Allemagne 10 ans": "IRLTLT01DEM156N",
    "France 10 ans":    "IRLTLT01FRM156N",
    "Japon 10 ans":     "IRLTLT01JPM156N",
}

def fetch_close_yf(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].squeeze()

@st.cache_data(ttl=86400)
def fetch_fred(series_id, period):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df = df[df.iloc[:, 0] != "."]
        s  = df.iloc[:, 0].astype(float)
        period_days = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        days   = period_days.get(period, 365)
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
        return s[s.index >= cutoff]
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def get_rates_yf(period):
    dfs = {}
    for name, ticker in RATES_US.items():
        s = fetch_close_yf(ticker, period)
        if not s.empty:
            dfs[name] = s
    return pd.DataFrame(dfs)

@st.cache_data(ttl=86400)
def get_rates_fred(series_dict, period):
    dfs = {}
    for name, sid in series_dict.items():
        s = fetch_fred(sid, period)
        if not s.empty:
            dfs[name] = s
    return pd.DataFrame(dfs)

@st.cache_data(ttl=3600)
def get_vix(period):
    return fetch_close_yf("^VIX", period)

def rate_chart(df, colors, title, height=400):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(color=colors.get(col, "white"), width=2)
        ))
    fig.update_layout(
        title=title, template="plotly_dark",
        yaxis_title="Taux (%)", height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified"
    )
    return fig

def spread_chart(s, label, height=280):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, fill="tozeroy",
        line=dict(color="#00d4aa", width=1.5),
        fillcolor="rgba(0,212,170,0.1)", name=label
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#ff6b6b", opacity=0.7)
    fig.update_layout(
        template="plotly_dark", yaxis_title="Spread (%)",
        height=height, margin=dict(l=0, r=0, t=20, b=0)
    )
    return fig

period = st.radio("Periode", ["3mo","6mo","1y","2y"], index=2, horizontal=True)

# ── TAUX US ───────────────────────────────────────────────
st.divider()
st.subheader("🇺🇸 Taux US")

cols = st.columns(4)
for i, (name, ticker) in enumerate(RATES_US.items()):
    try:
        info  = yf.Ticker(ticker).fast_info
        price = info.last_price
        prev  = info.previous_close
        chg   = price - prev
        cols[i].metric(label=f"US {name}", value=f"{price:.2f}%", delta=f"{chg:+.2f} bps")
    except Exception:
        cols[i].metric(label=f"US {name}", value="N/A")

df_us = get_rates_yf(period)
if not df_us.empty:
    colors_us = {"2 ans":"#ff6b6b","5 ans":"#f5a623","10 ans":"#00d4aa","30 ans":"#6c8ebf"}
    st.plotly_chart(rate_chart(df_us, colors_us, "Courbe des taux US"), use_container_width=True)

if "2 ans" in df_us.columns and "10 ans" in df_us.columns:
    st.markdown("**Spread 10Y - 2Y (US)**")
    st.caption("Spread negatif = courbe inversee = signal de recession historique")
    st.plotly_chart(spread_chart(df_us["10 ans"] - df_us["2 ans"], "Spread 10Y-2Y US"), use_container_width=True)

# ── TAUX EUROPEENS ────────────────────────────────────────
st.divider()
st.subheader("🇪🇺 Taux Europeens — Source FRED")

df_eu = get_rates_fred(FRED_EU, period)
cols_eu = st.columns(3)
for i, (name, sid) in enumerate(FRED_EU.items()):
    try:
        s     = fetch_fred(sid, "3mo")
        last  = float(s.iloc[-1])
        prev  = float(s.iloc[-2])
        cols_eu[i].metric(label=name, value=f"{last:.2f}%", delta=f"{last-prev:+.2f} bps")
    except Exception:
        cols_eu[i].metric(label=name, value="N/A")

if not df_eu.empty:
    colors_eu = {
        "Allemagne 10 ans": "#00d4aa",
        "France 10 ans":    "#6c8ebf",
        "Italie 10 ans":    "#f5a623"
    }
    st.plotly_chart(rate_chart(df_eu, colors_eu, "Taux souverains europeens 10 ans"), use_container_width=True)

if "Allemagne 10 ans" in df_eu.columns and "Italie 10 ans" in df_eu.columns:
    st.markdown("**Spread BTP - Bund (Italie 10Y - Allemagne 10Y)**")
    st.caption("Barometre du risque souverain en zone euro · Seuil attention : 200 bps")
    fig_btp = spread_chart(df_eu["Italie 10 ans"] - df_eu["Allemagne 10 ans"], "Spread BTP-Bund")
    fig_btp.add_hline(y=2.0, line_dash="dot", line_color="#f5a623", opacity=0.7, annotation_text="Seuil 200 bps")
    st.plotly_chart(fig_btp, use_container_width=True)

# ── TAUX JAPONAIS ─────────────────────────────────────────
st.divider()
st.subheader("🇯🇵 Taux Japonais — Source FRED")
st.caption("La BoJ a abandonne le Yield Curve Control en 2024")

df_jp = get_rates_fred(FRED_JP, period)
cols_jp = st.columns(4)
try:
    s_jp  = fetch_fred("IRLTLT01JPM156N", "3mo")
    last  = float(s_jp.iloc[-1])
    prev  = float(s_jp.iloc[-2])
    cols_jp[0].metric(label="Japon 10 ans", value=f"{last:.2f}%", delta=f"{last-prev:+.2f} bps")
except Exception:
    cols_jp[0].metric(label="Japon 10 ans", value="N/A")

if not df_jp.empty:
    st.plotly_chart(rate_chart(df_jp, {"Japon 10 ans": "#00d4aa"}, "Taux japonais 10 ans (JGB)"), use_container_width=True)

# ── COMPARAISON INTERNATIONALE ────────────────────────────
st.divider()
st.subheader("🌍 Comparaison internationale — Taux 10 ans")

us_10y   = fetch_close_yf("^TNX", period)
df_intl  = pd.DataFrame()
if not us_10y.empty:
    df_intl["US 10 ans"] = us_10y

df_fred_10y = get_rates_fred(FRED_10Y, period)
for col in df_fred_10y.columns:
    df_intl[col] = df_fred_10y[col]

if not df_intl.empty:
    colors_intl = {
        "US 10 ans":        "#00d4aa",
        "Allemagne 10 ans": "#f5a623",
        "France 10 ans":    "#6c8ebf",
        "Japon 10 ans":     "#ff6b6b"
    }
    st.plotly_chart(rate_chart(df_intl, colors_intl, "Taux 10 ans - Comparaison internationale", height=450), use_container_width=True)

# ── VIX ───────────────────────────────────────────────────
st.divider()
st.subheader("😰 VIX — Volatilite implicite S&P 500")
st.caption("VIX > 30 = stress · VIX 20-30 = vigilance · VIX < 20 = calme")

vix = get_vix(period)
if not vix.empty:
    current_vix = float(vix.iloc[-1])
    prev_vix    = float(vix.iloc[-2])
    color = "#ff6b6b" if current_vix > 30 else "#f5a623" if current_vix > 20 else "#00d4aa"

    st.metric("VIX actuel", f"{current_vix:.1f}", delta=f"{current_vix - prev_vix:+.1f}")

    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(
        x=vix.index, y=vix.values, fill="tozeroy",
        line=dict(color=color, width=1.5),
        fillcolor="rgba(0,212,170,0.08)", name="VIX"
    ))
    for level, col, label in [(30,"#ff6b6b","Stress"), (20,"#f5a623","Vigilance")]:
        fig_vix.add_hline(y=level, line_dash="dot", line_color=col, opacity=0.6, annotation_text=label)
    fig_vix.update_layout(
        template="plotly_dark", yaxis_title="VIX",
        height=300, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_vix, use_container_width=True)
else:
    st.warning("Impossible de charger le VIX.")
