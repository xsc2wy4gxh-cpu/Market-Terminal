import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Heatmap Sectorielle", page_icon="🟩", layout="wide")
st.title("🟩 Heatmap Sectorielle")
st.caption("Performance par secteur et par action · S&P 500 & CAC 40")

SECTORS_SP500 = {
    "Technologie": {
        "ETF": "XLK",
        "Actions": {"Apple":"AAPL","Microsoft":"MSFT","Nvidia":"NVDA","AMD":"AMD","Broadcom":"AVGO","Salesforce":"CRM","Adobe":"ADBE","Intel":"INTC"}
    },
    "Finance": {
        "ETF": "XLF",
        "Actions": {"JPMorgan":"JPM","Bank of America":"BAC","Visa":"V","Mastercard":"MA","Goldman Sachs":"GS","Morgan Stanley":"MS","Wells Fargo":"WFC","BlackRock":"BLK"}
    },
    "Sante": {
        "ETF": "XLV",
        "Actions": {"J&J":"JNJ","UnitedHealth":"UNH","AbbVie":"ABBV","Pfizer":"PFE","Merck":"MRK","Eli Lilly":"LLY","Bristol-Myers":"BMY","Medtronic":"MDT"}
    },
    "Consommation": {
        "ETF": "XLY",
        "Actions": {"Amazon":"AMZN","Tesla":"TSLA","Home Depot":"HD","Nike":"NKE","McDonald's":"MCD","Starbucks":"SBUX","Booking":"BKNG","TJX":"TJX"}
    },
    "Energie": {
        "ETF": "XLE",
        "Actions": {"Exxon":"XOM","Chevron":"CVX","ConocoPhillips":"COP","EOG":"EOG","Schlumberger":"SLB","Marathon":"MPC","Valero":"VLO","Hess":"HES"}
    },
    "Industrie": {
        "ETF": "XLI",
        "Actions": {"Boeing":"BA","Caterpillar":"CAT","Honeywell":"HON","Union Pacific":"UNP","RTX":"RTX","Deere":"DE","3M":"MMM","GE":"GE"}
    },
    "Utilities": {
        "ETF": "XLU",
        "Actions": {"NextEra":"NEE","Duke Energy":"DUK","Southern Co":"SO","Dominion":"D","Sempra":"SRE","AEP":"AEP","Exelon":"EXC","PG&E":"PCG"}
    },
    "Immobilier": {
        "ETF": "XLRE",
        "Actions": {"Prologis":"PLD","American Tower":"AMT","Equinix":"EQIX","Crown Castle":"CCI","Welltower":"WELL","AvalonBay":"AVB","Essex":"ESS","SBA Comm":"SBAC"}
    },
}

SECTORS_CAC40 = {
    "Luxe": {
        "ETF": "MC.PA",
        "Actions": {"LVMH":"MC.PA","Hermes":"RMS.PA","Kering":"KER.PA","EssilorLuxottica":"EL.PA"}
    },
    "Energie": {
        "ETF": "TTE.PA",
        "Actions": {"TotalEnergies":"TTE.PA","Engie":"ENGI.PA"}
    },
    "Finance": {
        "ETF": "BNP.PA",
        "Actions": {"BNP Paribas":"BNP.PA","AXA":"CS.PA","Societe Generale":"GLE.PA"}
    },
    "Industrie": {
        "ETF": "SAF.PA",
        "Actions": {"Safran":"SAF.PA","Vinci":"DG.PA","Schneider":"SU.PA","Saint-Gobain":"SGO.PA","Legrand":"LR.PA"}
    },
    "Sante": {
        "ETF": "SAN.PA",
        "Actions": {"Sanofi":"SAN.PA"}
    },
    "Consommation": {
        "ETF": "BN.PA",
        "Actions": {"Danone":"BN.PA","Carrefour":"CA.PA","Pernod Ricard":"RI.PA","Renault":"RNO.PA"}
    },
    "Technologie": {
        "ETF": "CAP.PA",
        "Actions": {"Capgemini":"CAP.PA","STMicroelectronics":"STM.PA"}
    },
    "Telecom": {
        "ETF": "ORA.PA",
        "Actions": {"Orange":"ORA.PA","Vivendi":"VIV.PA"}
    },
    "Materiaux": {
        "ETF": "AI.PA",
        "Actions": {"Air Liquide":"AI.PA","Michelin":"ML.PA"}
    },
    "Immobilier": {
        "ETF": "URW.PA",
        "Actions": {"Unibail-Rodamco":"URW.PA"}
    },
}

@st.cache_data(ttl=3600)
def get_performance(ticker, period="5d"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].squeeze()
        perf  = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        return round(float(perf), 2)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_sector_perfs(sectors, period):
    rows = []
    for sector, data in sectors.items():
        perf = get_performance(data["ETF"], period)
        rows.append({"Secteur": sector, "Perf %": perf})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def load_stock_perfs(sectors, sector_name, period):
    actions = sectors[sector_name]["Actions"]
    rows    = []
    for name, ticker in actions.items():
        perf = get_performance(ticker, period)
        rows.append({"Action": name, "Ticker": ticker, "Perf %": perf})
    return pd.DataFrame(rows)

def build_treemap(labels, values, title, height=400):
    max_abs = max(abs(v) for v in values if v) or 1
    colors, texts = [], []
    for v in values:
        v = v or 0
        intensity = abs(v) / max_abs
        if v >= 0:
            r = int(0   + (1 - intensity) * 50)
            g = int(150 + intensity * 105)
            b = int(0   + (1 - intensity) * 50)
        else:
            r = int(150 + intensity * 105)
            g = int(0   + (1 - intensity) * 50)
            b = int(0   + (1 - intensity) * 50)
        colors.append(f"rgb({r},{g},{b})")
        texts.append(f"{v:+.2f}%")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=[abs(v or 0) + 2 for v in values],
        text=texts,
        textinfo="label+text",
        marker=dict(colors=colors, line=dict(width=2, color="#0a0e14")),
        hovertemplate="<b>%{label}</b><br>Performance : %{text}<extra></extra>",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=0, r=0, t=30, b=0), title=title
    )
    return fig

def build_bar(df, height=350):
    df_sorted   = df.sort_values("Perf %", ascending=True).dropna()
    bar_colors  = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in df_sorted["Perf %"]]
    fig = go.Figure(go.Bar(
        x=df_sorted["Perf %"],
        y=df_sorted["Action"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.2f}%" for v in df_sorted["Perf %"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.5)
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Performance %",
    )
    return fig

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🇺🇸 S&P 500", "🇫🇷 CAC 40"])

period_labels = {"1 jour": "2d", "1 semaine": "5d", "1 mois": "1mo", "3 mois": "3mo"}

for tab, sectors, index_ticker, index_name in [
    (tab1, SECTORS_SP500, "^GSPC", "S&P 500"),
    (tab2, SECTORS_CAC40, "^FCHI", "CAC 40"),
]:
    with tab:
        period_choice = st.radio(
            "Periode", list(period_labels.keys()),
            index=1, horizontal=True,
            key=f"period_{index_name}"
        )
        period = period_labels[period_choice]

        # Performance indice
        perf_index = get_performance(index_ticker, period)
        if perf_index is not None:
            color = "#00d4aa" if perf_index >= 0 else "#ff6b6b"
            st.markdown(f"""
            <div style='display:inline-block;padding:8px 20px;
                        background:#111620;border:1px solid #1e2736;
                        border-radius:4px;margin-bottom:16px'>
                <span style='color:#9aa3b5;font-size:12px'>{index_name} · {period_choice} &nbsp;</span>
                <span style='color:{color};font-size:18px;font-weight:700'>
                    {perf_index:+.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Heatmap secteurs
        st.subheader("Performance par secteur")
        with st.spinner("Chargement..."):
            df_sec = load_sector_perfs(sectors, period)

        if not df_sec.empty:
            fig_tree = build_treemap(
                df_sec["Secteur"].tolist(),
                df_sec["Perf %"].tolist(),
                f"Secteurs {index_name} — {period_choice}",
                height=380
            )
            st.plotly_chart(fig_tree, use_container_width=True)

            def color_perf(val):
                if val is None: return "color:#9aa3b5"
                return "color:#00d4aa" if val >= 0 else "color:#ff6b6b"

            styled = df_sec.style \
                .applymap(color_perf, subset=["Perf %"]) \
                .format({"Perf %": lambda v: f"{v:+.2f}%" if v else "N/A"})
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Drill-down
        st.divider()
        st.subheader("Drill-down par secteur")
        selected = st.selectbox(
            "Choisir un secteur", list(sectors.keys()),
            key=f"sector_{index_name}"
        )
        with st.spinner(f"Chargement {selected}..."):
            df_stocks = load_stock_perfs(sectors, selected, period)

        if not df_stocks.empty:
            fig_tree2 = build_treemap(
                df_stocks["Action"].tolist(),
                df_stocks["Perf %"].tolist(),
                f"{selected} — Actions individuelles",
                height=320
            )
            st.plotly_chart(fig_tree2, use_container_width=True)
            st.plotly_chart(build_bar(df_stocks), use_container_width=True)
