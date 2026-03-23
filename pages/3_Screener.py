import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Screener Rebond", page_icon="🎯", layout="wide")
st.title("🎯 Screener — Probabilité de Rebond")
st.caption("12 signaux · 8 techniques + 4 fondamentaux · CAC 40 & S&P 500")

CAC40 = {
    "Air Liquide": "AI.PA", "AXA": "CS.PA", "BNP Paribas": "BNP.PA",
    "Capgemini": "CAP.PA", "Carrefour": "CA.PA", "Danone": "BN.PA",
    "Engie": "ENGI.PA", "EssilorLuxottica": "EL.PA", "Hermes": "RMS.PA",
    "Kering": "KER.PA", "LOreal": "OR.PA", "Legrand": "LR.PA",
    "LVMH": "MC.PA", "Michelin": "ML.PA", "Orange": "ORA.PA",
    "Pernod Ricard": "RI.PA", "Publicis": "PUB.PA", "Renault": "RNO.PA",
    "Safran": "SAF.PA", "Saint-Gobain": "SGO.PA", "Sanofi": "SAN.PA",
    "Schneider Electric": "SU.PA", "Societe Generale": "GLE.PA",
    "Stellantis": "STLAM.PA", "STMicroelectronics": "STM.PA",
    "TotalEnergies": "TTE.PA", "Unibail-Rodamco": "URW.PA",
    "Veolia": "VIE.PA", "Vinci": "DG.PA", "Vivendi": "VIV.PA",
}

SP500_SAMPLE = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
    "Nvidia": "NVDA", "Alphabet": "GOOGL", "Meta": "META",
    "Tesla": "TSLA", "Berkshire": "BRK-B", "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ", "Visa": "V", "Exxon": "XOM",
    "UnitedHealth": "UNH", "Procter & Gamble": "PG", "Mastercard": "MA",
    "Home Depot": "HD", "Chevron": "CVX", "AbbVie": "ABBV",
    "Coca-Cola": "KO", "PepsiCo": "PEP", "Pfizer": "PFE",
    "Broadcom": "AVGO", "Bank of America": "BAC", "Netflix": "NFLX",
    "Adobe": "ADBE", "Salesforce": "CRM", "AMD": "AMD",
    "Costco": "COST", "Disney": "DIS", "Intel": "INTC",
}

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        pe       = info.get("trailingPE", None)
        de       = info.get("debtToEquity", None)
        rev_grow = info.get("revenueGrowth", None)
        margin   = info.get("operatingMargins", None)
        return pe, de, rev_grow, margin
    except Exception:
        return None, None, None, None

def score_pe(pe):
    if pe is None: return 50
    if pe < 10:    return 90
    if pe < 15:    return 75
    if pe < 20:    return 60
    if pe < 25:    return 45
    if pe < 35:    return 30
    return 10

def score_de(de):
    if de is None: return 50
    if de < 30:    return 90
    if de < 60:    return 70
    if de < 100:   return 50
    if de < 150:   return 30
    return 10

def score_rev_growth(g):
    if g is None:  return 50
    if g > 0.20:   return 90
    if g > 0.10:   return 75
    if g > 0.05:   return 60
    if g > 0:      return 45
    return 20

def score_margin(m):
    if m is None:  return 50
    if m > 0.30:   return 90
    if m > 0.20:   return 75
    if m > 0.10:   return 60
    if m > 0.05:   return 40
    return 15

def compute_signals(df, ticker):
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    signals = {}

    # 1. RSI (12%)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))
    last_rsi = float(rsi.iloc[-1])
    signals["RSI"] = {
        "value": round(last_rsi, 1),
        "score": max(0, min(100, (30 - last_rsi) * 5 + 50)) if last_rsi < 50 else 0,
        "label": f"RSI = {last_rsi:.1f}",
        "weight": 0.12,
        "info": "Oversold < 30 signal haussier",
        "categorie": "Technique",
    }

    # 2. Bollinger (12%)
    ma20     = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_lower = ma20 - 2 * std20
    bb_upper = ma20 + 2 * std20
    last_c   = float(close.iloc[-1])
    bb_pct   = (last_c - float(bb_lower.iloc[-1])) / (float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1]))
    signals["Bollinger"] = {
        "value": round(bb_pct * 100, 1),
        "score": max(0, min(100, (1 - bb_pct) * 100)),
        "label": f"Position BB = {bb_pct*100:.1f}%",
        "weight": 0.12,
        "info": "Proche bande basse compression haussiere",
        "categorie": "Technique",
    }

    # 3. Z-score (8%)
    returns = close.pct_change()
    mu      = returns.rolling(60).mean()
    sigma   = returns.rolling(60).std()
    zscore  = float(((returns - mu) / sigma).iloc[-1])
    signals["Z-score"] = {
        "value": round(zscore, 2),
        "score": max(0, min(100, (-zscore) * 25 + 50)),
        "label": f"Z-score = {zscore:.2f}",
        "weight": 0.08,
        "info": "Rendement anormalement bas mean reversion",
        "categorie": "Technique",
    }

    # 4. Volume capitulation (8%)
    vol_ma20  = volume.rolling(20).mean()
    vol_ratio = float(volume.iloc[-1]) / float(vol_ma20.iloc[-1])
    last_ret  = float(returns.iloc[-1])
    vol_score = min(100, vol_ratio * 50) if last_ret < 0 else 0
    signals["Volume"] = {
        "value": round(vol_ratio, 2),
        "score": vol_score,
        "label": f"Vol ratio = {vol_ratio:.2f}x",
        "weight": 0.08,
        "info": "Volume eleve sur baisse capitulation vendeurs",
        "categorie": "Technique",
    }

    # 5. MA200 (8%)
    ma200      = close.rolling(200).mean()
    dist_ma200 = (last_c - float(ma200.iloc[-1])) / float(ma200.iloc[-1]) * 100
    signals["MA200"] = {
        "value": round(dist_ma200, 2),
        "score": max(0, min(100, (-dist_ma200) * 5 + 50)),
        "label": f"Distance MA200 = {dist_ma200:.1f}%",
        "weight": 0.08,
        "info": "Tres sous MA200 retour vers moyenne long terme",
        "categorie": "Technique",
    }

    # 6. MACD (16%)
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    hist_line = macd_line - sig_line
    last_macd = float(macd_line.iloc[-1])
    last_sig  = float(sig_line.iloc[-1])
    last_hist = float(hist_line.iloc[-1])
    prev_hist = float(hist_line.iloc[-2])
    macd_score = 0
    if last_macd < 0:         macd_score += 40
    if last_hist > prev_hist: macd_score += 40
    if last_macd > last_sig:  macd_score += 20
    signals["MACD"] = {
        "value": round(last_macd, 3),
        "score": min(100, macd_score),
        "label": f"MACD={last_macd:.3f} Hist={last_hist:.3f}",
        "weight": 0.16,
        "info": "MACD negatif histogramme remonte golden cross imminent",
        "categorie": "Technique",
    }

    # 7. Stochastique (12%)
    low14   = low.rolling(14).min()
    high14  = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14)
    stoch_d = stoch_k.rolling(3).mean()
    last_k  = float(stoch_k.iloc[-1])
    last_d  = float(stoch_d.iloc[-1])
    st_score = 0
    if last_k < 20:     st_score += 60
    elif last_k < 30:   st_score += 30
    if last_k > last_d: st_score += 40
    signals["Stochastique"] = {
        "value": round(last_k, 1),
        "score": min(100, st_score),
        "label": f"%K={last_k:.1f} %D={last_d:.1f}",
        "weight": 0.12,
        "info": "K inferieur 20 croisement D oversold confirme",
        "categorie": "Technique",
    }

    # 8. ATR (4%)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14     = tr.rolling(14).mean()
    atr50     = tr.rolling(50).mean()
    atr_ratio = float(atr14.iloc[-1]) / float(atr50.iloc[-1])
    signals["ATR"] = {
        "value": round(atr_ratio, 2),
        "score": max(0, min(100, (1 - atr_ratio) * 100 + 50)),
        "label": f"ATR ratio = {atr_ratio:.2f}",
        "weight": 0.04,
        "info": "ATR comprime energie accumulee avant rebond",
        "categorie": "Technique",
    }

    # 9. P/E (10%)
    pe, de, rev_grow, margin = get_fundamentals(ticker)
    signals["P/E Ratio"] = {
        "value": round(pe, 1) if pe else None,
        "score": score_pe(pe),
        "label": f"P/E = {pe:.1f}" if pe else "P/E = N/A",
        "weight": 0.10,
        "info": "Valorisation attractive < 15x",
        "categorie": "Fondamental",
    }

    # 10. Dette/Equity (5%)
    signals["Dette/Equity"] = {
        "value": round(de, 1) if de else None,
        "score": score_de(de),
        "label": f"D/E = {de:.1f}%" if de else "D/E = N/A",
        "weight": 0.05,
        "info": "Bilan solide D/E < 60%",
        "categorie": "Fondamental",
    }

    # 11. Croissance CA (5%)
    signals["Croissance CA"] = {
        "value": round(rev_grow * 100, 1) if rev_grow else None,
        "score": score_rev_growth(rev_grow),
        "label": f"CA growth = {rev_grow*100:.1f}%" if rev_grow else "CA growth = N/A",
        "weight": 0.05,
        "info": "Croissance revenue > 10% signal positif",
        "categorie": "Fondamental",
    }

    # 12. Marge operationnelle (5%)
    signals["Marge Op."] = {
        "value": round(margin * 100, 1) if margin else None,
        "score": score_margin(margin),
        "label": f"Marge = {margin*100:.1f}%" if margin else "Marge = N/A",
        "weight": 0.05,
        "info": "Marge operationnelle > 20% business de qualite",
        "categorie": "Fondamental",
    }

    total_score = sum(s["score"] * s["weight"] for s in signals.values())
    return signals, round(total_score, 1)

@st.cache_data(ttl=3600)
def backtest_signal(ticker, score_threshold=50, horizons=[5, 10, 20]):
    try:
        df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if df.empty or len(df) < 250:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # Recalcule RSI et Stochastique sur tout l historique
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss))

        low14   = low.rolling(14).min()
        high14  = high.rolling(14).max()
        stoch_k = 100 * (close - low14) / (high14 - low14)

        ma20    = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        bb_low  = ma20 - 2 * std20

        # Signal simplifie : RSI < 35 ET Stoch < 25 ET prix proche BB basse
        signal = (rsi < 35) & (stoch_k < 25) & (close < (bb_low * 1.02))

        results = {}
        for h in horizons:
            fwd_return = close.shift(-h) / close - 1
            signal_returns = fwd_return[signal]
            signal_returns = signal_returns.dropna()
            if len(signal_returns) == 0:
                results[h] = {"n": 0, "win_rate": None, "avg_return": None}
            else:
                win_rate   = (signal_returns > 0).mean() * 100
                avg_return = signal_returns.mean() * 100
                results[h] = {
                    "n":          len(signal_returns),
                    "win_rate":   round(win_rate, 1),
                    "avg_return": round(avg_return, 2),
                }
        return results
    except Exception:
        return None

def run_screener(universe):
    rows     = []
    tickers  = list(universe.items())
    progress = st.progress(0, text="Analyse en cours...")
    for i, (name, ticker) in enumerate(tickers):
        try:
            df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            signals, score = compute_signals(df, ticker)
            last_c = float(df["Close"].squeeze().iloc[-1])
            prev_c = float(df["Close"].squeeze().iloc[-2])
            chg    = (last_c - prev_c) / prev_c * 100
            rows.append({
                "Action":       name,
                "Ticker":       ticker,
                "Prix":         round(last_c, 2),
                "Var. %":       round(chg, 2),
                "Score":        score,
                "RSI":          signals["RSI"]["value"],
                "Stoch %K":     signals["Stochastique"]["value"],
                "MACD":         signals["MACD"]["value"],
                "BB %":         signals["Bollinger"]["value"],
                "Z-score":      signals["Z-score"]["value"],
                "P/E":          signals["P/E Ratio"]["value"],
                "D/E %":        signals["Dette/Equity"]["value"],
                "CA Growth %":  signals["Croissance CA"]["value"],
                "Marge Op. %":  signals["Marge Op."]["value"],
            })
        except Exception:
            continue
        progress.progress((i + 1) / len(tickers), text=f"Analyse : {name}...")
    progress.empty()
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Score", ascending=False)
    return df_out

# ── Session state ─────────────────────────────────────────
if "df_screen" not in st.session_state:
    st.session_state.df_screen = None
if "selected_action" not in st.session_state:
    st.session_state.selected_action = None

col1, col2 = st.columns([2, 1])
with col1:
    universe_choice = st.radio("Univers", ["CAC 40", "S&P 500 (Top 30)"], horizontal=True)
with col2:
    min_score = st.slider("Score minimum", 0, 100, 40)

universe = CAC40 if universe_choice == "CAC 40" else SP500_SAMPLE

if st.button("Lancer le screener", type="primary"):
    st.session_state.df_screen = run_screener(universe)
    st.session_state.selected_action = None

if st.session_state.df_screen is not None:
    df_screen   = st.session_state.df_screen
    df_filtered = df_screen[df_screen["Score"] >= min_score]

    st.divider()
    st.subheader(f"🏆 {len(df_filtered)} actions detectees")

    def color_score(val):
        if val >= 70:   return "background-color:rgba(0,212,170,0.2);color:#00d4aa"
        elif val >= 50: return "background-color:rgba(245,166,35,0.2);color:#f5a623"
        else:           return "background-color:rgba(255,107,107,0.1);color:#ff6b6b"

    def color_var(val):
        return "color:#00d4aa" if val >= 0 else "color:#ff6b6b"

    styled = df_filtered.style \
        .applymap(color_score, subset=["Score"]) \
        .applymap(color_var,   subset=["Var. %"]) \
        .format({
            "Prix":        "{:.2f}",
            "Var. %":      "{:+.2f}%",
            "Score":       "{:.0f}/100",
            "RSI":         "{:.1f}",
            "Stoch %K":    "{:.1f}",
            "MACD":        "{:.3f}",
            "BB %":        "{:.1f}%",
            "Z-score":     "{:.2f}",
            "P/E":         lambda v: f"{v:.1f}x" if v else "N/A",
            "D/E %":       lambda v: f"{v:.1f}%" if v else "N/A",
            "CA Growth %": lambda v: f"{v:+.1f}%" if v else "N/A",
            "Marge Op. %": lambda v: f"{v:.1f}%" if v else "N/A",
        })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Analyse détaillée ──────────────────────────────────
    st.divider()
    tab1, tab2 = st.tabs(["🔬 Analyse signaux", "📈 Backtest historique"])

    actions = df_filtered["Action"].tolist()
    idx = 0
    if st.session_state.selected_action in actions:
        idx = actions.index(st.session_state.selected_action)

    selected = st.selectbox("Choisir une action", actions, index=idx, key="action_select")
    st.session_state.selected_action = selected

    if selected:
        ticker = df_filtered[df_filtered["Action"] == selected]["Ticker"].values[0]
        df_d   = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if isinstance(df_d.columns, pd.MultiIndex):
            df_d.columns = df_d.columns.get_level_values(0)
        signals, score = compute_signals(df_d, ticker)

        # ── Tab 1 : Signaux ───────────────────────────────
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                score_color = "#00d4aa" if score >= 70 else "#f5a623" if score >= 50 else "#ff6b6b"
                st.markdown(f"""
                <div style='text-align:center;padding:20px;background:#111620;
                            border:1px solid #1e2736;border-radius:8px;margin-bottom:16px'>
                    <div style='font-size:48px;font-weight:800;color:{score_color}'>{score}</div>
                    <div style='font-size:12px;color:#9aa3b5;letter-spacing:0.1em'>
                        SCORE DE REBOND / 100
                    </div>
                </div>
                """, unsafe_allow_html=True)

                for categorie, label in [("Technique", "Signaux Techniques"), ("Fondamental", "Signaux Fondamentaux")]:
                    st.markdown(f"**{label}**")
                    for sig_name, sig in signals.items():
                        if sig["categorie"] != categorie:
                            continue
                        bar_color = "#00d4aa" if sig["score"] >= 60 else "#f5a623" if sig["score"] >= 35 else "#ff6b6b"
                        st.markdown(f"""
                        <div style='margin-bottom:10px'>
                            <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                                <span style='font-size:12px;color:#e8eaf0'>{sig_name}</span>
                                <span style='font-size:11px;color:{bar_color}'>{sig["score"]:.0f}/100</span>
                            </div>
                            <div style='background:#1e2736;border-radius:3px;height:6px;width:100%'>
                                <div style='background:{bar_color};border-radius:3px;height:6px;width:{sig["score"]}%'></div>
                            </div>
                            <div style='font-size:10px;color:#5a6478;margin-top:2px'>
                                {sig["label"]} · {sig["info"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            with col2:
                close  = df_d["Close"].squeeze()
                high   = df_d["High"].squeeze()
                low    = df_d["Low"].squeeze()
                ma20   = close.rolling(20).mean()
                ma200  = close.rolling(200).mean()
                std20  = close.rolling(20).std()
                bb_up  = ma20 + 2 * std20
                bb_low = ma20 - 2 * std20
                ema12  = close.ewm(span=12, adjust=False).mean()
                ema26  = close.ewm(span=26, adjust=False).mean()
                macd   = ema12 - ema26
                sig_l  = macd.ewm(span=9, adjust=False).mean()
                hist   = macd - sig_l
                low14  = low.rolling(14).min()
                high14 = high.rolling(14).max()
                stoch  = 100 * (close - low14) / (high14 - low14)

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.23, 0.22],
                    vertical_spacing=0.02,
                    subplot_titles=("Prix", "MACD", "Stochastique")
                )
                fig.add_trace(go.Candlestick(
                    x=df_d.index,
                    open=df_d["Open"], high=df_d["High"],
                    low=df_d["Low"], close=df_d["Close"],
                    name="Prix",
                    increasing_line_color="#00d4aa",
                    decreasing_line_color="#ff6b6b"
                ), row=1, col=1)
                for y, name_l, color, dash in [
                    (ma20,  "MA20",  "#f5a623",               "solid"),
                    (ma200, "MA200", "white",                  "dot"),
                    (bb_up, "BB+",   "rgba(108,142,191,0.5)", "dash"),
                    (bb_low,"BB-",   "rgba(108,142,191,0.5)", "dash"),
                ]:
                    fig.add_trace(go.Scatter(
                        x=df_d.index, y=y, name=name_l,
                        line=dict(color=color, width=1.2, dash=dash)
                    ), row=1, col=1)
                hist_colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in hist.values]
                fig.add_trace(go.Bar(
                    x=df_d.index, y=hist, name="Histogramme",
                    marker_color=hist_colors, showlegend=False
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df_d.index, y=macd, name="MACD",
                    line=dict(color="#00d4aa", width=1.5)
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df_d.index, y=sig_l, name="Signal",
                    line=dict(color="#f5a623", width=1.5)
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df_d.index, y=stoch, name="%K",
                    line=dict(color="#c084fc", width=1.5)
                ), row=3, col=1)
                for level, color in [(80, "#ff6b6b"), (20, "#00d4aa")]:
                    fig.add_hline(y=level, line_dash="dot",
                                  line_color=color, opacity=0.5, row=3, col=1)
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab 2 : Backtest ──────────────────────────────
        with tab2:
            st.markdown("#### Taux de succes historique du signal")
            st.caption("Sur les 2 dernières années · Signal declenche quand RSI < 35 + Stoch < 25 + prix proche BB basse")

            with st.spinner("Calcul du backtest..."):
                bt = backtest_signal(ticker)

            if bt:
                cols = st.columns(3)
                for i, (h, label) in enumerate([(5,"5 jours"),(10,"10 jours"),(20,"20 jours")]):
                    res = bt[h]
                    with cols[i]:
                        if res["win_rate"] is not None:
                            wr_color = "#00d4aa" if res["win_rate"] >= 55 else "#f5a623" if res["win_rate"] >= 45 else "#ff6b6b"
                            rt_color = "#00d4aa" if res["avg_return"] >= 0 else "#ff6b6b"
                            st.markdown(f"""
                            <div style='background:#111620;border:1px solid #1e2736;
                                        border-radius:8px;padding:16px;text-align:center'>
                                <div style='font-size:11px;color:#9aa3b5;margin-bottom:8px'>
                                    HORIZON {label.upper()}
                                </div>
                                <div style='font-size:32px;font-weight:800;color:{wr_color}'>
                                    {res["win_rate"]:.1f}%
                                </div>
                                <div style='font-size:11px;color:#9aa3b5;margin-bottom:4px'>
                                    Taux de succes
                                </div>
                                <div style='font-size:16px;font-weight:600;color:{rt_color}'>
                                    {res["avg_return"]:+.2f}%
                                </div>
                                <div style='font-size:11px;color:#9aa3b5'>
                                    Rendement moyen
                                </div>
                                <div style='font-size:10px;color:#5a6478;margin-top:8px'>
                                    {res["n"]} signaux detectes
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info(f"Pas assez de signaux sur {label}")

                # Distribution des rendements
                st.markdown("#### Distribution des rendements post-signal (10 jours)")
                try:
                    df_bt = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
                    if isinstance(df_bt.columns, pd.MultiIndex):
                        df_bt.columns = df_bt.columns.get_level_values(0)
                    close_bt = df_bt["Close"].squeeze()
                    high_bt  = df_bt["High"].squeeze()
                    low_bt   = df_bt["Low"].squeeze()
                    delta_bt = close_bt.diff()
                    gain_bt  = delta_bt.clip(lower=0).rolling(14).mean()
                    loss_bt  = (-delta_bt.clip(upper=0)).rolling(14).mean()
                    rsi_bt   = 100 - (100 / (1 + gain_bt / loss_bt))
                    low14_bt  = low_bt.rolling(14).min()
                    high14_bt = high_bt.rolling(14).max()
                    stoch_bt  = 100 * (close_bt - low14_bt) / (high14_bt - low14_bt)
                    ma20_bt   = close_bt.rolling(20).mean()
                    std20_bt  = close_bt.rolling(20).std()
                    bb_low_bt = ma20_bt - 2 * std20_bt
                    signal_bt = (rsi_bt < 35) & (stoch_bt < 25) & (close_bt < (bb_low_bt * 1.02))
                    fwd_10    = (close_bt.shift(-10) / close_bt - 1) * 100
                    rets      = fwd_10[signal_bt].dropna()

                    if len(rets) > 2:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=rets.values,
                            nbinsx=20,
                            marker_color=["#00d4aa" if v >= 0 else "#ff6b6b" for v in rets.values],
                            name="Rendements"
                        ))
                        fig_hist.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.7)
                        fig_hist.add_vline(
                            x=float(rets.mean()), line_dash="dash",
                            line_color="#f5a623", opacity=0.8,
                            annotation_text=f"Moy: {rets.mean():+.1f}%"
                        )
                        fig_hist.update_layout(
                            template="plotly_dark",
                            xaxis_title="Rendement % a 10 jours",
                            yaxis_title="Frequence",
                            height=300,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                except Exception:
                    pass
            else:
                st.warning("Donnees insuffisantes pour le backtest (2 ans minimum requis).")
