import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="Actualités & Calendrier", page_icon="📰", layout="wide")

st.title("📰 Actualités & Calendrier Économique")
st.caption("News marchés en temps réel · Événements macro à venir")

# ════════════════════════════════════════════════════════
# SECTION 1 — ACTUALITÉS MARCHÉS
# ════════════════════════════════════════════════════════
st.subheader("📡 Actualités Marchés")
st.caption("Source : GNews API (gratuite)")

GNEWS_API_KEY = "682d42c9f8564770f2fd64ba21821327"  # On va configurer ça juste après

TOPICS = {
    "Marchés financiers": "financial markets",
    "Banques centrales":  "central bank fed ecb",
    "Économie US":        "us economy",
    "Matières premières": "commodities gold oil",
}

@st.cache_data(ttl=900)  # Rafraîchissement toutes les 15 min
def get_news(query: str, api_key: str, max_articles: int = 6) -> list:
    url = (
        f"https://gnews.io/api/v4/search"
        f"?q={query}&lang=en&max={max_articles}"
        f"&sortby=publishedAt&token={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []

topic = st.selectbox("Thème", list(TOPICS.keys()))

if GNEWS_API_KEY == "VOTRE_CLE_API":
    st.warning("⚠️ Configure ta clé API GNews — voir les instructions ci-dessous.")
else:
    articles = get_news(TOPICS[topic], GNEWS_API_KEY)
    if articles:
        for article in articles:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.caption(
                        f"🕐 {article['publishedAt'][:10]}  ·  "
                        f"📰 {article['source']['name']}"
                    )
                    if article.get("description"):
                        st.markdown(
                            f"<p style='color:#9aa3b5;font-size:13px'>"
                            f"{article['description'][:200]}...</p>",
                            unsafe_allow_html=True
                        )
                with col2:
                    if article.get("image"):
                        st.image(article["image"], use_column_width=True)
                st.divider()
    else:
        st.info("Aucun article trouvé.")

# ════════════════════════════════════════════════════════
# SECTION 2 — CALENDRIER ÉCONOMIQUE
# ════════════════════════════════════════════════════════
st.divider()
st.subheader("📅 Calendrier Économique")
st.caption("Source : Investing.com · Événements à fort impact")

@st.cache_data(ttl=3600)
def get_economic_calendar() -> pd.DataFrame:
    """Calendrier économique via l'API publique d'Investing.com."""
    url = "https://economic-calendar.tradingview.com/events"
    now   = datetime.utcnow()
    start = now.strftime("%Y-%m-%dT00:00:00.000Z")
    end   = (now + timedelta(days=14)).strftime("%Y-%m-%dT23:59:59.000Z")

    payload = {
        "from":       start,
        "to":         end,
        "countries":  ["US", "EU", "JP", "GB", "DE", "FR"],
        "importance": [2, 3],  # 2 = medium, 3 = high
    }

    headers = {
        "Content-Type":  "application/x-www-form-urlencoded",
        "User-Agent":    "Mozilla/5.0",
        "Origin":        "https://www.tradingview.com",
        "Referer":       "https://www.tradingview.com/",
    }

    try:
        r = requests.post(url, data=payload, headers=headers, timeout=10)
        events = r.json().get("result", [])
        rows = []
        for e in events:
            rows.append({
                "Date":       e.get("date", "")[:10],
                "Heure":      e.get("date", "")[11:16] + " UTC",
                "Pays":       e.get("country", ""),
                "Événement":  e.get("title", ""),
                "Impact":     "🔴 Fort" if e.get("importance") == 3 else "🟡 Moyen",
                "Précédent":  e.get("previous", "—"),
                "Consensus":  e.get("forecast", "—"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# Filtres
col1, col2 = st.columns(2)
with col1:
    impact_filter = st.multiselect(
        "Impact", ["🔴 Fort", "🟡 Moyen"],
        default=["🔴 Fort"]
    )
with col2:
    country_filter = st.multiselect(
        "Pays", ["US", "EU", "JP", "GB", "DE", "FR"],
        default=["US", "EU", "JP"]
    )

df_cal = get_economic_calendar()

if not df_cal.empty:
    # Applique les filtres
    if impact_filter:
        df_cal = df_cal[df_cal["Impact"].isin(impact_filter)]
    if country_filter:
        df_cal = df_cal[df_cal["Pays"].isin(country_filter)]

    if not df_cal.empty:
        # Affiche par date
        for date in df_cal["Date"].unique():
            st.markdown(f"**📆 {date}**")
            df_day = df_cal[df_cal["Date"] == date].drop(columns=["Date"])
            st.dataframe(
                df_day,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Impact": st.column_config.TextColumn(width="small"),
                    "Pays":   st.column_config.TextColumn(width="small"),
                }
            )
    else:
        st.info("Aucun événement pour les filtres sélectionnés.")
else:
    st.warning("Impossible de charger le calendrier économique.")