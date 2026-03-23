import yfinance as yf
import pandas as pd

# --- Indices & Actions ---
INDICES = {
    "S&P 500":    "^GSPC",
    "Nasdaq":     "^IXIC",
    "CAC 40":     "^FCHI",
    "Euro Stoxx": "^STOXX50E",
}

# --- Matières premières ---
COMMODITIES = {
    "Or":          "GC=F",
    "Pétrole WTI": "CL=F",
    "Argent":      "SI=F",
    "Gaz naturel": "NG=F",
}

def get_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Retourne l'historique OHLCV d'un ticker."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return df

def get_snapshot(tickers: dict) -> pd.DataFrame:
    """Retourne prix actuel + variation J-1 pour une liste de tickers."""
    rows = []
    for name, ticker in tickers.items():
        try:
            info  = yf.Ticker(ticker).fast_info
            price = info.last_price
            prev  = info.previous_close
            chg   = (price - prev) / prev * 100
            rows.append({"Nom": name, "Prix": price, "Var. %": chg})
        except Exception:
            rows.append({"Nom": name, "Prix": None, "Var. %": None})
    return pd.DataFrame(rows)
    