import pandas as pd

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les moyennes mobiles 20, 50 et 200 jours."""
    df = df.copy()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calcule le RSI sur une fenêtre de 14 jours par défaut."""
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_bollinger(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calcule les bandes de Bollinger sur 20 jours."""
    df = df.copy()
    ma  = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_upper"] = ma + 2 * std
    df["BB_lower"] = ma - 2 * std
    df["BB_mid"]   = ma
    return df
    