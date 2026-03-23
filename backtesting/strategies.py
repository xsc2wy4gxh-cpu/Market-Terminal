import pandas as pd
import numpy as np

class MovingAverageCrossStrategy:
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window  = long_window
        self.min_periods  = long_window + 2
        self.name         = f"MA Cross ({short_window}/{long_window})"

    def generate_signal(self, data, i):
        close = data["Close"].squeeze()

        # Calcul direct sur la serie complete jusqu a i
        ma_short = close.iloc[:i].rolling(self.short_window).mean()
        ma_long  = close.iloc[:i].rolling(self.long_window).mean()

        if len(ma_short) < 2 or len(ma_long) < 2:
            return "HOLD"

        ms_now  = float(ma_short.iloc[-1])
        ms_prev = float(ma_short.iloc[-2])
        ml_now  = float(ma_long.iloc[-1])
        ml_prev = float(ma_long.iloc[-2])

        if pd.isna(ms_now) or pd.isna(ml_now):
            return "HOLD"

        # Golden Cross
        if ms_prev <= ml_prev and ms_now > ml_now:
            return "BUY"
        # Death Cross
        if ms_prev >= ml_prev and ms_now < ml_now:
            return "SELL"

        return "HOLD"


class RSIMeanReversionStrategy:
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought
        self.min_periods = rsi_period + 2
        self.name        = f"RSI Mean Reversion ({oversold}/{overbought})"

    def _compute_rsi(self, close, i):
        window = close.iloc[max(0, i - self.rsi_period - 1):i]
        delta  = window.diff().dropna()
        if len(delta) == 0:
            return 50.0
        gain = delta.clip(lower=0).mean()
        loss = (-delta.clip(upper=0)).mean()
        if loss == 0:
            return 100.0
        return float(100 - (100 / (1 + gain / loss)))

    def generate_signal(self, data, i):
        close = data["Close"].squeeze()
        rsi   = self._compute_rsi(close, i)
        if rsi < self.oversold:
            return "BUY"
        if rsi > self.overbought:
            return "SELL"
        return "HOLD"


class MomentumStrategy:
    def __init__(self, lookback=20, threshold=0.02):
        self.lookback    = lookback
        self.threshold   = threshold
        self.min_periods = lookback + 1
        self.name        = f"Momentum ({lookback}j, seuil {threshold*100:.0f}%)"

    def generate_signal(self, data, i):
        close = data["Close"].squeeze()
        if i < self.lookback:
            return "HOLD"
        ret = (float(close.iloc[i]) - float(close.iloc[i - self.lookback])) / float(close.iloc[i - self.lookback])
        if ret > self.threshold:
            return "BUY"
        if ret < 0:
            return "SELL"
        return "HOLD"


class PairsTradingStrategy:
    def __init__(self, window=30, z_threshold=2.0):
        self.window        = window
        self.z_threshold   = z_threshold
        self.min_periods   = window + 1
        self.name          = f"Pairs Trading (z={z_threshold})"
        self.spread_series = None

    def set_spread(self, spread):
        self.spread_series = spread

    def generate_signal(self, data, i):
        if self.spread_series is None or i >= len(self.spread_series):
            return "HOLD"
        window = self.spread_series.iloc[max(0, i - self.window):i]
        if len(window) < self.window:
            return "HOLD"
        mean   = window.mean()
        std    = window.std()
        if std == 0:
            return "HOLD"
        zscore = (float(self.spread_series.iloc[i]) - mean) / std
        if zscore < -self.z_threshold:
            return "BUY"
        if zscore > self.z_threshold:
            return "SELL"
        return "HOLD"


def get_strategy(name, params):
    if name == "MA Cross":
        return MovingAverageCrossStrategy(
            short_window=params.get("short_window", 50),
            long_window=params.get("long_window", 200)
        )
    elif name == "RSI Mean Reversion":
        return RSIMeanReversionStrategy(
            rsi_period=params.get("rsi_period", 14),
            oversold=params.get("oversold", 30),
            overbought=params.get("overbought", 70)
        )
    elif name == "Momentum":
        return MomentumStrategy(
            lookback=params.get("lookback", 20),
            threshold=params.get("threshold", 0.02)
        )
    elif name == "Pairs Trading":
        return PairsTradingStrategy(
            window=params.get("window", 30),
            z_threshold=params.get("z_threshold", 2.0)
        )
    else:
        raise ValueError(f"Strategie inconnue : {name}")
