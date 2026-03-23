import pandas as pd
import numpy as np
from backtesting.metrics import compute_all_metrics

class BacktestEngine:
    def __init__(self, data, strategy, initial_capital=100_000,
                 commission=0.001, slippage=0.001, benchmark=None):
        self.data            = data.copy()
        self.strategy        = strategy
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        self.benchmark       = benchmark
        self.cash            = initial_capital
        self.position        = 0
        self.position_price  = 0.0
        self.equity_curve    = []
        self.returns         = []
        self.trades          = []
        self.signals_log     = []

    def _execution_price(self, price, direction):
        if direction == "BUY":
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)

    def _buy(self, price, date):
        exec_price = self._execution_price(price, "BUY")
        amount     = self.cash * 0.99
        shares     = amount / exec_price
        cost       = shares * exec_price
        commission = cost * self.commission
        if cost + commission > self.cash:
            return False
        self.cash          -= (cost + commission)
        self.position      += shares
        self.position_price = exec_price
        self.signals_log.append({
            "Date": date, "Signal": "BUY",
            "Prix": round(exec_price, 2),
            "Shares": round(shares, 4),
            "Cash": round(self.cash, 2),
            "PnL %": None,
        })
        return True

    def _sell(self, price, date):
        if self.position <= 0:
            return False
        exec_price = self._execution_price(price, "SELL")
        shares     = self.position
        proceeds   = shares * exec_price
        commission = proceeds * self.commission
        trade_pnl  = (exec_price - self.position_price) / self.position_price
        self.cash     += (proceeds - commission)
        self.position  = 0
        self.trades.append(trade_pnl)
        self.signals_log.append({
            "Date": date, "Signal": "SELL",
            "Prix": round(exec_price, 2),
            "Shares": round(shares, 4),
            "Cash": round(self.cash, 2),
            "PnL %": round(trade_pnl * 100, 2),
        })
        return True

    def run(self):
        close       = self.data["Close"].squeeze()
        prev_equity = self.initial_capital

        # Pre-calcul de tous les signaux
        all_signals = ["HOLD"] * len(self.data)
        for i in range(self.strategy.min_periods, len(self.data)):
            all_signals[i] = self.strategy.generate_signal(self.data, i)

        # Simulation jour par jour
        for i in range(len(self.data)):
            date   = self.data.index[i]
            price  = float(close.iloc[i])
            signal = all_signals[i]

            # Execute les ordres
            if signal == "BUY" and self.position == 0:
                self._buy(price, date)
            elif signal == "SELL" and self.position > 0:
                self._sell(price, date)

            # Mise a jour equity
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            daily_return = (equity - prev_equity) / prev_equity
            self.returns.append(daily_return)
            prev_equity = equity

        # Liquidation finale si position encore ouverte
        if self.position > 0:
            self._sell(float(close.iloc[-1]), self.data.index[-1])
            # Met a jour le dernier point de l equity curve
            self.equity_curve[-1] = self.cash

        equity_series  = pd.Series(self.equity_curve, index=self.data.index)
        returns_series = pd.Series(self.returns,       index=self.data.index)

        if self.benchmark is not None:
            bm_returns = self.benchmark.pct_change().dropna()
        else:
            bm_returns = returns_series * 0

        metrics = compute_all_metrics(
            returns_series, equity_series, bm_returns, self.trades
        )

        bh_return = (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100

        # Debug
        print(f"Nb signaux BUY/SELL: {sum(1 for s in all_signals if s != 'HOLD')}")
        print(f"Nb trades executes: {len(self.trades)}")
        print(f"Capital final: {self.equity_curve[-1]:.2f}")

        return {
            "metrics":       metrics,
            "equity_curve":  equity_series,
            "returns":       returns_series,
            "trades":        self.trades,
            "signals":       pd.DataFrame(self.signals_log),
            "bh_return":     round(bh_return, 2),
            "final_capital": round(self.equity_curve[-1], 2),
        }
