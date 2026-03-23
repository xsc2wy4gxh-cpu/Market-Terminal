import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    """
    Sharpe Ratio annualisé.
    Mesure le rendement excédentaire par unité de risque total.
    risk_free = taux sans risque (4% par défaut = taux actuel ~)
    """
    if returns.std() == 0:
        return 0.0
    excess  = returns - risk_free / 252
    return round(float(excess.mean() / excess.std() * np.sqrt(252)), 3)

def sortino_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    """
    Sortino Ratio annualisé.
    Comme le Sharpe mais pénalise uniquement la volatilité baissière.
    Plus pertinent car les investisseurs ne se plaignent pas de la volatilité haussière.
    """
    excess       = returns - risk_free / 252
    downside_std = excess[excess < 0].std()
    if downside_std == 0:
        return 0.0
    return round(float(excess.mean() / downside_std * np.sqrt(252)), 3)

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum Drawdown : perte maximale depuis un sommet historique.
    Exprimé en % négatif. Ex: -0.25 = -25%
    """
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return round(float(drawdown.min()), 4)

def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    Calmar Ratio = Rendement annualisé / |Max Drawdown|
    Mesure le rendement obtenu par unité de drawdown maximum.
    """
    annual_return = float(returns.mean() * 252)
    mdd           = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return round(annual_return / mdd, 3)

def win_rate(trades: list) -> float:
    """
    Pourcentage de trades gagnants.
    trades = liste de rendements par trade (positif ou négatif)
    """
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t > 0)
    return round(wins / len(trades) * 100, 1)

def profit_factor(trades: list) -> float:
    """
    Profit Factor = somme des gains / somme des pertes.
    > 1.5 est considéré comme bon. > 2 est excellent.
    """
    gains  = sum(t for t in trades if t > 0)
    losses = abs(sum(t for t in trades if t < 0))
    if losses == 0:
        return float("inf")
    return round(gains / losses, 3)

def annual_return(returns: pd.Series) -> float:
    """Rendement annualisé moyen."""
    return round(float(returns.mean() * 252 * 100), 2)

def annual_volatility(returns: pd.Series) -> float:
    """Volatilité annualisée."""
    return round(float(returns.std() * np.sqrt(252) * 100), 2)

def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta vs benchmark.
    Mesure la sensibilité de la stratégie aux mouvements du marché.
    Beta > 1 = plus volatile que le marché.
    Beta < 1 = moins volatile.
    Beta < 0 = décorrélé ou inversé.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    cov    = aligned.cov().iloc[0, 1]
    var_bm = aligned.iloc[:, 1].var()
    if var_bm == 0:
        return 0.0
    return round(cov / var_bm, 3)

def alpha(returns: pd.Series, benchmark_returns: pd.Series,
          risk_free: float = 0.04) -> float:
    """
    Alpha de Jensen annualisé.
    Mesure la surperformance vs ce qu'on attendrait compte tenu du beta.
    Alpha > 0 = la stratégie génère de la valeur au-delà du risque pris.
    """
    b       = beta(returns, benchmark_returns)
    r_port  = returns.mean() * 252
    r_bench = benchmark_returns.mean() * 252
    return round(float(r_port - (risk_free + b * (r_bench - risk_free))), 4)

def compute_all_metrics(returns: pd.Series, equity_curve: pd.Series,
                        benchmark_returns: pd.Series, trades: list) -> dict:
    """Calcule toutes les métriques d'un coup et retourne un dictionnaire."""
    return {
        "Rendement annualisé %":  annual_return(returns),
        "Volatilité annualisée %": annual_volatility(returns),
        "Sharpe Ratio":           sharpe_ratio(returns),
        "Sortino Ratio":          sortino_ratio(returns),
        "Max Drawdown %":         round(max_drawdown(equity_curve) * 100, 2),
        "Calmar Ratio":           calmar_ratio(returns, equity_curve),
        "Win Rate %":             win_rate(trades),
        "Profit Factor":          profit_factor(trades),
        "Beta":                   beta(returns, benchmark_returns),
        "Alpha":                  alpha(returns, benchmark_returns),
        "Nb Trades":              len(trades),
    }
