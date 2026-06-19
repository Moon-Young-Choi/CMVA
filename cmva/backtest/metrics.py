"""Backtest performance metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series, periods_per_year: int = 365 * 24) -> dict[str, float]:
    clean = returns.fillna(0.0)
    if clean.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "worst_1h_return": 0.0,
        }
    equity = equity_curve(clean)
    cumulative = float(equity.iloc[-1] - 1.0)
    years = max(len(clean) / periods_per_year, 1 / periods_per_year)
    annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    annualized_vol = float(clean.std(ddof=0) * math.sqrt(periods_per_year))
    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else 0.0
    mdd = max_drawdown(equity)
    calmar = float(annualized_return / abs(mdd)) if mdd < 0 else 0.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "worst_1h_return": float(clean.min()),
    }
