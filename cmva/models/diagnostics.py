"""Lightweight model diagnostics."""

from __future__ import annotations

import pandas as pd


def standardized_residual(return_value: float, mean: float, volatility: float) -> float:
    if volatility <= 0:
        return 0.0
    return float((return_value - mean) / volatility)


def latest_standardized_residual(returns: pd.Series, mean: float, volatility: float) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return standardized_residual(float(clean.iloc[-1]), mean, volatility)
