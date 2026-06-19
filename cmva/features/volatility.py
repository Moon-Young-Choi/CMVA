"""Rolling volatility features."""

from __future__ import annotations

import pandas as pd


def realized_volatility(returns: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    periods = min_periods or min(max(3, window // 4), window)
    return returns.rolling(window=window, min_periods=periods).std()


def ewma_volatility(returns: pd.Series, span: int = 24, min_periods: int = 3) -> pd.Series:
    return returns.pow(2).ewm(span=span, min_periods=min_periods, adjust=False).mean().pow(0.5)
