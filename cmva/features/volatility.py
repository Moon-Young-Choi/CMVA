"""Rolling volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    periods = min_periods or min(max(3, window // 4), window)
    return returns.rolling(window=window, min_periods=periods).std()


def ewma_volatility(returns: pd.Series, span: int = 24, min_periods: int = 3) -> pd.Series:
    return returns.pow(2).ewm(span=span, min_periods=min_periods, adjust=False).mean().pow(0.5)


def range_based_volatility(candles: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.Series:
    if candles.empty:
        return pd.Series(dtype=float, name="range_vol")
    data = candles.copy()
    high_low = np.log(data["high"] / data["low"]).replace([np.inf, -np.inf], np.nan)
    parkinson = (high_low.pow(2) / (4.0 * np.log(2.0))).pow(0.5)
    data["range_vol"] = parkinson
    basket = data.pivot_table(index="open_time", columns="symbol", values="range_vol", aggfunc="last").mean(axis=1)
    periods = min_periods or min(max(3, window // 4), window)
    return basket.rolling(window=window, min_periods=periods).mean().rename("range_vol")


def rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    periods = min_periods or min(max(10, window // 4), window)

    def percentile(values):
        clean = pd.Series(values).dropna()
        if clean.empty:
            return np.nan
        latest = clean.iloc[-1]
        return float((clean <= latest).mean())

    return series.rolling(window=window, min_periods=periods).apply(percentile, raw=False)
