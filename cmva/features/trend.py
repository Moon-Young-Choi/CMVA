"""Rolling trend descriptors."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cmva.native.backend import backend


def rolling_ols_slope(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    periods = min_periods or min(max(4, window // 3), window)
    if min_periods is None:
        values = backend.rolling_ols_slope(series.to_numpy(dtype=float), window, periods)
        return pd.Series(values, index=series.index, name="trend_slope")
    return series.rolling(window=window, min_periods=periods).apply(_slope, raw=True).rename("trend_slope")


def rolling_ols_tstat(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    periods = min_periods or min(max(4, window // 3), window)
    if min_periods is None:
        values = backend.rolling_ols_tstat(series.to_numpy(dtype=float), window, periods)
        return pd.Series(values, index=series.index, name="trend_tstat")
    return series.rolling(window=window, min_periods=periods).apply(_slope_tstat, raw=True).rename("trend_tstat")


def rolling_autocorrelation(series: pd.Series, window: int, lag: int = 1, min_periods: int | None = None) -> pd.Series:
    periods = min_periods or min(max(4, window // 3), window)
    return series.rolling(window=window, min_periods=periods).apply(lambda values: _autocorr(values, lag), raw=True)


def up_down_ratio(returns: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    periods = min_periods or min(max(4, window // 3), window)

    def ratio(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if clean.size == 0:
            return np.nan
        down = float(np.sum(clean < 0))
        up = float(np.sum(clean > 0))
        denominator = up + down
        if denominator == 0:
            return 0.5
        return up / denominator

    return returns.rolling(window=window, min_periods=periods).apply(ratio, raw=True).rename("up_down_ratio")


def _slope(values: np.ndarray) -> float:
    y = values[np.isfinite(values)]
    if y.size < 2:
        return np.nan
    x = np.arange(y.size, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denominator)


def _slope_tstat(values: np.ndarray) -> float:
    y = values[np.isfinite(values)]
    n = y.size
    if n < 4:
        return np.nan
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return np.nan
    slope = float(np.dot(x, y - y.mean()) / denominator)
    intercept = float(y.mean())
    residuals = y - (intercept + slope * x)
    sse = float(np.dot(residuals, residuals))
    dof = max(n - 2, 1)
    se = math.sqrt((sse / dof) / denominator) if sse > 0 else 0.0
    if se <= 0:
        return 0.0
    return float(slope / se)


def _autocorr(values: np.ndarray, lag: int) -> float:
    clean = values[np.isfinite(values)]
    if clean.size <= lag:
        return np.nan
    left = clean[:-lag]
    right = clean[lag:]
    if left.std() <= 0 or right.std() <= 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
