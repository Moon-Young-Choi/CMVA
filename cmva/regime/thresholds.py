"""Rolling and expanding thresholds without full-sample look-ahead."""

from __future__ import annotations

import pandas as pd


def expanding_quantile(series: pd.Series, q: float, min_periods: int = 30) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.expanding(min_periods=min_periods).quantile(q)


def rolling_quantile(series: pd.Series, q: float, window: int, min_periods: int | None = None) -> pd.Series:
    periods = min_periods or min(window, max(10, window // 4))
    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=periods).quantile(q)
