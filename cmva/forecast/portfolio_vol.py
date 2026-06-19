"""Portfolio volatility helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def equal_weight_portfolio_volatility(returns: pd.DataFrame, window: int = 24) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    weights = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    values: list[float] = []
    for pos in range(len(returns)):
        start = max(0, pos - window + 1)
        sample = returns.iloc[start : pos + 1].dropna()
        if len(sample) < 3:
            values.append(np.nan)
            continue
        covariance = sample.cov().to_numpy()
        variance = float(weights @ covariance @ weights)
        values.append(math.sqrt(max(variance, 0.0)))
    return pd.Series(values, index=returns.index, name="portfolio_vol")
