"""Rolling correlation features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmva.native.backend import backend


def rolling_average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    max_exact_points: int = 2000,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="avg_pairwise_corr")
    if max_exact_points is None or len(returns) <= max_exact_points:
        return backend.rolling_average_correlation(returns, window).rename("avg_pairwise_corr")
    periods = min_periods or min(max(3, window // 4), window)
    index = returns.index
    step = 1
    if len(returns) > max_exact_points:
        step = max(1, min(96, window // 24))
    values = pd.Series(np.nan, index=index, name="avg_pairwise_corr")
    positions = list(range(0, len(returns), step))
    if positions[-1] != len(returns) - 1:
        positions.append(len(returns) - 1)
    for pos in positions:
        start = max(0, pos - window + 1)
        sample = returns.iloc[start : pos + 1].dropna(axis=1, how="all")
        if len(sample) < periods or sample.shape[1] < 2:
            continue
        corr = sample.corr(min_periods=max(3, min(periods, len(sample))))
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairwise = corr.where(mask).stack()
        values.iloc[pos] = float(pairwise.mean()) if not pairwise.empty else np.nan
    return values.ffill().rename("avg_pairwise_corr")
