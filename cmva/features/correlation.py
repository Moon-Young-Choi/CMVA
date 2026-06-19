"""Rolling correlation features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="avg_pairwise_corr")
    periods = min_periods or min(max(3, window // 4), window)
    values: list[float] = []
    index = returns.index
    for pos in range(len(returns)):
        start = max(0, pos - window + 1)
        sample = returns.iloc[start : pos + 1].dropna(axis=1, how="all")
        if len(sample) < periods or sample.shape[1] < 2:
            values.append(np.nan)
            continue
        corr = sample.corr(min_periods=max(3, min(periods, len(sample))))
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairwise = corr.where(mask).stack()
        values.append(float(pairwise.mean()) if not pairwise.empty else np.nan)
    return pd.Series(values, index=index, name="avg_pairwise_corr")
