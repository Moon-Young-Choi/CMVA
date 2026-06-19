"""Rolling PCA common-risk feature."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_pca1_share(
    returns: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    max_exact_points: int = 2000,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="pca1_share")
    periods = min_periods or min(max(5, window // 4), window)
    step = 1
    if len(returns) > max_exact_points:
        step = max(1, min(96, window // 24))
    shares = pd.Series(np.nan, index=returns.index, name="pca1_share")
    positions = list(range(0, len(returns), step))
    if positions[-1] != len(returns) - 1:
        positions.append(len(returns) - 1)
    for pos in positions:
        start = max(0, pos - window + 1)
        sample = returns.iloc[start : pos + 1].dropna(axis=1, how="all").dropna(axis=0, how="any")
        if len(sample) < periods or sample.shape[1] < 2:
            continue
        centered = sample - sample.mean(axis=0)
        covariance = np.cov(centered.to_numpy(), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        total = eigenvalues.sum()
        shares.iloc[pos] = float(eigenvalues.max() / total) if total > 0 else np.nan
    return shares.ffill().rename("pca1_share")
