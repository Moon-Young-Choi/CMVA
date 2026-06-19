"""Rolling PCA common-risk feature."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_pca1_share(returns: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="pca1_share")
    periods = min_periods or min(max(5, window // 4), window)
    shares: list[float] = []
    for pos in range(len(returns)):
        start = max(0, pos - window + 1)
        sample = returns.iloc[start : pos + 1].dropna(axis=1, how="all").dropna(axis=0, how="any")
        if len(sample) < periods or sample.shape[1] < 2:
            shares.append(np.nan)
            continue
        centered = sample - sample.mean(axis=0)
        covariance = np.cov(centered.to_numpy(), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        total = eigenvalues.sum()
        shares.append(float(eigenvalues.max() / total) if total > 0 else np.nan)
    return pd.Series(shares, index=returns.index, name="pca1_share")
