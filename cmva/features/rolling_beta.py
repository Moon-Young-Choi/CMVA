"""Rolling beta features."""

from __future__ import annotations

import pandas as pd


def rolling_beta(
    returns: pd.DataFrame,
    benchmark_symbol: str = "BTCUSDT",
    window: int = 168,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(index=returns.index)
    if benchmark_symbol not in returns.columns:
        return pd.DataFrame(index=returns.index)
    periods = min_periods or min(max(5, window // 4), window)
    benchmark = returns[benchmark_symbol]
    variance = benchmark.rolling(window=window, min_periods=periods).var()
    betas = {}
    for symbol in returns.columns:
        covariance = returns[symbol].rolling(window=window, min_periods=periods).cov(benchmark)
        betas[symbol] = covariance / variance
    return pd.DataFrame(betas, index=returns.index)
