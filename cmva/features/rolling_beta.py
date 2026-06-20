"""Rolling beta features."""

from __future__ import annotations

import pandas as pd

from cmva.native.backend import backend


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
    variance = pd.Series(backend.rolling_variance(benchmark.to_numpy(dtype=float), window, periods), index=returns.index)
    betas = {}
    for symbol in returns.columns:
        covariance = pd.Series(
            backend.rolling_covariance(returns[symbol].to_numpy(dtype=float), benchmark.to_numpy(dtype=float), window, periods),
            index=returns.index,
        )
        betas[symbol] = covariance / variance
    return pd.DataFrame(betas, index=returns.index)
