"""Python reference numerical backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmva.features.correlation import rolling_average_pairwise_correlation
from cmva.features.returns import compute_log_returns
from cmva.features.trend import rolling_ols_slope, rolling_ols_tstat
from cmva.features.volatility import ewma_volatility, realized_volatility


class PythonBackend:
    name = "python"

    def compute_log_returns(self, close_matrix: pd.DataFrame) -> pd.DataFrame:
        return compute_log_returns(close_matrix)

    def rolling_mean(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=min(max(3, window // 4), window)).mean()

    def rolling_std(self, series: pd.Series | pd.DataFrame, window: int):
        return series.rolling(window=window, min_periods=min(max(3, window // 4), window)).std()

    def ewma_variance(self, returns: pd.Series, span: int) -> pd.Series:
        return returns.pow(2).ewm(span=span, min_periods=3, adjust=False).mean()

    def realized_volatility(self, returns: pd.Series | pd.DataFrame, window: int):
        return realized_volatility(returns, window)

    def rolling_average_correlation(self, returns: pd.DataFrame, window: int) -> pd.Series:
        return rolling_average_pairwise_correlation(returns, window)

    def rolling_ols_slope_tstat(self, series: pd.Series, window: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "slope": rolling_ols_slope(series, window),
                "tstat": rolling_ols_tstat(series, window),
            }
        )

    def max_drawdown(self, values: pd.Series) -> float:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        drawdown = clean / clean.cummax() - 1.0
        return float(drawdown.min())

    def forecast_loss_qlike(self, realized_variance: pd.Series, forecast_variance: pd.Series) -> pd.Series:
        aligned = pd.concat(
            [realized_variance.rename("realized"), forecast_variance.rename("forecast")],
            axis=1,
        ).dropna()
        aligned = aligned.loc[aligned["forecast"] > 0]
        if aligned.empty:
            return pd.Series(dtype=float, name="qlike")
        return (aligned["realized"] / aligned["forecast"] + np.log(aligned["forecast"])).rename("qlike")
