"""Select C++ numerical kernels when available, otherwise use Python fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from cmva.native.python_backend import PythonBackend


USE_CPP = False
IMPORT_ERROR: str | None = None
backend = PythonBackend()


class CppBackend(PythonBackend):
    """Pandas-friendly adapter around the pybind11 extension."""

    name = "cpp"

    def __init__(self, module: Any) -> None:
        self._module = module

    def log_price(self, prices):
        return self._module.log_price(np.asarray(prices, dtype=float))

    def log_returns(self, prices):
        return self._module.log_returns(np.asarray(prices, dtype=float))

    def difference(self, values):
        return self._module.difference(np.asarray(values, dtype=float))

    def seasonal_difference(self, values, period: int):
        return self._module.seasonal_difference(np.asarray(values, dtype=float), int(period))

    def compute_log_returns(self, close_matrix: pd.DataFrame) -> pd.DataFrame:
        if close_matrix.empty:
            return close_matrix.copy()
        data = {
            column: self._module.log_returns(close_matrix[column].to_numpy(dtype=float))
            for column in close_matrix.columns
        }
        return pd.DataFrame(data, index=close_matrix.index).replace([np.inf, -np.inf], np.nan)

    def rolling_mean(self, series, window: int):
        if isinstance(series, pd.Series):
            values = self._module.rolling_mean(series.to_numpy(dtype=float), int(window))
            return pd.Series(values, index=series.index, name=series.name)
        return self._module.rolling_mean(np.asarray(series, dtype=float), int(window))

    def rolling_std(self, series: pd.Series | pd.DataFrame, window: int):
        if isinstance(series, pd.DataFrame):
            data = {
                column: self._module.rolling_std(series[column].to_numpy(dtype=float), int(window))
                for column in series.columns
            }
            return pd.DataFrame(data, index=series.index)
        if isinstance(series, pd.Series):
            values = self._module.rolling_std(series.to_numpy(dtype=float), int(window))
            return pd.Series(values, index=series.index, name=series.name)
        return self._module.rolling_std(np.asarray(series, dtype=float), int(window))

    def rolling_variance(self, values, window: int, min_periods: int | None = None):
        return self._module.rolling_variance(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def rolling_standard_deviation(self, values, window: int, min_periods: int | None = None):
        return self._module.rolling_std(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def rolling_skewness(self, values, window: int, min_periods: int | None = None):
        return self._module.rolling_skewness(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def rolling_kurtosis(self, values, window: int, min_periods: int | None = None):
        return self._module.rolling_kurtosis(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def ewma_variance(self, returns, span: int):
        if isinstance(returns, pd.Series):
            values = self._module.ewma_variance(returns.to_numpy(dtype=float), int(span))
            return pd.Series(values, index=returns.index, name=returns.name)
        return self._module.ewma_variance(np.asarray(returns, dtype=float), int(span))

    def realized_volatility(self, returns: pd.Series | pd.DataFrame, window: int):
        if isinstance(returns, pd.DataFrame):
            data = {
                column: self._module.realized_volatility(returns[column].to_numpy(dtype=float), int(window))
                for column in returns.columns
            }
            return pd.DataFrame(data, index=returns.index)
        values = self._module.realized_volatility(returns.to_numpy(dtype=float), int(window))
        return pd.Series(values, index=returns.index, name=returns.name)

    def realized_volatility_array(self, values, window: int, min_periods: int | None = None):
        return self._module.realized_volatility(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def range_based_volatility_array(self, high, low):
        return self._module.range_based_volatility(np.asarray(high, dtype=float), np.asarray(low, dtype=float))

    def rolling_covariance(self, left, right, window: int, min_periods: int | None = None):
        return self._module.rolling_covariance(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def rolling_average_correlation(self, returns: pd.DataFrame, window: int) -> pd.Series:
        if returns.empty or returns.shape[1] < 2:
            return pd.Series(index=returns.index, dtype=float, name="avg_pairwise_corr")
        periods = min(max(3, window // 4), window)
        correlations = []
        columns = list(returns.columns)
        for left_pos, left in enumerate(columns):
            for right in columns[left_pos + 1 :]:
                values = self._module.rolling_correlation(
                    returns[left].to_numpy(dtype=float),
                    returns[right].to_numpy(dtype=float),
                    int(window),
                    periods,
                )
                correlations.append(pd.Series(values, index=returns.index))
        return pd.concat(correlations, axis=1).mean(axis=1).ffill().rename("avg_pairwise_corr")

    def rolling_correlation(self, left, right, window: int, min_periods: int | None = None):
        return self._module.rolling_correlation(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def acf(self, values, max_lag: int):
        return self._module.acf(np.asarray(values, dtype=float), int(max_lag))

    def pacf_yule_walker(self, values, max_lag: int):
        return self._module.pacf_yule_walker(np.asarray(values, dtype=float), int(max_lag))

    def rolling_ols_slope(self, values, window: int, min_periods: int | None = None):
        return self._module.rolling_ols_slope(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def rolling_ols_tstat(self, values, window: int, min_periods: int | None = None):
        return self._module.rolling_ols_tstat(
            np.asarray(values, dtype=float),
            int(window),
            -1 if min_periods is None else int(min_periods),
        )

    def qlike(self, realized_variance, forecast_variance):
        return self._module.qlike(np.asarray(realized_variance, dtype=float), np.asarray(forecast_variance, dtype=float))

    def rmse(self, actual, forecast) -> float:
        return float(self._module.rmse(np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)))

    def mae(self, actual, forecast) -> float:
        return float(self._module.mae(np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)))

    def forecast_bias(self, actual, forecast) -> float:
        return float(self._module.forecast_bias(np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)))

    def forecast_realized_correlation(self, forecast, realized) -> float:
        return float(
            self._module.forecast_realized_correlation(np.asarray(forecast, dtype=float), np.asarray(realized, dtype=float))
        )

    def ljung_box_statistic(self, values, lags: int):
        return dict(self._module.ljung_box_statistic(np.asarray(values, dtype=float), int(lags)))

    def arch_lm_statistic(self, values, lags: int):
        return dict(self._module.arch_lm_statistic(np.asarray(values, dtype=float), int(lags)))

    def jarque_bera_statistic(self, values):
        return dict(self._module.jarque_bera_statistic(np.asarray(values, dtype=float)))

    def ar_fit(self, values, p: int):
        return dict(self._module.ar_fit(np.asarray(values, dtype=float), int(p)))

    def ar_conditional_loglikelihood(self, values, params):
        return float(self._module.ar_conditional_loglikelihood(np.asarray(values, dtype=float), np.asarray(params, dtype=float)))

    def arma_css_residuals(self, values, ar_params=None, ma_params=None, intercept: float = 0.0):
        return self._module.arma_css_residuals(
            np.asarray(values, dtype=float),
            None if ar_params is None else np.asarray(ar_params, dtype=float),
            None if ma_params is None else np.asarray(ma_params, dtype=float),
            float(intercept),
        )

    def arima_css_likelihood(self, values, d: int, ar_params=None, ma_params=None, intercept: float = 0.0):
        return float(
            self._module.arima_css_likelihood(
                np.asarray(values, dtype=float),
                int(d),
                None if ar_params is None else np.asarray(ar_params, dtype=float),
                None if ma_params is None else np.asarray(ma_params, dtype=float),
                float(intercept),
            )
        )

    def arch_likelihood(self, values, omega: float, alpha: float):
        return float(self._module.arch_likelihood(np.asarray(values, dtype=float), omega, alpha))

    def garch_likelihood(self, values, omega: float, alpha: float, beta: float):
        return float(self._module.garch_likelihood(np.asarray(values, dtype=float), omega, alpha, beta))

    def student_t_garch_likelihood(self, values, omega: float, alpha: float, beta: float, nu: float):
        return float(self._module.student_t_garch_likelihood(np.asarray(values, dtype=float), omega, alpha, beta, nu))

    def rank_stability(self, ranks):
        return float(self._module.rank_stability(np.asarray(ranks, dtype=float)))


if os.environ.get("CMVA_USE_CPP", "1").strip().lower() not in {"0", "false", "no"}:
    try:
        import cmva_cpp  # type: ignore

        backend = CppBackend(cmva_cpp)
        USE_CPP = True
    except ImportError as exc:
        IMPORT_ERROR = str(exc)
        backend = PythonBackend()
        USE_CPP = False


def backend_status() -> dict[str, object]:
    return {
        "active": "cpp" if USE_CPP else "python",
        "cpp_available": USE_CPP,
        "python_fallback": not USE_CPP,
        "import_error": IMPORT_ERROR,
    }
