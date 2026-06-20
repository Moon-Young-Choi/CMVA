"""Python reference numerical backend."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class PythonBackend:
    name = "python"

    def log_price(self, prices):
        values = _array(prices)
        return np.where(values > 0, np.log(values), np.nan)

    def log_returns(self, prices):
        values = _array(prices)
        out = np.full(values.shape, np.nan, dtype=float)
        valid = (values[1:] > 0) & (values[:-1] > 0)
        out[1:] = np.where(valid, np.log(values[1:] / values[:-1]), np.nan)
        return out

    def difference(self, values):
        data = _array(values)
        out = np.full(data.shape, np.nan, dtype=float)
        out[1:] = data[1:] - data[:-1]
        return out

    def seasonal_difference(self, values, period: int):
        data = _array(values)
        out = np.full(data.shape, np.nan, dtype=float)
        p = max(1, int(period))
        out[p:] = data[p:] - data[:-p]
        return out

    def rolling_variance(self, values, window: int, min_periods: int | None = None):
        return _rolling_series(values, window, min_periods).var().to_numpy()

    def rolling_standard_deviation(self, values, window: int, min_periods: int | None = None):
        return _rolling_series(values, window, min_periods).std().to_numpy()

    def rolling_skewness(self, values, window: int, min_periods: int | None = None):
        return _rolling_series(values, window, min_periods).skew().to_numpy()

    def rolling_kurtosis(self, values, window: int, min_periods: int | None = None):
        return _rolling_series(values, window, min_periods).kurt().to_numpy()

    def realized_volatility_array(self, values, window: int, min_periods: int | None = None):
        return self.rolling_standard_deviation(values, window, min_periods)

    def range_based_volatility_array(self, high, low):
        high_values = _array(high)
        low_values = _array(low)
        ratio = np.where((high_values > 0) & (low_values > 0), high_values / low_values, np.nan)
        return np.sqrt(np.log(ratio) ** 2 / (4.0 * np.log(2.0)))

    def rolling_covariance(self, left, right, window: int, min_periods: int | None = None):
        periods = _min_periods(window, min_periods)
        return pd.Series(_array(left)).rolling(window, min_periods=periods).cov(pd.Series(_array(right))).to_numpy()

    def rolling_correlation(self, left, right, window: int, min_periods: int | None = None):
        periods = _min_periods(window, min_periods)
        return pd.Series(_array(left)).rolling(window, min_periods=periods).corr(pd.Series(_array(right))).to_numpy()

    def acf(self, values, max_lag: int):
        data = _finite(_array(values))
        if data.size == 0:
            return np.full(max_lag + 1, np.nan)
        centered = data - data.mean()
        denom = float(np.dot(centered, centered))
        out = []
        for lag in range(max_lag + 1):
            if lag == 0:
                out.append(1.0)
            elif lag >= data.size or denom <= 0:
                out.append(np.nan)
            else:
                out.append(float(np.dot(centered[:-lag], centered[lag:]) / denom))
        return np.asarray(out)

    def pacf_yule_walker(self, values, max_lag: int):
        acf_values = self.acf(values, max_lag)
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        for lag in range(1, max_lag + 1):
            r = acf_values[1 : lag + 1]
            toeplitz = np.fromfunction(lambda i, j: acf_values[np.abs(i - j).astype(int)], (lag, lag))
            try:
                phi = np.linalg.solve(toeplitz, r)
                out[lag] = phi[-1]
            except np.linalg.LinAlgError:
                out[lag] = np.nan
        return out

    def qlike(self, realized_variance, forecast_variance):
        realized = _array(realized_variance)
        forecast = _array(forecast_variance)
        out = np.full(realized.shape, np.nan)
        mask = np.isfinite(realized) & np.isfinite(forecast) & (forecast > 0)
        out[mask] = realized[mask] / forecast[mask] + np.log(forecast[mask])
        return out

    def rmse(self, actual, forecast) -> float:
        error = _aligned_error(actual, forecast)
        return float(np.sqrt(np.mean(error * error))) if error.size else np.nan

    def mae(self, actual, forecast) -> float:
        error = _aligned_error(actual, forecast)
        return float(np.mean(np.abs(error))) if error.size else np.nan

    def forecast_bias(self, actual, forecast) -> float:
        error = _aligned_error(actual, forecast)
        return float(np.mean(error)) if error.size else np.nan

    def forecast_realized_correlation(self, forecast, realized) -> float:
        left = _array(forecast)
        right = _array(realized)
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() < 3:
            return np.nan
        return float(np.corrcoef(left[mask], right[mask])[0, 1])

    def ljung_box_statistic(self, values, lags: int):
        data = _finite(_array(values))
        n = data.size
        if n <= lags + 1:
            return {"statistic": np.nan, "p_value": np.nan, "sample_size": n}
        acf_values = self.acf(data, lags)
        statistic = float(n * (n + 2) * np.nansum([(acf_values[k] ** 2) / (n - k) for k in range(1, lags + 1)]))
        return {"statistic": statistic, "p_value": _chi2_sf(statistic, lags), "sample_size": n}

    def arch_lm_statistic(self, values, lags: int):
        data = _finite(_array(values))
        squared = data * data
        if squared.size <= lags + 2:
            return {"statistic": np.nan, "p_value": np.nan, "sample_size": squared.size}
        y = squared[lags:]
        x = [np.ones_like(y)]
        for lag in range(1, lags + 1):
            x.append(squared[lags - lag : squared.size - lag])
        matrix = np.column_stack(x)
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        fitted = matrix @ beta
        ssr = float(np.sum((y - fitted) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if sst <= 0 else 1.0 - ssr / sst
        statistic = float(len(y) * max(r2, 0.0))
        return {"statistic": statistic, "p_value": _chi2_sf(statistic, lags), "sample_size": len(y)}

    def jarque_bera_statistic(self, values):
        data = _finite(_array(values))
        n = data.size
        if n < 3:
            return {"statistic": np.nan, "p_value": np.nan, "sample_size": n, "skewness": np.nan, "kurtosis": np.nan}
        centered = data - data.mean()
        std = data.std(ddof=0)
        if std <= 0:
            return {"statistic": 0.0, "p_value": 1.0, "sample_size": n, "skewness": 0.0, "kurtosis": 3.0}
        skewness = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4))
        statistic = float(n / 6.0 * (skewness**2 + ((kurtosis - 3.0) ** 2) / 4.0))
        return {
            "statistic": statistic,
            "p_value": _chi2_sf(statistic, 2),
            "sample_size": n,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def ar_fit(self, values, p: int):
        data = _finite(_array(values))
        p = max(1, int(p))
        if data.size <= p + 1:
            return {"success": False, "params": [], "sigma2": np.nan, "log_likelihood": np.nan}
        y = data[p:]
        x = [np.ones_like(y)]
        for lag in range(1, p + 1):
            x.append(data[p - lag : data.size - lag])
        matrix = np.column_stack(x)
        params = np.linalg.lstsq(matrix, y, rcond=None)[0]
        residuals = y - matrix @ params
        sigma2 = max(float(np.mean(residuals * residuals)), 1e-12)
        loglik = float(-0.5 * len(residuals) * (np.log(2.0 * np.pi * sigma2) + 1.0))
        k = p + 2
        return {
            "success": True,
            "params": params.tolist(),
            "sigma2": sigma2,
            "log_likelihood": loglik,
            "aic": -2.0 * loglik + 2.0 * k,
            "bic": -2.0 * loglik + np.log(max(len(residuals), 2)) * k,
        }

    def ar_conditional_loglikelihood(self, values, params):
        data = _finite(_array(values))
        coef = _array(params)
        p = max(0, len(coef) - 1)
        if data.size <= p + 1:
            return np.nan
        y = data[p:]
        fitted = np.full_like(y, coef[0] if len(coef) else 0.0)
        for lag in range(1, p + 1):
            fitted += coef[lag] * data[p - lag : data.size - lag]
        resid = y - fitted
        sigma2 = max(float(np.mean(resid * resid)), 1e-12)
        return float(-0.5 * len(resid) * (np.log(2.0 * np.pi * sigma2) + 1.0))

    def arma_css_residuals(self, values, ar_params=None, ma_params=None, intercept: float = 0.0):
        data = _array(values)
        ar = _array([] if ar_params is None else ar_params)
        ma = _array([] if ma_params is None else ma_params)
        resid = np.zeros_like(data, dtype=float)
        for t in range(data.size):
            fitted = intercept
            for lag, coef in enumerate(ar, start=1):
                if t - lag >= 0:
                    fitted += coef * data[t - lag]
            for lag, coef in enumerate(ma, start=1):
                if t - lag >= 0:
                    fitted += coef * resid[t - lag]
            resid[t] = data[t] - fitted if np.isfinite(data[t]) else np.nan
        return resid

    def arima_css_likelihood(self, values, d: int, ar_params=None, ma_params=None, intercept: float = 0.0):
        data = _array(values)
        for _ in range(max(0, int(d))):
            data = np.diff(data)
        resid = _finite(self.arma_css_residuals(data, ar_params, ma_params, intercept))
        if resid.size == 0:
            return np.nan
        sigma2 = max(float(np.mean(resid * resid)), 1e-12)
        return float(-0.5 * len(resid) * (np.log(2.0 * np.pi * sigma2) + 1.0))

    def arch_likelihood(self, values, omega: float, alpha: float):
        data = _finite(_array(values))
        variance = max(float(np.var(data)) if data.size else 0.0, 1e-12)
        ll = 0.0
        for value in data:
            variance = max(float(omega) + float(alpha) * value * value, 1e-12)
            ll += -0.5 * (np.log(2.0 * np.pi * variance) + value * value / variance)
        return float(ll)

    def garch_likelihood(self, values, omega: float, alpha: float, beta: float):
        data = _finite(_array(values))
        variance = max(float(np.var(data)) if data.size else 0.0, 1e-12)
        ll = 0.0
        for value in data:
            variance = max(float(omega) + float(alpha) * value * value + float(beta) * variance, 1e-12)
            ll += -0.5 * (np.log(2.0 * np.pi * variance) + value * value / variance)
        return float(ll)

    def student_t_garch_likelihood(self, values, omega: float, alpha: float, beta: float, nu: float):
        data = _finite(_array(values))
        variance = max(float(np.var(data)) if data.size else 0.0, 1e-12)
        nu = max(float(nu), 2.1)
        ll = 0.0
        for value in data:
            variance = max(float(omega) + float(alpha) * value * value + float(beta) * variance, 1e-12)
            z2 = value * value / variance
            ll += (
                math.lgamma((nu + 1.0) / 2.0)
                - math.lgamma(nu / 2.0)
                - 0.5 * np.log((nu - 2.0) * np.pi * variance)
                - ((nu + 1.0) / 2.0) * np.log1p(z2 / (nu - 2.0))
            )
        return float(ll)

    def rank_stability(self, ranks):
        values = _finite(_array(ranks))
        if values.size <= 1:
            return 1.0
        return float(1.0 / (1.0 + np.abs(np.diff(values)).mean()))

    def compute_log_returns(self, close_matrix: pd.DataFrame) -> pd.DataFrame:
        if close_matrix.empty:
            return close_matrix.copy()
        return np.log(close_matrix / close_matrix.shift(1)).replace([np.inf, -np.inf], np.nan)

    def rolling_mean(self, series: pd.Series, window: int) -> pd.Series:
        if isinstance(series, pd.Series):
            return series.rolling(window=window, min_periods=min(max(3, window // 4), window)).mean()
        return _rolling_series(series, window).mean().to_numpy()

    def rolling_std(self, series: pd.Series | pd.DataFrame, window: int):
        return series.rolling(window=window, min_periods=min(max(3, window // 4), window)).std()

    def ewma_variance(self, returns: pd.Series, span: int) -> pd.Series:
        if isinstance(returns, pd.Series):
            return returns.pow(2).ewm(span=span, min_periods=3, adjust=False).mean()
        return pd.Series(_array(returns)).pow(2).ewm(span=span, min_periods=3, adjust=False).mean().to_numpy()

    def realized_volatility(self, returns: pd.Series | pd.DataFrame, window: int):
        periods = min(max(3, window // 4), window)
        return returns.rolling(window=window, min_periods=periods).std()

    def rolling_average_correlation(self, returns: pd.DataFrame, window: int) -> pd.Series:
        if returns.empty or returns.shape[1] < 2:
            return pd.Series(index=returns.index, dtype=float, name="avg_pairwise_corr")
        correlations = []
        columns = list(returns.columns)
        periods = min(max(3, window // 4), window)
        for left_pos, left in enumerate(columns):
            for right in columns[left_pos + 1 :]:
                correlations.append(returns[left].rolling(window=window, min_periods=periods).corr(returns[right]))
        if not correlations:
            return pd.Series(index=returns.index, dtype=float, name="avg_pairwise_corr")
        values = pd.concat(correlations, axis=1).mean(axis=1)
        return values.ffill().rename("avg_pairwise_corr")

    def rolling_ols_slope(self, values, window: int, min_periods: int | None = None):
        data = _array(values)
        periods = min_periods or min(max(4, window // 3), window)
        out = np.full(data.shape, np.nan, dtype=float)
        for pos in range(data.size):
            start = max(0, pos - int(window) + 1)
            sample = _finite(data[start : pos + 1])
            if sample.size >= periods:
                out[pos] = _ols_slope(sample)
        return out

    def rolling_ols_tstat(self, values, window: int, min_periods: int | None = None):
        data = _array(values)
        periods = min_periods or min(max(4, window // 3), window)
        out = np.full(data.shape, np.nan, dtype=float)
        for pos in range(data.size):
            start = max(0, pos - int(window) + 1)
            sample = _finite(data[start : pos + 1])
            if sample.size >= periods:
                out[pos] = _ols_tstat(sample)
        return out

    def rolling_ols_slope_tstat(self, series: pd.Series, window: int) -> pd.DataFrame:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        periods = min(max(4, window // 3), window)
        slope = self.rolling_ols_slope(values, window, periods)
        tstat = self.rolling_ols_tstat(values, window, periods)
        return pd.DataFrame({"slope": slope, "tstat": tstat}, index=series.index)

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


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _min_periods(window: int, min_periods: int | None = None) -> int:
    return min_periods or min(max(3, int(window) // 4), int(window))


def _rolling_series(values, window: int, min_periods: int | None = None) -> pd.core.window.rolling.Rolling:
    return pd.Series(_array(values)).rolling(window=int(window), min_periods=_min_periods(window, min_periods))


def _aligned_error(actual, forecast) -> np.ndarray:
    left = _array(actual)
    right = _array(forecast)
    mask = np.isfinite(left) & np.isfinite(right)
    return right[mask] - left[mask]


def _chi2_sf(statistic: float, dof: int) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.sf(statistic, dof))
    except Exception:
        z = ((statistic / max(dof, 1)) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _ols_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return np.nan
    x = np.arange(values.size, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return np.nan
    return float(np.dot(x, values - values.mean()) / denominator)


def _ols_tstat(values: np.ndarray) -> float:
    n = values.size
    if n < 4:
        return np.nan
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return np.nan
    slope = float(np.dot(x, values - values.mean()) / denominator)
    residuals = values - (values.mean() + slope * x)
    sse = float(np.dot(residuals, residuals))
    se = math.sqrt((sse / max(n - 2, 1)) / denominator) if sse > 0 else 0.0
    if se <= 0:
        return 0.0
    return float(slope / se)
