"""GARCH volatility model with EWMA fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmva.models.base import FitResult, VolForecast


@dataclass
class GarchVolatilityModel:
    min_observations: int = 100
    ewma_span: int = 24
    max_fit_observations: int | None = 5000

    name: str = "garch"

    def __post_init__(self) -> None:
        self._result = None
        self._fit_result = FitResult(False, self.name, message="not fitted")

    def fit(self, returns: pd.Series) -> FitResult:
        clean = _clean_returns(returns)
        fit_sample = clean.tail(self.max_fit_observations) if self.max_fit_observations else clean
        if len(fit_sample) < self.min_observations:
            self._result = None
            self._fit_result = FitResult(
                False,
                self.name,
                message=f"not enough observations for GARCH: {len(fit_sample)} < {self.min_observations}",
            )
            return self._fit_result
        try:
            from arch import arch_model

            model = arch_model(fit_sample * 100.0, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT", rescale=False)
            self._result = model.fit(disp="off", show_warning=False)
            params = {str(key): float(value) for key, value in self._result.params.items()}
            self._fit_result = FitResult(
                True,
                self.name,
                params=params,
                aic=float(self._result.aic),
                bic=float(self._result.bic),
                message="ok",
            )
            return self._fit_result
        except Exception as exc:
            self._result = None
            self._fit_result = FitResult(False, self.name, message=f"GARCH fit failed: {exc}")
            return self._fit_result

    def forecast_one_step(self, returns: pd.Series) -> VolForecast:
        clean = _clean_returns(returns)
        if clean.empty:
            return VolForecast(0.0, 0.0, model_name=self.name, degraded=True, message="no returns")
        if self._result is None:
            self.fit(clean)
        if self._result is not None:
            try:
                forecast = self._result.forecast(horizon=1, reindex=False)
                variance_pct = float(forecast.variance.iloc[-1, 0])
                mean_pct = _mean_from_result(self._result)
                variance = max(variance_pct, 0.0) / 10000.0
                return VolForecast(
                    volatility=math.sqrt(variance),
                    variance=variance,
                    mean=mean_pct / 100.0,
                    model_name=self.name,
                    degraded=False,
                    message="ok",
                )
            except Exception as exc:
                return _ewma_forecast(clean, self.ewma_span, f"GARCH forecast failed: {exc}")
        return _ewma_forecast(clean, self.ewma_span, self._fit_result.message)


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _mean_from_result(result) -> float:
    for key in ("mu", "Const"):
        if key in result.params:
            return float(result.params[key])
    return 0.0


def _ewma_forecast(returns: pd.Series, span: int, message: str) -> VolForecast:
    clean = _clean_returns(returns)
    if clean.empty:
        return VolForecast(0.0, 0.0, model_name="ewma_fallback", degraded=True, message=message)
    variance = float(clean.pow(2).ewm(span=span, min_periods=1, adjust=False).mean().iloc[-1])
    variance = max(variance, 0.0)
    return VolForecast(
        volatility=math.sqrt(variance),
        variance=variance,
        mean=0.0,
        model_name="ewma_fallback",
        degraded=True,
        message=message,
    )


def historical_garch_forecast(
    returns: pd.Series,
    refit_frequency: int = 24,
    min_observations: int = 100,
    ewma_span: int = 24,
    max_fit_observations: int | None = 5000,
    max_refits: int | None = 12,
) -> pd.Series:
    clean = _clean_returns(returns)
    if clean.empty:
        return pd.Series(dtype=float, name="forecast_vol")
    fallback = clean.pow(2).ewm(span=ewma_span, min_periods=1, adjust=False).mean().pow(0.5)
    try:
        import arch  # noqa: F401
    except Exception:
        return fallback.rename("forecast_vol")

    forecasts = pd.Series(index=clean.index, dtype=float, name="forecast_vol")
    model = GarchVolatilityModel(
        min_observations=min_observations,
        ewma_span=ewma_span,
        max_fit_observations=max_fit_observations,
    )
    start_pos = 0
    if max_refits is not None and refit_frequency > 0:
        start_pos = max(0, len(clean) - refit_frequency * max_refits)
        if start_pos > 0:
            forecasts.iloc[:start_pos] = fallback.iloc[:start_pos]
    last_refit_pos = -refit_frequency
    for pos, idx in enumerate(clean.index):
        if pos < start_pos:
            continue
        history = clean.iloc[: pos + 1]
        if len(history) < min_observations:
            forecasts.loc[idx] = fallback.loc[idx]
            continue
        if model._result is None or pos - last_refit_pos >= refit_frequency:
            fit = model.fit(history)
            if fit.success:
                last_refit_pos = pos
            else:
                forecasts.loc[idx] = fallback.loc[idx]
                continue
        forecast = model.forecast_one_step(history)
        forecasts.loc[idx] = forecast.volatility if forecast.volatility > 0 else fallback.loc[idx]
    return forecasts.ffill().fillna(fallback).rename("forecast_vol")
