"""Symbol and basket volatility forecasting service."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cmva.features.returns import equal_weight_basket_return
from cmva.models.base import VolForecast
from cmva.models.garch import GarchVolatilityModel


@dataclass
class ForecastSnapshot:
    asset_forecasts: dict[str, VolForecast]
    basket_forecast: VolForecast
    refit_countdown: int
    status: dict[str, object] = field(default_factory=dict)


class VolatilityForecaster:
    def __init__(self, refit_frequency: int = 24, min_observations: int = 100) -> None:
        self.refit_frequency = refit_frequency
        self.min_observations = min_observations
        self.models: dict[str, GarchVolatilityModel] = {}
        self.basket_model = GarchVolatilityModel(min_observations=min_observations)
        self.closed_since_refit = 0

    def forecast(self, returns: pd.DataFrame, force_refit: bool = False) -> ForecastSnapshot:
        clean = returns.replace([np.inf, -np.inf], np.nan)
        do_refit = force_refit or self.closed_since_refit == 0 or self.closed_since_refit >= self.refit_frequency
        asset_forecasts: dict[str, VolForecast] = {}
        status: dict[str, object] = {"degraded_symbols": [], "refit": do_refit}

        for symbol in clean.columns:
            series = clean[symbol].dropna()
            model = self.models.setdefault(symbol, GarchVolatilityModel(min_observations=self.min_observations))
            if do_refit:
                fit = model.fit(series)
                if not fit.success:
                    status.setdefault("fit_messages", {})[symbol] = fit.message
            forecast = model.forecast_one_step(series)
            asset_forecasts[symbol] = forecast
            if forecast.degraded:
                status["degraded_symbols"].append(symbol)

        basket = equal_weight_basket_return(clean).dropna()
        if do_refit:
            fit = self.basket_model.fit(basket)
            status["basket_fit"] = fit.message
        basket_forecast = self.basket_model.forecast_one_step(basket)
        if do_refit:
            self.closed_since_refit = 1
        else:
            self.closed_since_refit += 1
        return ForecastSnapshot(
            asset_forecasts=asset_forecasts,
            basket_forecast=basket_forecast,
            refit_countdown=max(self.refit_frequency - self.closed_since_refit, 0),
            status=status,
        )
