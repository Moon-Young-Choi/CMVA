"""Naive volatility targeting policy."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolTargetPolicy:
    def __init__(self, target_vol_per_period: float, max_leverage: float = 1.5) -> None:
        self.target_vol_per_period = target_vol_per_period
        self.max_leverage = max_leverage

    def target_exposure(self, forecast_vol: float, regime: str | None = None) -> float:
        if forecast_vol <= 0 or not np.isfinite(forecast_vol):
            return 0.0
        return float(np.clip(self.target_vol_per_period / forecast_vol, 0.0, self.max_leverage))

    def exposure_series(self, forecast_vol: pd.Series) -> pd.Series:
        return forecast_vol.apply(self.target_exposure).rename("exposure")
