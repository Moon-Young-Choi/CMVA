"""Regime-aware volatility targeting policy."""

from __future__ import annotations

import pandas as pd

from cmva.policy.vol_target import VolTargetPolicy

REGIME_MULTIPLIERS = {
    "SYSTEMIC_RISK": 0.20,
    "IDIOSYNCRATIC_HIGH_VOL": 0.50,
    "ASSET_SPECIFIC": 1.00,
    "QUIET_CORRELATED": 0.70,
}


class RegimeVolTargetPolicy(VolTargetPolicy):
    def __init__(
        self,
        target_vol_per_period: float,
        max_leverage: float = 1.5,
        regime_multipliers: dict[str, float] | None = None,
    ) -> None:
        super().__init__(target_vol_per_period, max_leverage)
        self.regime_multipliers = regime_multipliers or REGIME_MULTIPLIERS

    def target_exposure(self, forecast_vol: float, regime: str | None = None) -> float:
        base = super().target_exposure(forecast_vol)
        multiplier = self.regime_multipliers.get(regime or "QUIET_CORRELATED", 0.70)
        return min(base * multiplier, self.max_leverage)

    def exposure_series(self, forecast_vol: pd.Series, regimes: pd.Series) -> pd.Series:
        aligned_regimes = regimes.reindex(forecast_vol.index)
        values = [
            self.target_exposure(float(vol), None if pd.isna(regime) else str(regime))
            for vol, regime in zip(forecast_vol, aligned_regimes, strict=False)
        ]
        return pd.Series(values, index=forecast_vol.index, name="regime_aware_exposure")
