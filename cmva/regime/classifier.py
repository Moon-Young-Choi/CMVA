"""Market regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmva.regime.thresholds import expanding_quantile

SYSTEMIC_RISK = "SYSTEMIC_RISK"
IDIOSYNCRATIC_HIGH_VOL = "IDIOSYNCRATIC_HIGH_VOL"
ASSET_SPECIFIC = "ASSET_SPECIFIC"
QUIET_CORRELATED = "QUIET_CORRELATED"


def classify_regime(
    forecast_vol: float,
    market_vol: float,
    avg_corr: float,
    pca1_share: float,
    dispersion: float,
    thresholds: dict[str, float],
) -> str:
    vol_high = forecast_vol >= thresholds.get("forecast_vol_high", np.inf) or market_vol >= thresholds.get("market_vol_high", np.inf)
    corr_high = avg_corr >= thresholds.get("corr_high", np.inf)
    pca_high = pca1_share >= thresholds.get("pca_high", np.inf)
    dispersion_high = dispersion >= thresholds.get("dispersion_high", np.inf)
    vol_low_or_mid = forecast_vol <= thresholds.get("forecast_vol_high", np.inf)
    corr_low = avg_corr <= thresholds.get("corr_low", -np.inf)
    pca_low = pca1_share <= thresholds.get("pca_low", -np.inf)

    if vol_high and corr_high and pca_high:
        return SYSTEMIC_RISK
    if vol_high and not corr_high and dispersion_high:
        return IDIOSYNCRATIC_HIGH_VOL
    if vol_low_or_mid and corr_low and pca_low:
        return ASSET_SPECIFIC
    return QUIET_CORRELATED


def classify_regime_series(
    features: pd.DataFrame,
    forecast_vol: pd.Series,
    min_history: int = 30,
) -> pd.Series:
    index = features.index
    forecast = forecast_vol.reindex(index)
    thresholds = pd.DataFrame(
        {
            "forecast_vol_high": expanding_quantile(forecast, 0.75, min_history),
            "market_vol_high": expanding_quantile(features["market_vol"], 0.75, min_history),
            "corr_high": expanding_quantile(features["avg_pairwise_corr"], 0.75, min_history),
            "corr_low": expanding_quantile(features["avg_pairwise_corr"], 0.40, min_history),
            "pca_high": expanding_quantile(features["pca1_share"], 0.75, min_history),
            "pca_low": expanding_quantile(features["pca1_share"], 0.40, min_history),
            "dispersion_high": expanding_quantile(features["dispersion"], 0.75, min_history),
        },
        index=index,
    )
    labels: list[str] = []
    for idx in index:
        row = features.loc[idx]
        current_thresholds = thresholds.loc[idx].dropna().to_dict()
        if len(current_thresholds) < 7:
            labels.append(QUIET_CORRELATED)
            continue
        labels.append(
            classify_regime(
                forecast_vol=float(_finite_or_default(forecast.loc[idx], 0.0)),
                market_vol=float(_finite_or_default(row.get("market_vol"), 0.0)),
                avg_corr=float(_finite_or_default(row.get("avg_pairwise_corr"), 0.0)),
                pca1_share=float(_finite_or_default(row.get("pca1_share"), 0.0)),
                dispersion=float(_finite_or_default(row.get("dispersion"), 0.0)),
                thresholds=current_thresholds,
            )
        )
    return pd.Series(labels, index=index, name="regime")


def _finite_or_default(value, default: float) -> float:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return default
    return float(value)
