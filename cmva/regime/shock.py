"""Volatility shock classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmva.features.returns import equal_weight_basket_return
from cmva.features.volatility import realized_volatility

NORMAL = "NORMAL"
MODERATE_SHOCK = "MODERATE_SHOCK"
IDIOSYNCRATIC_VOL_SHOCK = "IDIOSYNCRATIC_VOL_SHOCK"
SYSTEMIC_VOL_SHOCK = "SYSTEMIC_VOL_SHOCK"
VOL_REGIME_BUILDUP = "VOL_REGIME_BUILDUP"


@dataclass
class ShockSnapshot:
    shock_type: str
    shock_score: float
    rv_jump_ratio: float
    shock_breadth: float


def classify_shock(
    shock_score: float,
    rv_jump_ratio: float = 1.0,
    shock_breadth: float = 0.0,
    avg_corr: float = 0.0,
    dispersion: float = 0.0,
    severe_threshold: float = 3.0,
    moderate_threshold: float = 2.0,
    breadth_threshold: float = 0.60,
    corr_high_threshold: float = 0.50,
    dispersion_high_threshold: float | None = None,
) -> str:
    dispersion_high = dispersion_high_threshold is not None and dispersion >= dispersion_high_threshold
    if shock_score >= severe_threshold and shock_breadth >= breadth_threshold and avg_corr >= corr_high_threshold:
        return SYSTEMIC_VOL_SHOCK
    if shock_score >= severe_threshold and shock_breadth < breadth_threshold and dispersion_high:
        return IDIOSYNCRATIC_VOL_SHOCK
    if rv_jump_ratio >= 2.0:
        return VOL_REGIME_BUILDUP
    if shock_score >= moderate_threshold:
        return MODERATE_SHOCK
    return NORMAL


def compute_shock_series(
    returns: pd.DataFrame,
    forecast_vol: pd.Series,
    avg_corr: pd.Series,
    dispersion: pd.Series,
    window: int = 24,
    severe_threshold: float = 3.0,
    moderate_threshold: float = 2.0,
) -> pd.DataFrame:
    basket = equal_weight_basket_return(returns)
    # A shock at t must be scored against the forecast made at t-1.
    forecast = forecast_vol.reindex(basket.index).shift(1).replace(0.0, np.nan)
    shock_score = (basket.abs() / forecast).replace([np.inf, -np.inf], np.nan)
    rv_6h = realized_volatility(basket, 6, min_periods=3)
    rv_base = rv_6h.rolling(window=max(window, 30), min_periods=6).median().shift(1).replace(0.0, np.nan)
    rv_jump_ratio = (rv_6h / rv_base).replace([np.inf, -np.inf], np.nan)
    asset_scale = realized_volatility(returns, window).shift(1).replace(0.0, np.nan)
    asset_shocks = returns.abs() / asset_scale
    breadth = (asset_shocks > moderate_threshold).mean(axis=1)
    dispersion_threshold = dispersion.expanding(min_periods=10).quantile(0.75).shift(1)

    labels = []
    for idx in basket.index:
        labels.append(
            classify_shock(
                shock_score=float(_safe_at(shock_score, idx, 0.0)),
                rv_jump_ratio=float(_safe_at(rv_jump_ratio, idx, 1.0)),
                shock_breadth=float(_safe_at(breadth, idx, 0.0)),
                avg_corr=float(_safe_at(avg_corr, idx, 0.0)),
                dispersion=float(_safe_at(dispersion, idx, 0.0)),
                severe_threshold=severe_threshold,
                moderate_threshold=moderate_threshold,
                dispersion_high_threshold=float(_safe_at(dispersion_threshold, idx, np.inf)),
            )
        )

    return pd.DataFrame(
        {
            "shock_type": labels,
            "shock_score": shock_score,
            "rv_jump_ratio": rv_jump_ratio,
            "shock_breadth": breadth,
        },
        index=basket.index,
    )


def _safe_at(series: pd.Series, index, default: float) -> float:
    value = series.get(index, default)
    if pd.isna(value):
        return default
    return float(value)
