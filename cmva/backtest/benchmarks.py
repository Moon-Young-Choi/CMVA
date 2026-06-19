"""Benchmark strategy returns."""

from __future__ import annotations

import pandas as pd

from cmva.features.returns import equal_weight_basket_return
from cmva.policy.vol_target import VolTargetPolicy


def btc_buy_and_hold(returns: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.Series:
    if symbol not in returns.columns:
        return pd.Series(0.0, index=returns.index, name="btc_buy_hold")
    return returns[symbol].fillna(0.0).rename("btc_buy_hold")


def equal_weight_buy_and_hold(returns: pd.DataFrame) -> pd.Series:
    return equal_weight_basket_return(returns).fillna(0.0).rename("equal_weight_buy_hold")


def naive_vol_target(
    basket_returns: pd.Series,
    forecast_vol: pd.Series,
    policy: VolTargetPolicy,
) -> tuple[pd.Series, pd.Series]:
    exposure = policy.exposure_series(forecast_vol).reindex(basket_returns.index).fillna(0.0)
    return (exposure.shift(1).fillna(0.0) * basket_returns.fillna(0.0)).rename("naive_vol_target"), exposure
