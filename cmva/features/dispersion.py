"""Cross-sectional dispersion features."""

from __future__ import annotations

import pandas as pd

from cmva.features.returns import equal_weight_basket_return
from cmva.features.volatility import realized_volatility


def rolling_dispersion(returns: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="dispersion")
    asset_vol = realized_volatility(returns, window, min_periods=min_periods).mean(axis=1)
    basket_vol = realized_volatility(equal_weight_basket_return(returns), window, min_periods=min_periods)
    return (asset_vol - basket_vol).rename("dispersion")
