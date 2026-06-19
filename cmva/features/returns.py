"""Return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmva.data.candle import closed_only, normalize_candle_frame


def close_matrix(candles: pd.DataFrame) -> pd.DataFrame:
    data = closed_only(normalize_candle_frame(candles))
    if data.empty:
        return pd.DataFrame()
    close = data.pivot_table(index="open_time", columns="symbol", values="close", aggfunc="last")
    return close.sort_index()


def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return close.copy()
    return np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)


def equal_weight_basket_return(returns: pd.DataFrame) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="basket_return")
    return returns.mean(axis=1, skipna=True).rename("basket_return")
