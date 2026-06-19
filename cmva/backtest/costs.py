"""Transaction cost helpers."""

from __future__ import annotations

import pandas as pd


def turnover_from_exposure(exposure: pd.Series) -> pd.Series:
    return exposure.fillna(0.0).diff().abs().fillna(exposure.fillna(0.0).abs())


def cost_from_turnover(turnover: pd.Series, transaction_cost_bps: float, slippage_bps: float) -> pd.Series:
    return turnover * ((transaction_cost_bps + slippage_bps) / 10000.0)
