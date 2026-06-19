from __future__ import annotations

import pandas as pd

from cmva.validation import run_walk_forward_validation


def test_walk_forward_validation_shifts_forecasts_by_one_period():
    index = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    returns = pd.Series([0.0, 0.10, 0.0, 0.0], index=index)
    forecast = pd.Series([0.05, 999.0, 999.0, 999.0], index=index)
    ewma = pd.Series(0.05, index=index)
    realized = pd.Series(0.05, index=index)

    result = run_walk_forward_validation(returns, forecast, ewma, realized)

    first_loss_time = result.losses.index[0]
    assert first_loss_time == index[1]
    assert result.losses.loc[index[1], "garch_qlike"] < result.losses.loc[index[2], "garch_qlike"]


def test_walk_forward_validation_compares_garch_ewma_and_naive():
    index = pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC")
    returns = pd.Series([0.001 * ((idx % 5) - 2) for idx in range(40)], index=index)
    forecast = pd.Series(0.003, index=index)
    ewma = pd.Series(0.004, index=index)
    realized = returns.rolling(6, min_periods=2).std().fillna(0.003)
    regimes = pd.Series(["QUIET_CORRELATED"] * 40, index=index)

    result = run_walk_forward_validation(returns, forecast, ewma, realized, regimes)

    assert result.sample_size > 20
    assert {"garch_rmse", "ewma_rmse", "naive_rmse", "best_qlike_model"} <= set(result.metrics)
    assert not result.losses.empty
    assert not result.realized_vol_by_regime.empty
