from __future__ import annotations

import pandas as pd

from cmva.backtest.engine import run_historical_backtest


def test_backtest_shifts_weights_by_one_period():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    returns = pd.DataFrame({"BTCUSDT": [0.01, 0.02, -0.01], "ETHUSDT": [0.01, 0.02, -0.01]}, index=index)
    forecast = pd.Series([0.001, 0.001, 0.001], index=index)
    regimes = pd.Series(["ASSET_SPECIFIC"] * 3, index=index)
    result = run_historical_backtest(returns, forecast, regimes, 0.01, 1.0, 0.0, 0.0)
    strategy = result.returns["regime_aware_vol_target"]
    assert strategy.iloc[0] == 0.0
    assert strategy.iloc[1] == 0.02


def test_transaction_costs_reduce_returns_when_turnover_positive():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    returns = pd.DataFrame({"BTCUSDT": [0.0, 0.02, 0.02], "ETHUSDT": [0.0, 0.02, 0.02]}, index=index)
    forecast = pd.Series([0.001, 0.001, 0.001], index=index)
    regimes = pd.Series(["ASSET_SPECIFIC"] * 3, index=index)
    no_cost = run_historical_backtest(returns, forecast, regimes, 0.01, 1.0, 0.0, 0.0)
    with_cost = run_historical_backtest(returns, forecast, regimes, 0.01, 1.0, 10.0, 10.0)
    assert with_cost.returns["regime_aware_vol_target"].sum() < no_cost.returns["regime_aware_vol_target"].sum()
