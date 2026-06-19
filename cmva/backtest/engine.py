"""Historical walk-forward backtest."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cmva.backtest.benchmarks import btc_buy_and_hold, equal_weight_buy_and_hold, naive_vol_target
from cmva.backtest.costs import cost_from_turnover, turnover_from_exposure
from cmva.backtest.metrics import equity_curve, performance_metrics
from cmva.features.returns import equal_weight_basket_return
from cmva.policy.regime_vol_target import RegimeVolTargetPolicy
from cmva.policy.vol_target import VolTargetPolicy


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    exposure: pd.Series
    turnover: pd.Series
    metrics: dict[str, dict[str, float]]
    average_exposure_by_regime: dict[str, float]
    return_by_regime: dict[str, float]


def run_historical_backtest(
    returns: pd.DataFrame,
    forecast_vol: pd.Series,
    regimes: pd.Series,
    target_vol_per_period: float,
    max_leverage: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    periods_per_year: int = 365 * 24,
) -> BacktestResult:
    basket = equal_weight_basket_return(returns).fillna(0.0)
    forecast = forecast_vol.reindex(basket.index).ffill()
    regimes = regimes.reindex(basket.index).ffill().fillna("QUIET_CORRELATED")
    naive_policy = VolTargetPolicy(target_vol_per_period, max_leverage)
    regime_policy = RegimeVolTargetPolicy(target_vol_per_period, max_leverage)

    btc = btc_buy_and_hold(returns).reindex(basket.index).fillna(0.0)
    equal_weight = equal_weight_buy_and_hold(returns).reindex(basket.index).fillna(0.0)
    naive_returns, naive_exposure = naive_vol_target(basket, forecast, naive_policy)
    regime_exposure = regime_policy.exposure_series(forecast, regimes).reindex(basket.index).fillna(0.0)
    turnover = turnover_from_exposure(regime_exposure)
    costs = cost_from_turnover(turnover, transaction_cost_bps, slippage_bps)

    # Exposure decided at t is applied to return observed at t+1.
    shifted_exposure = regime_exposure.shift(1).fillna(0.0)
    regime_returns = (shifted_exposure * basket - costs).rename("regime_aware_vol_target")

    result_returns = pd.concat([btc, equal_weight, naive_returns, regime_returns], axis=1).fillna(0.0)
    result_equity = result_returns.apply(equity_curve)
    metrics = {
        column: performance_metrics(result_returns[column], periods_per_year=periods_per_year)
        for column in result_returns.columns
    }
    metrics["regime_aware_vol_target"]["turnover"] = float(turnover.sum())
    metrics["regime_aware_vol_target"]["total_cost_impact"] = float(costs.sum())
    average_exposure_by_regime = regime_exposure.groupby(regimes).mean().astype(float).to_dict()
    return_by_regime = regime_returns.groupby(regimes).mean().astype(float).to_dict()
    return BacktestResult(
        returns=result_returns,
        equity=result_equity,
        exposure=regime_exposure,
        turnover=turnover,
        metrics=metrics,
        average_exposure_by_regime=average_exposure_by_regime,
        return_by_regime=return_by_regime,
    )
