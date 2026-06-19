"""Feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmva.features.correlation import rolling_average_pairwise_correlation
from cmva.features.dispersion import rolling_dispersion
from cmva.features.pca import rolling_pca1_share
from cmva.features.returns import close_matrix, compute_log_returns, equal_weight_basket_return
from cmva.features.rolling_beta import rolling_beta
from cmva.features.trend import rolling_autocorrelation, rolling_ols_slope, rolling_ols_tstat, up_down_ratio
from cmva.features.volatility import ewma_volatility, range_based_volatility, realized_volatility, rolling_percentile


@dataclass
class FeatureBundle:
    close: pd.DataFrame
    returns: pd.DataFrame
    features: pd.DataFrame


def compute_feature_bundle(
    candles: pd.DataFrame,
    short_window: int = 24,
    medium_window: int = 168,
    long_window: int = 720,
    trend_window: int | None = None,
) -> FeatureBundle:
    close = close_matrix(candles)
    returns = compute_log_returns(close)
    basket = equal_weight_basket_return(returns)
    trend_periods = trend_window or short_window
    market_vol = realized_volatility(basket, short_window).rename("market_vol")
    realized_short = realized_volatility(returns, short_window).mean(axis=1).rename("avg_asset_vol_short")
    realized_medium = realized_volatility(returns, medium_window).mean(axis=1).rename("avg_asset_vol_medium")
    realized_long = realized_volatility(returns, long_window).mean(axis=1).rename("avg_asset_vol_long")
    ewma = ewma_volatility(basket, span=short_window).rename("ewma_vol")
    range_vol = range_based_volatility(candles, short_window)
    vol_percentile = rolling_percentile(market_vol, long_window).rename("vol_percentile")
    rolling_mean = market_vol.rolling(long_window, min_periods=min(10, long_window)).mean()
    rolling_std = market_vol.rolling(long_window, min_periods=min(10, long_window)).std()
    vol_z = ((market_vol - rolling_mean) / rolling_std).rename("vol_z_score")
    vol_of_vol = market_vol.rolling(medium_window, min_periods=min(10, medium_window)).std().rename("vol_of_vol")
    avg_corr = rolling_average_pairwise_correlation(returns, long_window).rename("avg_pairwise_corr")
    beta = rolling_beta(returns, benchmark_symbol="BTCUSDT", window=medium_window)
    beta_mean = beta.drop(columns=["BTCUSDT"], errors="ignore").mean(axis=1).rename("avg_btc_beta")
    pca1 = rolling_pca1_share(returns, long_window).rename("pca1_share")
    dispersion = rolling_dispersion(returns, short_window).rename("dispersion")
    basket_price = close.mean(axis=1, skipna=True)
    log_basket_price = np.log(basket_price.where(basket_price > 0)).replace([np.inf, -np.inf], np.nan)
    trend_slope = rolling_ols_slope(log_basket_price, trend_periods)
    trend_tstat = rolling_ols_tstat(log_basket_price, trend_periods)
    trend_strength = (trend_slope / market_vol.replace(0.0, np.nan)).rename("trend_strength")
    trend_autocorr = rolling_autocorrelation(basket, trend_periods).rename("trend_autocorr")
    candle_ratio = up_down_ratio(basket, trend_periods)
    features = pd.concat(
        [
            basket.rename("basket_return"),
            market_vol,
            realized_short,
            realized_medium,
            realized_long,
            ewma,
            range_vol,
            vol_percentile,
            vol_z,
            vol_of_vol,
            avg_corr,
            beta_mean,
            pca1,
            dispersion,
            trend_slope,
            trend_tstat,
            trend_strength,
            trend_autocorr,
            candle_ratio,
        ],
        axis=1,
    ).sort_index()
    return FeatureBundle(close=close, returns=returns, features=features)
