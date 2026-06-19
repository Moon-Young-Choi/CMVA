"""Feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cmva.features.correlation import rolling_average_pairwise_correlation
from cmva.features.dispersion import rolling_dispersion
from cmva.features.pca import rolling_pca1_share
from cmva.features.returns import close_matrix, compute_log_returns, equal_weight_basket_return
from cmva.features.rolling_beta import rolling_beta
from cmva.features.volatility import ewma_volatility, realized_volatility


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
) -> FeatureBundle:
    close = close_matrix(candles)
    returns = compute_log_returns(close)
    basket = equal_weight_basket_return(returns)
    market_vol = realized_volatility(basket, short_window).rename("market_vol")
    realized_short = realized_volatility(returns, short_window).mean(axis=1).rename("avg_asset_vol_short")
    realized_medium = realized_volatility(returns, medium_window).mean(axis=1).rename("avg_asset_vol_medium")
    realized_long = realized_volatility(returns, long_window).mean(axis=1).rename("avg_asset_vol_long")
    ewma = ewma_volatility(basket).rename("ewma_vol")
    avg_corr = rolling_average_pairwise_correlation(returns, long_window).rename("avg_pairwise_corr")
    beta = rolling_beta(returns, benchmark_symbol="BTCUSDT", window=medium_window)
    beta_mean = beta.drop(columns=["BTCUSDT"], errors="ignore").mean(axis=1).rename("avg_btc_beta")
    pca1 = rolling_pca1_share(returns, long_window).rename("pca1_share")
    dispersion = rolling_dispersion(returns, short_window).rename("dispersion")
    features = pd.concat(
        [
            basket.rename("basket_return"),
            market_vol,
            realized_short,
            realized_medium,
            realized_long,
            ewma,
            avg_corr,
            beta_mean,
            pca1,
            dispersion,
        ],
        axis=1,
    ).sort_index()
    return FeatureBundle(close=close, returns=returns, features=features)
