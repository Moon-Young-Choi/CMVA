from __future__ import annotations

import pandas as pd

from cmva.features import compute_feature_bundle


def test_rolling_features_do_not_use_future_data(synthetic_candles):
    frame = synthetic_candles(periods=40)
    target_time = pd.Timestamp("2026-01-01T20:00:00Z")
    full = frame.copy()
    future_mask = full["open_time"] > target_time
    full.loc[future_mask, "close"] = full.loc[future_mask, "close"] * 10.0
    truncated = frame.loc[frame["open_time"] <= target_time].copy()

    full_features = compute_feature_bundle(full, short_window=6, medium_window=12, long_window=18).features
    truncated_features = compute_feature_bundle(truncated, short_window=6, medium_window=12, long_window=18).features

    pd.testing.assert_series_equal(
        full_features.loc[target_time],
        truncated_features.loc[target_time],
        check_names=False,
    )
