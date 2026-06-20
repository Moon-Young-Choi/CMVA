from __future__ import annotations

import pandas as pd
import pytest

from cmva.config import CMVAConfig
from cmva.data.websocket_client import BinanceWebSocketClient
from cmva.engine.interval import bars_for_duration, latest_closed_open_time, normalize_interval


def test_default_interval_and_duration_windows_convert_to_bars():
    config = CMVAConfig()

    assert config.interval == "1h"
    assert config.forecast_horizon == "1 bar = next 1 hour"
    assert config.window_bar_counts["volatility_window"] == 24
    assert config.window_bar_counts["correlation_window"] == 7 * 24


@pytest.mark.parametrize(
    ("interval", "bars"),
    [
        ("1m", 1440),
        ("5m", 288),
        ("15m", 96),
        ("1h", 24),
        ("4h", 6),
        ("1d", 1),
    ],
)
def test_duration_window_to_bar_count(interval, bars):
    assert bars_for_duration("24h", interval) == bars


def test_unsupported_interval_is_rejected():
    with pytest.raises(ValueError):
        normalize_interval("2m")


def test_interval_propagates_to_websocket_stream_url():
    client = BinanceWebSocketClient(["BTCUSDT", "ETHUSDT"], interval="15m", base_url="wss://example.test")

    assert "btcusdt@kline_15m" in client.stream_url
    assert "ethusdt@kline_15m" in client.stream_url


def test_latest_closed_open_time_is_conservative():
    now = pd.Timestamp("2026-01-01T10:03:00Z")

    assert latest_closed_open_time(now, "15m") == pd.Timestamp("2026-01-01T09:45:00Z")
