from __future__ import annotations

import pandas as pd

from cmva.data.candle import Candle
from cmva.data.validation import validate_candles
from cmva.features.returns import close_matrix


def test_unclosed_candles_are_not_used_in_feature_calculation():
    closed = Candle(
        "BTCUSDT",
        "1h",
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:59:59.999Z"),
        100.0,
        101.0,
        99.0,
        100.0,
        1.0,
        True,
    )
    open_candle = Candle(
        "BTCUSDT",
        "1h",
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T01:59:59.999Z"),
        100.0,
        200.0,
        99.0,
        200.0,
        1.0,
        False,
    )
    frame = pd.DataFrame([closed.to_record(), open_candle.to_record()])
    close = close_matrix(frame)
    assert list(close.index) == [closed.open_time]
    assert close.loc[closed.open_time, "BTCUSDT"] == 100.0


def test_duplicate_timestamps_are_detected(synthetic_candles):
    frame = synthetic_candles(periods=4)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_candles(duplicated, symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert report.count("duplicates") > 0
    assert not report.is_valid


def test_invalid_ohlc_candles_are_rejected(synthetic_candles):
    frame = synthetic_candles(periods=4)
    frame.loc[0, "high"] = frame.loc[0, "low"] * 0.5
    report = validate_candles(frame, symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert report.count("ohlc") > 0
    assert not report.is_valid


def test_websocket_closed_flag_parsing():
    payload = {
        "data": {
            "k": {
                "t": 1767225600000,
                "T": 1767229199999,
                "s": "BTCUSDT",
                "i": "1h",
                "o": "100.0",
                "h": "110.0",
                "l": "99.0",
                "c": "105.0",
                "v": "123.0",
                "x": False,
            }
        }
    }
    candle = Candle.from_websocket_payload(payload)
    assert candle.symbol == "BTCUSDT"
    assert candle.is_closed is False
