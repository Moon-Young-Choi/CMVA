"""Candle representation and Binance payload parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def utc_timestamp(value: int | float | str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, (int, float)):
        ts = pd.to_datetime(int(value), unit="ms", utc=True)
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class Candle:
    symbol: str
    interval: str
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool

    @classmethod
    def from_rest_kline(cls, symbol: str, interval: str, raw: list[Any]) -> "Candle":
        return cls(
            symbol=symbol.upper(),
            interval=interval,
            open_time=utc_timestamp(raw[0]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
            close_time=utc_timestamp(raw[6]),
            is_closed=True,
        )

    @classmethod
    def from_websocket_payload(cls, payload: dict[str, Any]) -> "Candle":
        data = payload.get("data", payload)
        kline = data["k"]
        return cls(
            symbol=str(kline["s"]).upper(),
            interval=str(kline["i"]),
            open_time=utc_timestamp(kline["t"]),
            close_time=utc_timestamp(kline["T"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            is_closed=bool(kline["x"]),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_closed": self.is_closed,
        }


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    if not candles:
        return candle_frame_schema()
    frame = pd.DataFrame([candle.to_record() for candle in candles])
    return normalize_candle_frame(frame)


def candle_frame_schema() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "interval",
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_closed",
        ]
    )


def normalize_candle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return candle_frame_schema()
    normalized = frame.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized["interval"] = normalized["interval"].astype(str)
    normalized["open_time"] = pd.to_datetime(normalized["open_time"], utc=True)
    normalized["close_time"] = pd.to_datetime(normalized["close_time"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["is_closed"] = normalized["is_closed"].astype(bool)
    return normalized.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def closed_only(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return normalize_candle_frame(frame)
    return normalize_candle_frame(frame.loc[frame["is_closed"]].copy())
