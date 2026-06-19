"""Candle interval and duration conversion helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CandleInterval:
    value: str
    seconds: int
    bars_per_day: float
    supported_by_rest: bool = True
    supported_by_ws: bool = True


SUPPORTED_INTERVALS: dict[str, CandleInterval] = {
    "1m": CandleInterval("1m", 60, 24 * 60),
    "3m": CandleInterval("3m", 3 * 60, 24 * 20),
    "5m": CandleInterval("5m", 5 * 60, 24 * 12),
    "15m": CandleInterval("15m", 15 * 60, 24 * 4),
    "30m": CandleInterval("30m", 30 * 60, 24 * 2),
    "1h": CandleInterval("1h", 60 * 60, 24),
    "2h": CandleInterval("2h", 2 * 60 * 60, 12),
    "4h": CandleInterval("4h", 4 * 60 * 60, 6),
    "6h": CandleInterval("6h", 6 * 60 * 60, 4),
    "8h": CandleInterval("8h", 8 * 60 * 60, 3),
    "12h": CandleInterval("12h", 12 * 60 * 60, 2),
    "1d": CandleInterval("1d", 24 * 60 * 60, 1),
    "3d": CandleInterval("3d", 3 * 24 * 60 * 60, 1 / 3),
    "1w": CandleInterval("1w", 7 * 24 * 60 * 60, 1 / 7),
    "1M": CandleInterval("1M", 30 * 24 * 60 * 60, 1 / 30),
}

_DURATION_RE = re.compile(r"^([1-9][0-9]*)(s|m|h|d|w|y)$")


def normalize_interval(value: str) -> str:
    raw = str(value).strip()
    if raw == "1M":
        return raw
    normalized = raw.lower()
    if normalized not in SUPPORTED_INTERVALS:
        supported = ", ".join(SUPPORTED_INTERVALS)
        raise ValueError(f"unsupported candle interval: {value}. Supported intervals: {supported}")
    return normalized


def interval_to_timedelta(value: str) -> pd.Timedelta:
    interval = SUPPORTED_INTERVALS[normalize_interval(value)]
    return pd.Timedelta(seconds=interval.seconds)


def duration_to_seconds(value: str) -> int:
    raw = str(value).strip().lower()
    match = _DURATION_RE.match(raw)
    if match is None:
        raise ValueError(f"unsupported duration: {value}")
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = {
        "s": 1,
        "m": 30 * 24 * 60 * 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
        "y": 365 * 24 * 60 * 60,
    }[unit]
    return amount * multiplier


def bars_for_duration(duration: str, interval: str, minimum: int = 1) -> int:
    interval_seconds = SUPPORTED_INTERVALS[normalize_interval(interval)].seconds
    bars = math.ceil(duration_to_seconds(duration) / interval_seconds)
    return max(int(bars), minimum)


def periods_per_year(interval: str) -> int:
    seconds = SUPPORTED_INTERVALS[normalize_interval(interval)].seconds
    return max(1, int(round(365 * 24 * 60 * 60 / seconds)))


def describe_horizon(interval: str, horizon_bars: int) -> str:
    interval_seconds = SUPPORTED_INTERVALS[normalize_interval(interval)].seconds
    total_seconds = max(1, int(horizon_bars)) * interval_seconds
    label = _format_seconds(total_seconds)
    plural = "bar" if int(horizon_bars) == 1 else "bars"
    return f"{int(horizon_bars)} {plural} = next {label}"


def latest_closed_open_time(now: pd.Timestamp, interval: str) -> pd.Timestamp:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    delta = interval_to_timedelta(interval)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    elapsed_ns = (ts - epoch).value
    interval_ns = delta.value
    floored = epoch + pd.Timedelta(elapsed_ns // interval_ns * interval_ns, unit="ns")
    return floored - delta


def _format_seconds(seconds: int) -> str:
    units = [
        ("day", 24 * 60 * 60),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ]
    for name, unit_seconds in units:
        if seconds % unit_seconds == 0 and seconds >= unit_seconds:
            value = seconds // unit_seconds
            suffix = "" if value == 1 else "s"
            return f"{value} {name}{suffix}"
    return f"{seconds} seconds"
