"""Time range parsing and slicing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

DEFAULT_ALLOWED_TIME_RANGES = ["1d", "1w", "1m", "3m", "6m", "1y", "all"]

_PRESET_HOURS = {
    "1d": 24,
    "1w": 7 * 24,
    "1m": 30 * 24,
    "3m": 90 * 24,
    "6m": 180 * 24,
    "1y": 365 * 24,
}
_CUSTOM_RE = re.compile(r"^([1-9][0-9]*)(h|d|w|m|y)$")


@dataclass(frozen=True)
class TimeRange:
    raw: str
    normalized: str
    label: str
    hours: int | None


@dataclass(frozen=True)
class RangeMetadata:
    label: str
    normalized: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    expected_points: int | None
    actual_points: int

    def to_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "range": self.normalized,
            "start": self.start,
            "end": self.end,
            "expected_points": self.expected_points,
            "actual_points": self.actual_points,
        }


def parse_time_range(value: str) -> TimeRange:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("time range is required")
    if normalized == "all":
        return TimeRange(raw=value, normalized="all", label="ALL", hours=None)
    if normalized in _PRESET_HOURS:
        return TimeRange(raw=value, normalized=normalized, label=normalized.upper(), hours=_PRESET_HOURS[normalized])
    match = _CUSTOM_RE.match(normalized)
    if match is None:
        raise ValueError(f"unsupported time range: {value}")
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = {"h": 1, "d": 24, "w": 7 * 24, "m": 30 * 24, "y": 365 * 24}[unit]
    return TimeRange(raw=value, normalized=normalized, label=normalized.upper(), hours=amount * multiplier)


def normalize_time_range(value: str) -> str:
    return parse_time_range(value).normalized


def slice_by_time_range(
    data: pd.Series | pd.DataFrame,
    value: str,
    latest: pd.Timestamp | None = None,
) -> tuple[pd.Series | pd.DataFrame, RangeMetadata]:
    parsed = parse_time_range(value)
    if data.empty:
        return data.copy(), RangeMetadata(parsed.label, parsed.normalized, None, None, parsed.hours, 0)
    if not isinstance(data.index, pd.DatetimeIndex):
        sliced = data.copy()
        return sliced, RangeMetadata(parsed.label, parsed.normalized, None, None, parsed.hours, len(sliced))
    index = pd.to_datetime(data.index, utc=True)
    normalized_data = data.copy()
    normalized_data.index = index
    end = pd.Timestamp(latest) if latest is not None else index.max()
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    if parsed.hours is None:
        sliced = normalized_data.loc[normalized_data.index <= end].copy()
    else:
        start = end - pd.Timedelta(hours=max(parsed.hours - 1, 0))
        sliced = normalized_data.loc[(normalized_data.index >= start) & (normalized_data.index <= end)].copy()
    start_value = sliced.index.min() if not sliced.empty else None
    end_value = sliced.index.max() if not sliced.empty else None
    metadata = RangeMetadata(
        label=parsed.label,
        normalized=parsed.normalized,
        start=start_value,
        end=end_value,
        expected_points=parsed.hours,
        actual_points=len(sliced),
    )
    return sliced, metadata
