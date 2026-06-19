"""Terminal-native time-series graph renderers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from cmva.time_ranges import RangeMetadata, slice_by_time_range

_BARS = "▁▂▃▄▅▆▇█"


def time_series_panel(
    series: pd.Series,
    title: str,
    range_label: str,
    width: int = 72,
    unit: str = "",
) -> Panel:
    sliced, metadata = slice_by_time_range(series, range_label)
    clean = pd.to_numeric(sliced, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    body = Text()
    body.append(_metadata_line(metadata, unit))
    body.append("\n")
    if clean.empty:
        body.append("No finite data in selected range.")
        return Panel(body, title=title, expand=True)
    body.append(_sparkline(clean, width=width))
    body.append("\n")
    body.append(
        f"min={_fmt(clean.min(), unit)}  max={_fmt(clean.max(), unit)}  "
        f"latest={_fmt(clean.iloc[-1], unit)}"
    )
    return Panel(body, title=f"{title} [{metadata.label}]", expand=True)


def multi_series_panels(
    frame: pd.DataFrame,
    title: str,
    range_label: str,
    columns: list[str] | None = None,
    width: int = 72,
    unit: str = "",
) -> Group:
    selected = columns or list(frame.columns)
    panels = []
    for column in selected:
        if column in frame:
            panels.append(time_series_panel(frame[column], f"{title}: {column}", range_label, width=width, unit=unit))
    if not panels:
        panels.append(Panel("No data available.", title=title, expand=True))
    return Group(*panels)


def _sparkline(series: pd.Series, width: int) -> str:
    values = series.to_numpy(dtype=float)
    if len(values) > width:
        positions = np.linspace(0, len(values) - 1, width).round().astype(int)
        values = values[positions]
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not math.isfinite(low) or not math.isfinite(high):
        return "No finite data."
    if high == low:
        return _BARS[len(_BARS) // 2] * len(values)
    chars = []
    for value in values:
        ratio = (float(value) - low) / (high - low)
        idx = max(0, min(len(_BARS) - 1, int(round(ratio * (len(_BARS) - 1)))))
        chars.append(_BARS[idx])
    return "".join(chars)


def _metadata_line(metadata: RangeMetadata, unit: str) -> str:
    start = metadata.start if metadata.start is not None else "-"
    end = metadata.end if metadata.end is not None else "-"
    expected = metadata.expected_points if metadata.expected_points is not None else "all"
    suffix = f" unit={unit}" if unit else ""
    return f"start={start}  end={end}  n={metadata.actual_points}/{expected}{suffix}"


def _fmt(value: object, unit: str) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(parsed):
        return "-"
    if unit == "%":
        return f"{parsed * 100:.2f}%"
    if unit == "x":
        return f"{parsed:.2f}x"
    return f"{parsed:.6f}"
