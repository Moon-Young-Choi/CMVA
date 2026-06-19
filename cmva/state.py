"""Application state shared by services and TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from cmva.analysis_types import DiagnosticSnapshot, MethodStep

AppMode = Literal["BOOTSTRAP", "LIVE", "PAUSED", "DEGRADED", "ERROR"]


@dataclass
class AppState:
    mode: AppMode = "BOOTSTRAP"
    latest_closed_time: pd.Timestamp | None = None
    current_regime: str | None = None
    current_shock_type: str | None = None
    forecast_vol: float | None = None
    backtest_summary: dict[str, float | str | None] = field(default_factory=dict)
    model_status: dict[str, object] = field(default_factory=dict)
    data_status: dict[str, object] = field(default_factory=dict)
    bootstrap_progress: dict[str, object] = field(default_factory=dict)
    range_status: dict[str, object] = field(default_factory=dict)
    trend_buffers: dict[str, list[float]] = field(default_factory=dict)
    latest_diagnostics: DiagnosticSnapshot = field(default_factory=DiagnosticSnapshot)
    process_timeline: list[MethodStep] = field(default_factory=list)
    last_stat_test_run: pd.Timestamp | None = None
    websocket_status: str = "disconnected"
    paused: bool = False
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-300:]

    def push_trend(self, name: str, value: float | None, max_points: int = 80) -> None:
        if value is None or pd.isna(value):
            return
        values = self.trend_buffers.setdefault(name, [])
        values.append(float(value))
        self.trend_buffers[name] = values[-max_points:]
