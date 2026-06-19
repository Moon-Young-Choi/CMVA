"""Application state shared by services and TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

AppMode = Literal["BOOTSTRAP", "LIVE", "PAUSED", "DEGRADED", "ERROR"]


@dataclass
class AppState:
    mode: AppMode = "BOOTSTRAP"
    latest_closed_time: pd.Timestamp | None = None
    current_regime: str | None = None
    current_shock_type: str | None = None
    forecast_vol_1h: float | None = None
    target_exposure: float | None = None
    backtest_summary: dict[str, float | str | None] = field(default_factory=dict)
    model_status: dict[str, object] = field(default_factory=dict)
    data_status: dict[str, object] = field(default_factory=dict)
    live_paper_pnl: float = 0.0
    websocket_status: str = "disconnected"
    paused: bool = False
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-300:]
