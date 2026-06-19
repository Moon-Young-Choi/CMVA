"""Policy protocol."""

from __future__ import annotations

from typing import Protocol


class ExposurePolicy(Protocol):
    def target_exposure(self, forecast_vol: float, regime: str | None = None) -> float:
        ...
