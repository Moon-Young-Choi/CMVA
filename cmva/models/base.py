"""Model protocols and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class FitResult:
    success: bool
    model_name: str
    params: dict[str, float] = field(default_factory=dict)
    aic: float | None = None
    bic: float | None = None
    message: str = ""


@dataclass
class VolForecast:
    volatility: float
    variance: float
    mean: float = 0.0
    model_name: str = "unknown"
    degraded: bool = False
    message: str = ""


class VolatilityModel(Protocol):
    name: str

    def fit(self, returns: pd.Series) -> FitResult:
        ...

    def forecast_one_step(self, returns: pd.Series) -> VolForecast:
        ...
