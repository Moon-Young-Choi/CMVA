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
    model_id: str | None = None
    model_family: str | None = None
    target: str | None = None
    estimator: str | None = None
    log_likelihood: float | None = None
    nobs: int | None = None
    parameter_count: int | None = None
    hqic: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    converged: bool | None = None
    forecast_mean: float | None = None
    forecast_variance: float | None = None
    forecast_volatility: float | None = None
    model_family_label: str | None = None
    degraded: bool = False


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
