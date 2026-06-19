"""Model-selection hook.

MVP only exposes GARCH as a candidate, but this module records a real selection
result so the app can schedule and display model-selection status without
pretending additional models exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cmva.models.registry import create_model


@dataclass(frozen=True)
class ModelSelectionResult:
    selected_model: str
    success: bool
    aic: float | None
    bic: float | None
    message: str


def select_volatility_model(returns: pd.Series, candidates: tuple[str, ...] = ("garch",)) -> ModelSelectionResult:
    best: ModelSelectionResult | None = None
    for candidate in candidates:
        model = create_model(candidate)
        fit = model.fit(returns)
        result = ModelSelectionResult(candidate, fit.success, fit.aic, fit.bic, fit.message)
        if best is None:
            best = result
        elif result.success and (best.aic is None or (result.aic is not None and result.aic < best.aic)):
            best = result
    if best is None:
        return ModelSelectionResult("garch", False, None, None, "no model candidates")
    return best


def selected_model_name() -> str:
    return "garch"
