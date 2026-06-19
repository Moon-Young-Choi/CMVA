"""Walk-forward model validation for market-state analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelValidationResult:
    metrics: dict[str, float | str | None]
    losses: pd.DataFrame
    calibration_by_decile: pd.DataFrame
    realized_vol_by_regime: pd.DataFrame
    sample_size: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None

    def summary(self) -> dict[str, object]:
        return {
            "validation_observations": self.sample_size,
            "validation_start": self.start,
            "validation_end": self.end,
            **self.metrics,
        }


def run_walk_forward_validation(
    basket_returns: pd.Series,
    forecast_vol: pd.Series,
    ewma_vol: pd.Series,
    realized_vol: pd.Series,
    regimes: pd.Series | None = None,
) -> ModelValidationResult:
    returns = _clean_series(basket_returns).rename("return")
    aligned = pd.concat(
        [
            returns,
            _clean_series(forecast_vol).reindex(returns.index).shift(1).rename("garch"),
            _clean_series(ewma_vol).reindex(returns.index).shift(1).rename("ewma"),
            _clean_series(realized_vol).reindex(returns.index).shift(1).rename("naive"),
        ],
        axis=1,
    ).dropna()
    for column in ("garch", "ewma", "naive"):
        aligned = aligned.loc[aligned[column] > 0]

    if aligned.empty:
        return ModelValidationResult({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, None, None)

    realized_variance = aligned["return"].pow(2).rename("realized_variance")
    loss_columns: dict[str, pd.Series] = {}
    metrics: dict[str, float | str | None] = {}
    for model in ("garch", "ewma", "naive"):
        forecast_variance = aligned[model].pow(2)
        error = realized_variance - forecast_variance
        qlike = _qlike(realized_variance, forecast_variance)
        loss_columns[f"{model}_squared_error"] = error.pow(2)
        loss_columns[f"{model}_absolute_error"] = error.abs()
        loss_columns[f"{model}_qlike"] = qlike
        metrics[f"{model}_rmse"] = float(math.sqrt(error.pow(2).mean()))
        metrics[f"{model}_mae"] = float(error.abs().mean())
        metrics[f"{model}_qlike"] = float(qlike.mean())
        metrics[f"{model}_forecast_realized_corr"] = _safe_corr(aligned[model], aligned["return"].abs())

    qlike_scores = {model: metrics.get(f"{model}_qlike") for model in ("garch", "ewma", "naive")}
    finite_scores = {key: value for key, value in qlike_scores.items() if value is not None and np.isfinite(float(value))}
    metrics["best_qlike_model"] = min(finite_scores, key=finite_scores.get) if finite_scores else None

    losses = pd.DataFrame(loss_columns, index=aligned.index)
    calibration = _calibration_by_decile(aligned["garch"], aligned["return"])
    regime_table = _realized_vol_by_regime(aligned["return"], regimes)
    return ModelValidationResult(
        metrics=metrics,
        losses=losses,
        calibration_by_decile=calibration,
        realized_vol_by_regime=regime_table,
        sample_size=len(aligned),
        start=pd.Timestamp(aligned.index.min()) if len(aligned) else None,
        end=pd.Timestamp(aligned.index.max()) if len(aligned) else None,
    )


def _clean_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _qlike(realized_variance: pd.Series, forecast_variance: pd.Series) -> pd.Series:
    aligned = pd.concat([realized_variance.rename("realized"), forecast_variance.rename("forecast")], axis=1).dropna()
    aligned = aligned.loc[aligned["forecast"] > 0]
    return (aligned["realized"] / aligned["forecast"] + np.log(aligned["forecast"])).rename("qlike")


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(aligned) < 3 or aligned["left"].nunique() < 2 or aligned["right"].nunique() < 2:
        return None
    value = aligned["left"].corr(aligned["right"])
    if pd.isna(value):
        return None
    return float(value)


def _calibration_by_decile(forecast_vol: pd.Series, returns: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([forecast_vol.rename("forecast_vol"), returns.abs().rename("realized_abs_return")], axis=1).dropna()
    if len(aligned) < 10 or aligned["forecast_vol"].nunique() < 2:
        return pd.DataFrame()
    try:
        aligned["forecast_decile"] = pd.qcut(aligned["forecast_vol"], q=10, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    grouped = aligned.groupby("forecast_decile", observed=True)
    return grouped.agg(
        forecast_vol=("forecast_vol", "mean"),
        realized_abs_return=("realized_abs_return", "mean"),
        observations=("realized_abs_return", "size"),
    )


def _realized_vol_by_regime(returns: pd.Series, regimes: pd.Series | None) -> pd.DataFrame:
    if regimes is None or regimes.empty:
        return pd.DataFrame()
    aligned = pd.concat([returns.rename("return"), regimes.reindex(returns.index).rename("regime")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()
    return aligned.groupby("regime", observed=True).agg(
        realized_vol=("return", "std"),
        mean_abs_return=("return", lambda value: float(value.abs().mean())),
        observations=("return", "size"),
    )
