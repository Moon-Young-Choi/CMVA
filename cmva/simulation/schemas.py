"""Dataclasses used by simulation artifacts and API payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STATUS_LABELS_KO = {
    "pending": "대기 중",
    "preparing_data": "데이터 준비 중",
    "validating_data": "데이터 검증 중",
    "running": "실행 중",
    "completed": "완료",
    "failed": "실패",
    "cancelled": "취소됨",
}


@dataclass
class SimulationProgress:
    run_id: str
    status: str = "pending"
    status_label: str = "대기 중"
    progress_pct: float = 0.0
    current_step: str = ""
    current_origin_time: str | None = None
    active_symbol: str | None = None
    active_model: str | None = None
    completed_fits: int = 0
    total_fits: int = 0
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    latest_scores: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status_label"] = STATUS_LABELS_KO.get(self.status, self.status_label or self.status)
        return payload


@dataclass
class SimulationStepResult:
    run_id: str
    symbol: str
    origin_index: int
    origin_time: str
    realization_time: str
    train_start_time: str
    train_end_time: str
    evaluation_start_time: str
    evaluation_end_time: str
    model_id: str
    model_family: str
    target: str
    candidate: str
    estimator: str
    params: dict[str, Any]
    converged: bool
    fit_status: str
    fit_message: str
    log_likelihood: float | None
    nobs: int
    parameter_count: int
    aic: float | None
    bic: float | None
    hqic: float | None
    qlike: float | None
    rmse_loss: float | None
    mae_loss: float | None
    forecast_mean: float | None
    realized_mean: float | None
    realized_return: float | None
    forecast_variance: float | None
    realized_variance: float | None
    forecast_volatility: float | None
    realized_volatility: float | None
    residual_diagnostics: dict[str, Any]
    rank: int | None
    composite_rank_score: float | None
    fit_time_ms: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreResult:
    run_id: str
    symbol: str
    origin_time: str
    selected_model_id: str | None
    selected_model_family: str | None
    selected_estimator: str | None
    model_selection_method: str
    model_weight_summary: dict[str, float]
    volatility_score_0_100: int | None
    volatility_level_label: str
    volatility_score_basis: str
    volatility_confidence_0_1: float
    expectation_score_0_100: float | None
    expectation_level_label: str
    expectation_score_basis: str
    expectation_confidence_0_1: float
    trend_score_minus100_100: float | None
    trend_level_label: str
    trend_score_basis: str
    trend_confidence_0_1: float
    seasonality_score_0_100: float | None
    seasonality_level_label: str
    seasonality_score_basis: str
    seasonality_confidence_0_1: float
    aic: float | None = None
    bic: float | None = None
    hqic: float | None = None
    qlike: float | None = None
    rmse_loss: float | None = None
    mae_loss: float | None = None
    forecast_volatility: float | None = None
    realized_volatility: float | None = None
    forecast_mean: float | None = None
    realized_return: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
