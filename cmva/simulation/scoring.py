"""Simulation score calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from cmva.simulation.schemas import ScoreResult


EPS = 1e-12


def information_criterion_weights(records: Iterable[dict[str, Any]], method: str = "bic_weighted_percentile") -> dict[str, float]:
    rows = [record for record in records if _finite(record.get("forecast_volatility"))]
    if not rows:
        return {}
    if method == "equal_weighted":
        return _equal_weights(rows)
    if method == "best_qlike_model":
        finite = [row for row in rows if _finite(row.get("qlike"))]
        selected = min(finite or rows, key=lambda row: _value(row.get("qlike"), default=float("inf")))
        return {str(row["candidate"]): 1.0 if row is selected else 0.0 for row in rows}
    metric = "aic" if method == "aic_weighted_percentile" else "bic"
    finite = [row for row in rows if bool(row.get("converged")) and _finite(row.get(metric))]
    if not finite:
        return _equal_weights(rows)
    values = np.asarray([float(row[metric]) for row in finite], dtype=float)
    deltas = values - float(np.nanmin(values))
    raw = np.exp(-0.5 * np.clip(deltas, 0.0, 700.0))
    total = float(raw.sum())
    if total <= 0 or not np.isfinite(total):
        return _equal_weights(finite)
    return {str(row["candidate"]): float(weight / total) for row, weight in zip(finite, raw, strict=True)}


def score_origin(
    run_id: str,
    symbol: str,
    origin_time: str,
    records: list[dict[str, Any]],
    prior_forecasts: dict[str, list[float]],
    scoring_method: str,
    missing_rate: float = 0.0,
) -> ScoreResult:
    usable = [record for record in records if _finite(record.get("forecast_volatility"))]
    if not usable:
        return ScoreResult(
            run_id=run_id,
            symbol=symbol,
            origin_time=origin_time,
            selected_model_id=None,
            selected_model_family=None,
            selected_estimator=None,
            model_selection_method=scoring_method,
            model_weight_summary={},
            volatility_score_0_100=None,
            volatility_level_label="산출 불가",
            volatility_score_basis="수렴한 모델이 없어 변동성 점수를 계산하지 못했습니다.",
            volatility_confidence_0_1=0.0,
            expectation_score_0_100=None,
            expectation_level_label="산출 불가",
            expectation_score_basis="수렴한 모델이 없어 기대값 점수를 계산하지 못했습니다.",
            expectation_confidence_0_1=0.0,
            trend_score_minus100_100=None,
            trend_level_label="산출 불가",
            trend_score_basis="수렴한 모델이 없어 추세 점수를 계산하지 못했습니다.",
            trend_confidence_0_1=0.0,
            seasonality_score_0_100=None,
            seasonality_level_label="산출 불가",
            seasonality_score_basis="수렴한 모델이 없어 계절성 점수를 계산하지 못했습니다.",
            seasonality_confidence_0_1=0.0,
            warnings=["수렴한 후보 모델이 없습니다."],
        )
    weights = information_criterion_weights(usable, scoring_method)
    if not weights:
        weights = _equal_weights(usable)
    selected = max(usable, key=lambda row: weights.get(str(row["candidate"]), 0.0))
    warnings: list[str] = []
    calibration_counts = []
    weighted_percentile = 0.0
    for row in usable:
        candidate = str(row["candidate"])
        weight = weights.get(candidate, 0.0)
        history = prior_forecasts.get(candidate, [])
        calibration_counts.append(len(history))
        percentile = percentile_rank(history, float(row.get("forecast_volatility") or 0.0))
        if len(history) < 5:
            warnings.append(f"{candidate}: 과거 보정 표본이 부족하여 낮은 신뢰도로 점수화했습니다.")
        weighted_percentile += weight * percentile
    vol_score = int(round(max(0.0, min(100.0, 100.0 * weighted_percentile))))
    expectation_score = _weighted_expectation_score(usable, weights)
    trend_score = _weighted_trend_score(usable, weights)
    seasonality_score = _seasonality_score(usable)
    convergence_rate = sum(1 for row in records if bool(row.get("converged"))) / max(1, len(records))
    agreement = _model_agreement(usable, weights)
    calibration_factor = min(1.0, (max(calibration_counts) if calibration_counts else 0) / 10.0)
    missing_factor = max(0.0, 1.0 - min(1.0, missing_rate * 2.0))
    confidence = _clamp01(0.30 + 0.25 * convergence_rate + 0.20 * agreement + 0.15 * calibration_factor + 0.10 * missing_factor)
    expectation_confidence = _clamp01(confidence * 0.82)
    trend_confidence = _clamp01(confidence * 0.88)
    seasonality_confidence = _clamp01(confidence * (0.75 if seasonality_score > 0 else 0.55))
    selected_weights = {candidate: round(float(weight), 6) for candidate, weight in weights.items() if weight > 0}
    basis_metric = "BIC" if scoring_method == "bic_weighted_percentile" else "AIC" if scoring_method == "aic_weighted_percentile" else scoring_method
    return ScoreResult(
        run_id=run_id,
        symbol=symbol,
        origin_time=origin_time,
        selected_model_id=selected.get("model_id"),
        selected_model_family=selected.get("model_family"),
        selected_estimator=selected.get("estimator"),
        model_selection_method=scoring_method,
        model_weight_summary=selected_weights,
        volatility_score_0_100=vol_score,
        volatility_level_label=volatility_label(vol_score),
        volatility_score_basis=f"{basis_metric} 가중치와 과거 예측 변동성 분위수를 결합했습니다.",
        volatility_confidence_0_1=round(confidence, 4),
        expectation_score_0_100=round(expectation_score, 4),
        expectation_level_label=expectation_label(expectation_score),
        expectation_score_basis="예측 평균을 예측 변동성으로 표준화한 신호대잡음비를 모델 가중 평균했습니다.",
        expectation_confidence_0_1=round(expectation_confidence, 4),
        trend_score_minus100_100=round(trend_score, 4),
        trend_level_label=trend_label(trend_score),
        trend_score_basis="예측 평균 방향성과 추세 계열 후보 모델의 정보를 가중 결합했습니다.",
        trend_confidence_0_1=round(trend_confidence, 4),
        seasonality_score_0_100=round(seasonality_score, 4),
        seasonality_level_label=seasonality_label(seasonality_score),
        seasonality_score_basis="계절성 후보의 정보기준 개선폭과 가용성을 안정화해 산출했습니다.",
        seasonality_confidence_0_1=round(seasonality_confidence, 4),
        aic=_optional_float(selected.get("aic")),
        bic=_optional_float(selected.get("bic")),
        hqic=_optional_float(selected.get("hqic")),
        qlike=_optional_float(selected.get("qlike")),
        rmse_loss=_optional_float(selected.get("rmse_loss")),
        mae_loss=_optional_float(selected.get("mae_loss")),
        forecast_volatility=_optional_float(selected.get("forecast_volatility")),
        realized_volatility=_optional_float(selected.get("realized_volatility")),
        forecast_mean=_optional_float(selected.get("forecast_mean")),
        realized_return=_optional_float(selected.get("realized_return")),
        warnings=sorted(set(warnings)),
    )


def percentile_rank(history: list[float], value: float) -> float:
    clean = np.asarray([item for item in history if np.isfinite(item)], dtype=float)
    if clean.size == 0 or not np.isfinite(value):
        return 0.5
    return float((np.sum(clean <= value) + 0.5 * np.sum(clean == value)) / max(1, clean.size))


def volatility_label(score: float | int | None) -> str:
    if score is None:
        return "산출 불가"
    if score < 20:
        return "매우 낮은 변동성"
    if score < 40:
        return "낮은 변동성"
    if score < 60:
        return "보통 변동성"
    if score < 80:
        return "높은 변동성"
    return "극단적 변동성"


def expectation_label(score: float | None) -> str:
    if score is None:
        return "산출 불가"
    if score < 20:
        return "강한 하방 기대"
    if score < 40:
        return "약한 하방 기대"
    if score < 60:
        return "중립"
    if score < 80:
        return "약한 상방 기대"
    return "강한 상방 기대"


def trend_label(score: float | None) -> str:
    if score is None:
        return "산출 불가"
    if score <= -60:
        return "강한 하락 추세"
    if score <= -20:
        return "약한 하락 추세"
    if score < 20:
        return "중립 추세"
    if score < 60:
        return "약한 상승 추세"
    return "강한 상승 추세"


def seasonality_label(score: float | None) -> str:
    if score is None:
        return "산출 불가"
    if score < 20:
        return "계절성 거의 없음"
    if score < 40:
        return "약한 계절성"
    if score < 60:
        return "보통 계절성"
    if score < 80:
        return "강한 계절성"
    return "매우 강한 계절성"


def _weighted_expectation_score(records: list[dict[str, Any]], weights: dict[str, float]) -> float:
    score = 0.0
    total = 0.0
    for row in records:
        candidate = str(row["candidate"])
        weight = weights.get(candidate, 0.0)
        sigma = max(float(row.get("forecast_volatility") or 0.0), EPS)
        mu = float(row.get("forecast_mean") or 0.0)
        score += weight * (50.0 + 50.0 * math.tanh(mu / sigma))
        total += weight
    return max(0.0, min(100.0, score / max(total, EPS)))


def _weighted_trend_score(records: list[dict[str, Any]], weights: dict[str, float]) -> float:
    score = 0.0
    total = 0.0
    for row in records:
        candidate = str(row["candidate"])
        weight = weights.get(candidate, 0.0)
        sigma = max(float(row.get("forecast_volatility") or 0.0), EPS)
        mu = float(row.get("forecast_mean") or 0.0)
        family_boost = 1.25 if str(row.get("model_family")) in {"trend", "combined"} else 1.0
        score += weight * 100.0 * math.tanh(family_boost * mu / sigma)
        total += weight
    return max(-100.0, min(100.0, score / max(total, EPS)))


def _seasonality_score(records: list[dict[str, Any]]) -> float:
    seasonal = [
        row
        for row in records
        if "season" in str(row.get("model_id", "")).lower() or "stl" in str(row.get("model_id", "")).lower()
    ]
    if not seasonal:
        return 0.0
    baseline_bic = min(
        [float(row["bic"]) for row in records if str(row.get("model_id")) in {"naive_mean", "constant_mean"} and _finite(row.get("bic"))]
        or [float("nan")]
    )
    improvements = []
    for row in seasonal:
        if _finite(row.get("bic")) and np.isfinite(baseline_bic):
            improvements.append(max(0.0, baseline_bic - float(row["bic"])))
        elif row.get("converged"):
            improvements.append(1.0)
    if not improvements:
        return 0.0
    strength = 1.0 - math.exp(-float(np.nanmean(improvements)) / 10.0)
    return max(0.0, min(100.0, 100.0 * strength))


def _model_agreement(records: list[dict[str, Any]], weights: dict[str, float]) -> float:
    weighted = []
    for row in records:
        weight = weights.get(str(row["candidate"]), 0.0)
        if weight > 0 and _finite(row.get("forecast_volatility")):
            weighted.append((weight, float(row["forecast_volatility"])))
    if len(weighted) <= 1:
        return 0.65
    values = np.asarray([value for _, value in weighted], dtype=float)
    mean = float(np.mean(values))
    if mean <= EPS:
        return 0.5
    cv = float(np.std(values) / mean)
    return _clamp01(1.0 / (1.0 + cv))


def _equal_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    weight = 1.0 / len(rows)
    return {str(row["candidate"]): weight for row in rows}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _value(value: Any, default: float) -> float:
    return float(value) if _finite(value) else default


def _optional_float(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
