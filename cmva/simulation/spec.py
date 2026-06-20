"""Simulation input specification and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cmva.engine.interval import bars_for_duration, interval_to_timedelta, normalize_interval


SEARCH_MODES = {"fast", "two_stage", "detailed"}
TARGET_VIEWS = {"log_price", "log_return", "both"}
SCORING_METHODS = {
    "bic_weighted_percentile",
    "aic_weighted_percentile",
    "best_qlike_model",
    "equal_weighted",
}
S_ROLES = {"scenario_count", "initial_scale", "simulation_horizon", "custom"}


@dataclass
class SimulationSpec:
    run_name: str
    symbols: list[str]
    interval: str
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    T: str
    dT: str
    S: str
    s_role: str
    forecast_horizon_bars: int = 1
    training_window: str | None = None
    candidate_model_groups: list[str] = field(default_factory=lambda: ["mean", "trend", "volatility", "combined"])
    candidate_model_count: int = 16
    search_mode: str = "two_stage"
    target_view: str = "both"
    scoring_method: str = "bic_weighted_percentile"
    calibration_window: str = "30d"
    timezone_display: str = "Asia/Seoul"
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))
    seed: int | None = None
    run_id: str | None = None
    t_bars: int = 0
    dt_bars: int = 0
    calibration_bars: int = 0

    def __post_init__(self) -> None:
        self.run_name = str(self.run_name).strip()
        if not self.run_name:
            raise ValueError("실행명을 입력해야 합니다.")
        self.symbols = _normalize_symbols(self.symbols)
        self.interval = normalize_interval(self.interval)
        self.data_start = _utc_timestamp(self.data_start, "data_start")
        self.data_end = _utc_timestamp(self.data_end, "data_end")
        if self.data_start >= self.data_end:
            raise ValueError("data_start는 data_end보다 빨라야 합니다.")
        self.T = str(self.T).strip()
        self.dT = str(self.dT).strip()
        if not self.T:
            raise ValueError("T는 필수 입력값입니다.")
        if not self.dT:
            raise ValueError("dT는 필수 입력값입니다.")
        self.S = str(self.S).strip()
        if not self.S:
            raise ValueError("S는 필수 입력값입니다.")
        self.s_role = str(self.s_role or "custom").strip().lower()
        if self.s_role not in S_ROLES:
            raise ValueError("S 역할은 시나리오 수, 초기 스케일, 시뮬레이션 구간, 사용자 정의 중 하나여야 합니다.")
        self.forecast_horizon_bars = int(self.forecast_horizon_bars)
        if self.forecast_horizon_bars <= 0:
            raise ValueError("예측 horizon은 1 이상이어야 합니다.")
        self.search_mode = str(self.search_mode).strip().lower().replace("-", "_")
        if self.search_mode not in SEARCH_MODES:
            raise ValueError("탐색 모드는 fast, two_stage, detailed 중 하나여야 합니다.")
        self.target_view = str(self.target_view).strip().lower()
        if self.target_view not in TARGET_VIEWS:
            raise ValueError("대상 뷰는 log_price, log_return, both 중 하나여야 합니다.")
        self.scoring_method = str(self.scoring_method).strip().lower()
        if self.scoring_method not in SCORING_METHODS:
            raise ValueError("점수화 방식이 지원되지 않습니다.")
        self.candidate_model_groups = _normalize_groups(self.candidate_model_groups)
        self.candidate_model_count = max(1, int(self.candidate_model_count))
        self.training_window = self.training_window or self.T
        self.t_bars = resolve_bar_count(self.T, self.interval, "T")
        self.dt_bars = resolve_bar_count(self.dT, self.interval, "dT")
        self.calibration_bars = resolve_bar_count(self.calibration_window, self.interval, "calibration_window")
        if self.t_bars <= 0 or self.dt_bars <= 0:
            raise ValueError("T와 dT는 0보다 커야 합니다.")
        self.created_at = _utc_timestamp(self.created_at, "created_at")

    @property
    def interval_delta(self) -> pd.Timedelta:
        return interval_to_timedelta(self.interval)

    @property
    def horizon_delta(self) -> pd.Timedelta:
        return self.interval_delta * self.forecast_horizon_bars

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SimulationSpec":
        payload = dict(data)
        payload["symbols"] = _symbols_from_value(payload.get("symbols"))
        payload["candidate_model_groups"] = _groups_from_value(payload.get("candidate_model_groups"))
        return cls(
            run_name=str(payload.get("run_name") or ""),
            symbols=payload["symbols"],
            interval=str(payload.get("interval") or ""),
            data_start=payload.get("data_start"),
            data_end=payload.get("data_end"),
            T=payload.get("T") or payload.get("training_window") or "",
            dT=payload.get("dT") or payload.get("refit_stride") or "",
            S=payload.get("S") or "",
            s_role=str(payload.get("s_role") or "custom"),
            forecast_horizon_bars=int(payload.get("forecast_horizon_bars") or 1),
            training_window=payload.get("training_window"),
            candidate_model_groups=payload["candidate_model_groups"],
            candidate_model_count=int(payload.get("candidate_model_count") or 16),
            search_mode=str(payload.get("search_mode") or "two_stage"),
            target_view=str(payload.get("target_view") or "both"),
            scoring_method=str(payload.get("scoring_method") or "bic_weighted_percentile"),
            calibration_window=str(payload.get("calibration_window") or "30d"),
            timezone_display=str(payload.get("timezone_display") or "Asia/Seoul"),
            created_at=payload.get("created_at") or pd.Timestamp.now(tz="UTC"),
            seed=int(payload["seed"]) if payload.get("seed") not in {None, ""} else None,
            run_id=payload.get("run_id"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationSpec":
        return cls.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "symbols": self.symbols,
            "interval": self.interval,
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "T": self.T,
            "dT": self.dT,
            "S": self.S,
            "s_role": self.s_role,
            "forecast_horizon_bars": self.forecast_horizon_bars,
            "training_window": self.training_window,
            "candidate_model_groups": self.candidate_model_groups,
            "candidate_model_count": self.candidate_model_count,
            "search_mode": self.search_mode,
            "target_view": self.target_view,
            "scoring_method": self.scoring_method,
            "calibration_window": self.calibration_window,
            "timezone_display": self.timezone_display,
            "created_at": self.created_at.isoformat(),
            "seed": self.seed,
            "t_bars": self.t_bars,
            "dt_bars": self.dt_bars,
            "calibration_bars": self.calibration_bars,
        }


def resolve_bar_count(value: str | int, interval: str, field_name: str) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field_name}는 0보다 커야 합니다.")
        return value
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name}는 필수 입력값입니다.")
    lowered = raw.lower().replace("봉", " bars")
    if lowered.isdigit():
        return int(lowered)
    if "bar" in lowered:
        amount = lowered.replace("bars", "").replace("bar", "").strip()
        if amount.isdigit() and int(amount) > 0:
            return int(amount)
        raise ValueError(f"{field_name}의 bar 수를 해석할 수 없습니다.")
    try:
        return bars_for_duration(raw, interval)
    except ValueError as exc:
        raise ValueError(f"{field_name}는 예: 24h, 7d, 20 bars 형식이어야 합니다.") from exc


def _utc_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    if value in {None, ""}:
        raise ValueError(f"{field_name}는 필수 입력값입니다.")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _symbols_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not normalized:
        raise ValueError("심볼은 1개 이상 입력해야 합니다.")
    seen: set[str] = set()
    unique = []
    for symbol in normalized:
        if symbol not in seen:
            unique.append(symbol)
            seen.add(symbol)
    return unique


def _groups_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return ["mean", "trend", "volatility", "combined"]


def _normalize_groups(groups: list[str]) -> list[str]:
    normalized = [str(group).strip().lower() for group in groups if str(group).strip()]
    return normalized or ["mean", "trend", "volatility", "combined"]
