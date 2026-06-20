"""Rolling time-series model laboratory.

The model lab compares statistical descriptions of closed-candle returns and
does not produce account actions or trade-performance artifacts.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from cmva.config import CMVAConfig
from cmva.engine.interval import bars_for_duration
from cmva.native.backend import backend


EPS = 1e-12
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    family: str
    target: str = "log_return"
    order: tuple[int, ...] = ()
    description: str = ""


@dataclass
class ModelLabResult:
    config_hash: str
    data_hash: str
    stage1: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage2: pd.DataFrame = field(default_factory=pd.DataFrame)
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    timelines: pd.DataFrame = field(default_factory=pd.DataFrame)
    rank_heatmap: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected: dict[str, object] = field(default_factory=dict)
    current_state: dict[str, object] = field(default_factory=dict)
    job_status: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "stage1_candidates": int(len(self.stage1)),
            "stage2_candidates": int(len(self.stage2)),
            "leaderboard_rows": int(len(self.leaderboard)),
            "selected_model": self.selected.get("model_id"),
            "selected_family": self.selected.get("family"),
            "best_qlike": self.selected.get("avg_qlike"),
            "rank_stability": self.selected.get("rank_stability"),
            **self.job_status,
        }


def default_candidate_specs(
    groups: list[str] | tuple[str, ...] | None = None,
    target_view: str = "both",
    limit: int = 16,
) -> list[CandidateSpec]:
    enabled = {str(group).strip().lower() for group in (groups or ["mean", "trend", "volatility", "combined"])}
    targets = _target_list(target_view)
    base_specs: list[CandidateSpec] = []
    if "mean" in enabled:
        base_specs.extend(
            [
                CandidateSpec("naive_mean", "mean", description="zero-change mean baseline"),
                CandidateSpec("constant_mean", "mean", description="constant mean Gaussian model"),
                CandidateSpec("ar_1", "mean", order=(1,), description="AR(1) conditional mean"),
                CandidateSpec("ar_2", "mean", order=(2,), description="AR(2) conditional mean"),
                CandidateSpec("ma_1", "mean", order=(1,), description="MA(1) CSS residual recursion"),
                CandidateSpec("arma_1_1", "mean", order=(1, 1), description="ARMA(1,1) CSS approximation"),
                CandidateSpec("arima_1_1_0", "mean", order=(1, 1, 0), description="ARIMA(1,1,0) CSS approximation"),
            ]
        )
    if "trend" in enabled:
        base_specs.extend(
            [
                CandidateSpec("rolling_ols_trend", "trend", order=(24,), description="rolling OLS trend diagnostic"),
                CandidateSpec("seasonal_ar_1", "trend", order=(1, 24), description="seasonal-differenced AR(1) diagnostic"),
                CandidateSpec("stl_trend", "trend", order=(24,), description="lightweight seasonal trend diagnostic"),
            ]
        )
    if "volatility" in enabled:
        base_specs.extend(
            [
                CandidateSpec("realized_vol", "volatility", description="previous realized volatility baseline"),
                CandidateSpec("ewma_vol", "volatility", description="EWMA volatility baseline"),
                CandidateSpec("arch_1", "volatility", order=(1,), description="ARCH(1) Gaussian likelihood"),
                CandidateSpec("garch_1_1", "volatility", order=(1, 1), description="GARCH(1,1) Gaussian likelihood"),
                CandidateSpec("student_t_garch", "volatility", order=(1, 1), description="GARCH(1,1) Student-t likelihood"),
            ]
        )
    if "combined" in enabled:
        base_specs.extend(
            [
                CandidateSpec("ar_1_garch", "combined", order=(1, 1, 1), description="AR(1) mean plus GARCH variance"),
                CandidateSpec("arma_1_1_garch", "combined", order=(1, 1, 1, 1), description="ARMA(1,1) mean plus GARCH variance"),
                CandidateSpec("arima_1_1_0_garch", "combined", order=(1, 1, 0, 1, 1), description="ARIMA mean plus GARCH variance"),
            ]
        )
    specs: list[CandidateSpec] = []
    for base in _interleave_by_family(base_specs):
        for target in targets:
            specs.append(
                CandidateSpec(
                    model_id=base.model_id,
                    family=base.family,
                    target=target,
                    order=base.order,
                    description=base.description,
                )
            )
    seen: set[tuple[str, str]] = set()
    unique: list[CandidateSpec] = []
    for spec in specs:
        key = (spec.model_id, spec.target)
        if key not in seen:
            unique.append(spec)
            seen.add(key)
    return unique[: max(1, int(limit))]


def generate_rolling_origins(
    n_observations: int,
    training_window_bars: int,
    horizon_bars: int = 1,
    refit_stride_bars: int = 1,
) -> list[int]:
    n = int(n_observations)
    horizon = max(1, int(horizon_bars))
    stride = max(1, int(refit_stride_bars))
    if n <= horizon + 2:
        return []
    training = max(4, min(int(training_window_bars), max(4, n - horizon - 1)))
    first = min(training - 1, n - horizon - 1)
    return list(range(first, n - horizon, stride))


def model_lab_cache_key(config: CMVAConfig, candles_or_returns: pd.DataFrame | pd.Series) -> str:
    config_bits = {
        "symbols": config.symbols,
        "interval": config.interval,
        "analysis_period": config.analysis_period,
        "training_window": config.training_window,
        "horizon": config.forecast_horizon_bars,
        "refit_stride": config.refit_stride_bars,
        "search_mode": config.search_mode,
        "candidate_count": config.candidate_model_count,
        "candidate_groups": config.candidate_model_groups,
        "target_view": config.target_view,
    }
    data_hash = hash_time_series(candles_or_returns)
    payload = repr((config_bits, data_hash)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_time_series(data: pd.DataFrame | pd.Series) -> str:
    if data.empty:
        return "empty"
    frame = data.to_frame("value") if isinstance(data, pd.Series) else data.copy()
    frame = frame.sort_index()
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def run_two_stage_search(
    series: pd.Series,
    config: CMVAConfig,
    candidate_specs: list[CandidateSpec] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ModelLabResult:
    started = time.perf_counter()
    target = _clean_series(series)
    config_hash = model_lab_cache_key(config, target)
    data_hash = hash_time_series(target)
    if target.empty:
        _emit_progress(progress_callback, _job_status(started, "no_data", 0, 0, 0))
        return ModelLabResult(
            config_hash=config_hash,
            data_hash=data_hash,
            job_status=_job_status(started, "no_data", 0, 0, 0),
        )

    training_bars = bars_for_duration(config.training_window, config.interval)
    effective_training = min(training_bars, max(8, len(target) // 2))
    origins = generate_rolling_origins(
        len(target),
        training_window_bars=effective_training,
        horizon_bars=config.forecast_horizon_bars,
        refit_stride_bars=config.refit_stride_bars,
    )
    specs = candidate_specs or default_candidate_specs(
        groups=config.candidate_model_groups,
        target_view=config.target_view,
        limit=config.candidate_model_count,
    )
    if not origins:
        _emit_progress(progress_callback, _job_status(started, "not_enough_data", len(specs), 0, 0))
        return ModelLabResult(
            config_hash=config_hash,
            data_hash=data_hash,
            stage1=_empty_result_frame(specs),
            job_status=_job_status(started, "not_enough_data", len(specs), 0, 0),
        )

    stage1_origins = _budget_origins(origins, max_origins=max(8, min(40, len(origins) // 2 or len(origins))))
    expected_stage1_fits = len(specs) * len(stage1_origins)
    _emit_progress(
        progress_callback,
        _job_status(started, "stage1_running", len(specs), 0, 0),
        active_stage="stage1",
        total_fits=expected_stage1_fits,
        completed_fits=0,
        queue_size=expected_stage1_fits,
        progress_pct=0.0,
    )
    stage1 = _evaluate_specs(
        target,
        specs,
        stage1_origins,
        effective_training,
        config.forecast_horizon_bars,
        progress_callback=progress_callback,
        progress_started=started,
        progress_stage="stage1",
        progress_candidate_count=len(specs),
        progress_total_fits=expected_stage1_fits,
    )
    stage1 = _rank_stage(stage1)
    if config.search_mode == "fast":
        stage2_specs = _top_specs(stage1, specs, max(3, min(5, len(specs))))
        stage2_origins = stage1_origins
    elif config.search_mode == "detailed":
        stage2_specs = specs
        stage2_origins = origins
    else:
        stage2_specs = _top_specs(stage1, specs, max(3, min(6, len(specs) // 2 or len(specs))))
        stage2_origins = origins
    expected_stage2_fits = len(stage2_specs) * len(stage2_origins)
    _emit_progress(
        progress_callback,
        _job_status(started, "stage2_running", len(specs), len(stage1), 0),
        active_stage="stage2",
        total_fits=expected_stage2_fits,
        completed_fits=0,
        queue_size=expected_stage2_fits,
        progress_pct=0.0,
    )
    stage2 = _evaluate_specs(
        target,
        stage2_specs,
        stage2_origins,
        effective_training,
        config.forecast_horizon_bars,
        progress_callback=progress_callback,
        progress_started=started,
        progress_stage="stage2",
        progress_candidate_count=len(specs),
        progress_total_fits=expected_stage2_fits,
        progress_stage1_count=len(stage1),
    )
    stage2 = _rank_stage(stage2)
    _emit_progress(
        progress_callback,
        _job_status(started, "diagnostics_running", len(specs), len(stage1), len(stage2)),
        active_stage="diagnostics",
        total_fits=expected_stage2_fits,
        completed_fits=expected_stage2_fits,
        queue_size=0,
        progress_pct=1.0,
    )
    timeline = _build_timeline(target, stage2_specs, stage2_origins, effective_training, config.forecast_horizon_bars)
    heatmap = _rank_heatmap(timeline)
    leaderboard = _leaderboard(stage2, heatmap)
    selected = leaderboard.iloc[0].dropna().to_dict() if not leaderboard.empty else {}
    current_state = _current_state(selected, stage2, target)
    final_job_status = {
        **_job_status(started, "complete", len(specs), len(stage1), len(stage2)),
        "progress_pct": 1.0,
        "queue_size": 0,
        "active_stage": "complete",
        "active_candidate": None,
    }
    _emit_progress(
        progress_callback,
        final_job_status,
        active_stage="complete",
        total_fits=expected_stage1_fits + expected_stage2_fits,
        completed_fits=expected_stage1_fits + expected_stage2_fits,
        queue_size=0,
        progress_pct=1.0,
    )
    return ModelLabResult(
        config_hash=config_hash,
        data_hash=data_hash,
        stage1=stage1,
        stage2=stage2,
        leaderboard=leaderboard,
        timelines=timeline,
        rank_heatmap=heatmap,
        selected=selected,
        current_state=current_state,
        job_status=final_job_status,
    )


def run_model_lab(
    series: pd.Series | Mapping[str, pd.Series],
    config: CMVAConfig,
    progress_callback: ProgressCallback | None = None,
) -> ModelLabResult:
    if isinstance(series, pd.Series):
        return run_two_stage_search(series, config, progress_callback=progress_callback)
    started = time.perf_counter()
    targets = _selected_target_series(series, config.target_view)
    target_frame = pd.DataFrame(targets).sort_index() if targets else pd.DataFrame()
    config_hash = model_lab_cache_key(config, target_frame)
    data_hash = hash_time_series(target_frame)
    if not targets:
        _emit_progress(progress_callback, _job_status(started, "no_data", 0, 0, 0))
        return ModelLabResult(
            config_hash=config_hash,
            data_hash=data_hash,
            job_status=_job_status(started, "no_data", 0, 0, 0),
        )
    all_specs = default_candidate_specs(
        groups=config.candidate_model_groups,
        target_view=config.target_view,
        limit=config.candidate_model_count,
    )
    _emit_progress(
        progress_callback,
        _job_status(started, "queued", len(all_specs), 0, 0),
        queue_size=len(all_specs),
        evaluated_targets=list(targets),
        target_count=len(targets),
    )
    results: list[ModelLabResult] = []
    for target_position, (target_name, target_series) in enumerate(targets.items(), start=1):
        target_specs = [spec for spec in all_specs if spec.target == target_name]
        if target_specs:
            _emit_progress(
                progress_callback,
                _job_status(started, "target_running", len(all_specs), 0, 0),
                queue_size=max(len(targets) - target_position, 0),
                active_target=target_name,
                target_position=target_position,
                target_count=len(targets),
                evaluated_targets=list(targets),
            )

            def relay_progress(status: dict[str, object], target: str = target_name) -> None:
                relayed = dict(status)
                relayed.update(
                    {
                        "candidate_count": len(all_specs),
                        "active_target": target,
                        "target_position": target_position,
                        "target_count": len(targets),
                        "evaluated_targets": list(targets),
                    }
                )
                _emit_progress(progress_callback, relayed)

            results.append(run_two_stage_search(target_series, config, target_specs, progress_callback=relay_progress))
    if not results:
        _emit_progress(
            progress_callback,
            _job_status(started, "no_candidates", len(all_specs), 0, 0),
            evaluated_targets=list(targets),
        )
        return ModelLabResult(
            config_hash=config_hash,
            data_hash=data_hash,
            job_status=_job_status(started, "no_candidates", len(all_specs), 0, 0),
        )
    stage1 = pd.concat([result.stage1 for result in results if not result.stage1.empty], ignore_index=True)
    stage2 = pd.concat([result.stage2 for result in results if not result.stage2.empty], ignore_index=True)
    timeline = pd.concat([result.timelines for result in results if not result.timelines.empty], ignore_index=True)
    heatmap = _rank_heatmap(timeline)
    leaderboard = _leaderboard(stage2, heatmap) if not stage2.empty else pd.DataFrame()
    selected = leaderboard.iloc[0].dropna().to_dict() if not leaderboard.empty else {}
    selected_target = str(selected.get("target", "")) if selected else ""
    selected_series = targets[selected_target] if selected_target in targets else next(iter(targets.values()))
    current_state = _current_state(selected, stage2, selected_series)
    current_state["evaluated_targets"] = list(targets)
    final_job_status = {
        **_job_status(started, "complete", len(all_specs), len(stage1), len(stage2)),
        "evaluated_targets": list(targets),
        "progress_pct": 1.0,
        "queue_size": 0,
        "active_stage": "complete",
        "active_target": None,
        "active_candidate": None,
    }
    _emit_progress(progress_callback, final_job_status)
    return ModelLabResult(
        config_hash=config_hash,
        data_hash=data_hash,
        stage1=stage1,
        stage2=stage2,
        leaderboard=leaderboard,
        timelines=timeline,
        rank_heatmap=heatmap,
        selected=selected,
        current_state=current_state,
        job_status=final_job_status,
    )


def _evaluate_specs(
    values: pd.Series,
    specs: list[CandidateSpec],
    origins: list[int],
    training_window_bars: int,
    horizon_bars: int,
    progress_callback: ProgressCallback | None = None,
    progress_started: float | None = None,
    progress_stage: str = "",
    progress_candidate_count: int = 0,
    progress_total_fits: int = 0,
    progress_stage1_count: int = 0,
) -> pd.DataFrame:
    records = []
    array = values.to_numpy(dtype=float)
    for spec_index, spec in enumerate(specs, start=1):
        start = time.perf_counter()
        losses: list[dict[str, float]] = []
        fit_messages: list[str] = []
        last_fit: _FitSummary | None = None
        for origin in origins:
            train_start = max(0, origin - training_window_bars + 1)
            train = array[train_start : origin + 1]
            realized = float(array[origin + horizon_bars])
            fit = _fit_candidate(spec, train)
            last_fit = fit
            fit_messages.append(fit.message)
            forecast_mean = fit.mean_forecast
            forecast_variance = max(fit.variance_forecast, EPS)
            realized_variance = realized * realized
            qlike = float(backend.qlike(np.asarray([realized_variance]), np.asarray([forecast_variance]))[0])
            losses.append(
                {
                    "realized_variance": realized_variance,
                    "forecast_variance": forecast_variance,
                    "squared_error": (realized_variance - forecast_variance) ** 2,
                    "absolute_error": abs(realized_variance - forecast_variance),
                    "qlike": qlike,
                    "bias": forecast_variance - realized_variance,
                    "forecast_vol": math.sqrt(forecast_variance),
                    "realized_abs": abs(realized),
                    "directional_hit": float(np.sign(forecast_mean) == np.sign(realized)),
                }
            )
        loss_frame = pd.DataFrame(losses)
        fit_time_ms = (time.perf_counter() - start) * 1000.0
        in_sample = last_fit or _FitSummary(False, 0, np.nan, np.nan, np.nan, 0.0, EPS, "not fitted")
        records.append(
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "target": spec.target,
                "candidate": f"{spec.model_id}:{spec.target}",
                "log_likelihood": in_sample.log_likelihood,
                "aic": in_sample.aic,
                "bic": in_sample.bic,
                "hqic": in_sample.hqic,
                "parameter_count": in_sample.parameter_count,
                "converged": bool(in_sample.success),
                "fit_status": "ok" if in_sample.success else "failed",
                "fit_time_ms": fit_time_ms,
                "avg_rmse": backend.rmse(loss_frame["realized_variance"], loss_frame["forecast_variance"])
                if not loss_frame.empty
                else np.nan,
                "avg_mae": backend.mae(loss_frame["realized_variance"], loss_frame["forecast_variance"])
                if not loss_frame.empty
                else np.nan,
                "avg_qlike": float(loss_frame["qlike"].mean()) if not loss_frame.empty else np.nan,
                "forecast_bias": backend.forecast_bias(loss_frame["realized_variance"], loss_frame["forecast_variance"])
                if not loss_frame.empty
                else np.nan,
                "forecast_realized_corr": backend.forecast_realized_correlation(
                    loss_frame["forecast_vol"], loss_frame["realized_abs"]
                )
                if not loss_frame.empty
                else np.nan,
                "directional_agreement_rate": float(loss_frame["directional_hit"].mean())
                if not loss_frame.empty
                else np.nan,
                "evaluation_origins": int(len(origins)),
                "warning": _first_warning(fit_messages),
            }
        )
        if progress_callback is not None and progress_started is not None:
            completed_fits = spec_index * len(origins)
            total_fits = max(progress_total_fits, completed_fits, 1)
            stage1_count = len(records) if progress_stage == "stage1" else progress_stage1_count
            stage2_count = len(records) if progress_stage == "stage2" else 0
            _emit_progress(
                progress_callback,
                _job_status(
                    progress_started,
                    f"{progress_stage}_running" if progress_stage else "running",
                    progress_candidate_count or len(specs),
                    stage1_count,
                    stage2_count,
                ),
                active_stage=progress_stage,
                active_candidate=f"{spec.model_id}:{spec.target}",
                completed_fits=completed_fits,
                total_fits=total_fits,
                queue_size=max(total_fits - completed_fits, 0),
                progress_pct=min(1.0, completed_fits / total_fits),
            )
    return pd.DataFrame(records)


def _build_timeline(
    values: pd.Series,
    specs: list[CandidateSpec],
    origins: list[int],
    training_window_bars: int,
    horizon_bars: int,
) -> pd.DataFrame:
    records = []
    array = values.to_numpy(dtype=float)
    index = list(values.index)
    for origin in origins:
        origin_records = []
        realized = float(array[origin + horizon_bars])
        realized_variance = realized * realized
        for spec in specs:
            train_start = max(0, origin - training_window_bars + 1)
            fit = _fit_candidate(spec, array[train_start : origin + 1])
            forecast_variance = max(fit.variance_forecast, EPS)
            qlike = float(backend.qlike(np.asarray([realized_variance]), np.asarray([forecast_variance]))[0])
            origin_records.append(
                {
                    "origin_time": index[origin],
                    "realization_time": index[origin + horizon_bars],
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "target": spec.target,
                    "candidate": f"{spec.model_id}:{spec.target}",
                    "aic": fit.aic,
                    "bic": fit.bic,
                    "hqic": fit.hqic,
                    "qlike": qlike,
                    "rmse_loss": (realized_variance - forecast_variance) ** 2,
                    "mae_loss": abs(realized_variance - forecast_variance),
                    "forecast_vol": math.sqrt(forecast_variance),
                    "realized_vol": abs(realized),
                }
            )
        frame = pd.DataFrame(origin_records)
        if not frame.empty:
            frame["rank"] = frame["qlike"].rank(method="dense", ascending=True).astype(int)
            records.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(records)


def _rank_heatmap(timeline: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame()
    heatmap = timeline.pivot_table(
        index="candidate",
        columns="realization_time",
        values="rank",
        aggfunc="min",
        observed=True,
    ).sort_index()
    return heatmap


def _leaderboard(stage: pd.DataFrame, heatmap: pd.DataFrame) -> pd.DataFrame:
    if stage.empty:
        return stage
    ranked = stage.copy()
    stability = {}
    if not heatmap.empty:
        for candidate, row in heatmap.iterrows():
            values = pd.to_numeric(row, errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) <= 1:
                stability[candidate] = 1.0
            else:
                stability[candidate] = backend.rank_stability(values)
    ranked["rank_stability"] = ranked["candidate"].map(stability).fillna(0.0)
    for metric, ascending in (
        ("aic", True),
        ("bic", True),
        ("hqic", True),
        ("avg_qlike", True),
        ("avg_rmse", True),
        ("avg_mae", True),
        ("rank_stability", False),
    ):
        ranked[f"{metric}_rank"] = ranked[metric].rank(method="dense", ascending=ascending)
    ranked["composite_rank_score"] = (
        ranked["aic_rank"].fillna(len(ranked))
        + ranked["bic_rank"].fillna(len(ranked))
        + ranked["hqic_rank"].fillna(len(ranked))
        + 2.0 * ranked["avg_qlike_rank"].fillna(len(ranked))
        + ranked["avg_rmse_rank"].fillna(len(ranked))
        + ranked["avg_mae_rank"].fillna(len(ranked))
        + ranked["rank_stability_rank"].fillna(len(ranked))
        + np.where(ranked["converged"], 0.0, len(ranked))
    )
    ranked = ranked.sort_values(["composite_rank_score", "avg_qlike", "aic"], kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _rank_stage(stage: pd.DataFrame) -> pd.DataFrame:
    if stage.empty:
        return stage
    return _leaderboard(stage, pd.DataFrame()).drop(columns=["rank"], errors="ignore")


def _top_specs(stage: pd.DataFrame, specs: list[CandidateSpec], top_n: int) -> list[CandidateSpec]:
    if stage.empty:
        return specs[:top_n]
    candidates = set(stage.sort_values(["composite_rank_score", "avg_qlike"], kind="mergesort")["candidate"].head(top_n))
    return [spec for spec in specs if f"{spec.model_id}:{spec.target}" in candidates]


@dataclass
class _FitSummary:
    success: bool
    parameter_count: int
    log_likelihood: float
    aic: float
    bic: float
    mean_forecast: float
    variance_forecast: float
    message: str

    @property
    def hqic(self) -> float:
        n = max(3, self.parameter_count + 2)
        return _hqic(self.log_likelihood, self.parameter_count, n)


def _fit_candidate(spec: CandidateSpec, train: np.ndarray) -> _FitSummary:
    clean = train[np.isfinite(train)]
    if clean.size < 4:
        return _FitSummary(False, 0, np.nan, np.nan, np.nan, 0.0, EPS, "not enough observations")
    try:
        if spec.model_id == "naive_mean":
            residuals = clean
            mean_forecast = 0.0
            k = 1
        elif spec.model_id == "constant_mean":
            mean = float(np.mean(clean))
            residuals = clean - mean
            mean_forecast = mean
            k = 2
        elif spec.model_id.startswith("ar_") and "garch" not in spec.model_id and "arima" not in spec.model_id:
            p = int(spec.order[0] if spec.order else 1)
            mean_forecast, residuals, k = _ar_fit_forecast(clean, p)
        elif spec.model_id == "ma_1":
            mean_forecast, residuals, k = _ma1_forecast(clean)
        elif spec.model_id == "arma_1_1":
            ar_mean, ar_resid, _ = _ar_fit_forecast(clean, 1)
            ma_mean, residuals, k = _ma1_forecast(ar_resid)
            mean_forecast = ar_mean + ma_mean
            k = 4
        elif spec.model_id.startswith("arima"):
            differenced = np.diff(clean)
            mean_delta, residuals, _ = _ar_fit_forecast(differenced, 1)
            mean_forecast = float(clean[-1] + mean_delta)
            k = 4
        elif spec.model_id == "rolling_ols_trend":
            mean_forecast, residuals, k = _rolling_ols_forecast(clean)
        elif spec.model_id == "seasonal_ar_1":
            mean_forecast, residuals, k = _seasonal_ar_forecast(clean, period=24)
        elif spec.model_id == "stl_trend":
            mean_forecast, residuals, k = _seasonal_mean_forecast(clean, period=24)
        else:
            mean_forecast = 0.0
            residuals = clean
            k = 1

        variance_forecast = _variance_forecast(spec.model_id, residuals)
        log_likelihood = _log_likelihood(residuals, variance_forecast, spec.model_id)
        k = _parameter_count(spec.model_id, k)
        n = max(1, residuals[np.isfinite(residuals)].size)
        return _FitSummary(
            True,
            k,
            log_likelihood,
            _aic(log_likelihood, k),
            _bic(log_likelihood, k, n),
            mean_forecast,
            variance_forecast,
            "ok",
        )
    except Exception as exc:
        fallback_variance = float(np.nanvar(clean, ddof=1)) if clean.size > 1 else EPS
        return _FitSummary(False, 1, np.nan, np.nan, np.nan, 0.0, max(fallback_variance, EPS), str(exc))


def _ar_fit_forecast(values: np.ndarray, p: int) -> tuple[float, np.ndarray, int]:
    p = max(1, int(p))
    if values.size <= p + 2:
        mean = float(np.mean(values))
        return mean, values - mean, 2
    fit = backend.ar_fit(values, p)
    if not fit.get("success"):
        mean = float(np.mean(values))
        return mean, values - mean, p + 2
    beta = np.asarray(fit.get("params", []), dtype=float)
    if beta.size != p + 1 or not np.isfinite(beta).all():
        mean = float(np.mean(values))
        return mean, values - mean, p + 2
    y = values[p:]
    x_cols = [np.ones_like(y)]
    for lag in range(1, p + 1):
        x_cols.append(values[p - lag : values.size - lag])
    x = np.column_stack(x_cols)
    fitted = x @ beta
    residuals = y - fitted
    latest = [1.0, *[values[-lag] for lag in range(1, p + 1)]]
    mean_forecast = float(np.dot(np.asarray(latest), beta))
    return mean_forecast, residuals, p + 2


def _ma1_forecast(values: np.ndarray) -> tuple[float, np.ndarray, int]:
    mean = float(np.mean(values))
    demeaned = values - mean
    theta = 0.35
    residuals = np.empty_like(demeaned)
    prev = 0.0
    for idx, value in enumerate(demeaned):
        residuals[idx] = value - theta * prev
        prev = residuals[idx]
    return float(mean + theta * residuals[-1]), residuals, 3


def _rolling_ols_forecast(values: np.ndarray) -> tuple[float, np.ndarray, int]:
    n = values.size
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0:
        mean = float(np.mean(values))
        return mean, values - mean, 2
    slope = float(np.dot(x_centered, values - values.mean()) / denom)
    intercept = float(values.mean() - slope * x.mean())
    fitted = intercept + slope * x
    return float(intercept + slope * n), values - fitted, 3


def _seasonal_ar_forecast(values: np.ndarray, period: int) -> tuple[float, np.ndarray, int]:
    if values.size <= period + 4:
        return _ar_fit_forecast(values, 1)
    differenced = values[period:] - values[:-period]
    delta_forecast, residuals, _ = _ar_fit_forecast(differenced, 1)
    return float(values[-period] + delta_forecast), residuals, 4


def _seasonal_mean_forecast(values: np.ndarray, period: int) -> tuple[float, np.ndarray, int]:
    if values.size <= period:
        mean = float(np.mean(values))
        return mean, values - mean, 2
    trend_window = min(values.size, period)
    trend = float(np.mean(values[-trend_window:]))
    seasonal = float(values[-period] - np.mean(values[-period:]))
    fitted = np.full_like(values, trend)
    return trend + seasonal, values - fitted, 4


def _variance_forecast(model_id: str, residuals: np.ndarray) -> float:
    clean = residuals[np.isfinite(residuals)]
    if clean.size < 2:
        return EPS
    if model_id in {"ewma_vol", "student_t_garch"}:
        return _ewma_variance(clean, span=min(30, max(4, clean.size // 3)))
    if model_id in {"arch_1", "garch_1_1", "ar_1_garch", "arma_1_1_garch", "arima_1_1_0_garch"}:
        return _garch_variance(clean)
    if model_id == "realized_vol":
        window = min(30, max(4, clean.size // 3))
        return float(np.var(clean[-window:], ddof=1))
    return float(np.var(clean, ddof=1))


def _ewma_variance(values: np.ndarray, span: int) -> float:
    variances = np.asarray(backend.ewma_variance(values, span), dtype=float)
    finite = variances[np.isfinite(variances)]
    if finite.size:
        return max(float(finite[-1]), EPS)
    return max(float(np.nanvar(values, ddof=1)), EPS)


def _garch_variance(values: np.ndarray) -> float:
    unconditional = max(float(np.var(values, ddof=1)), EPS)
    omega = 0.05 * unconditional
    alpha = 0.08
    beta = 0.90
    variance = unconditional
    for value in values:
        variance = omega + alpha * float(value * value) + beta * variance
    return max(variance, EPS)


def _log_likelihood(residuals: np.ndarray, variance: float, model_id: str) -> float:
    clean = residuals[np.isfinite(residuals)]
    if clean.size == 0:
        return np.nan
    variance = max(float(variance), EPS)
    if model_id == "arch_1":
        return backend.arch_likelihood(clean, 0.05 * variance, 0.08)
    if model_id in {"garch_1_1", "ar_1_garch", "arma_1_1_garch", "arima_1_1_0_garch"}:
        return backend.garch_likelihood(clean, 0.05 * variance, 0.08, 0.90)
    if model_id == "student_t_garch":
        return backend.student_t_garch_likelihood(clean, 0.05 * variance, 0.08, 0.90, 8.0)
    return float(-0.5 * np.sum(np.log(2.0 * math.pi * variance) + clean * clean / variance))


def _parameter_count(model_id: str, base: int) -> int:
    extra = {
        "realized_vol": 1,
        "ewma_vol": 2,
        "arch_1": 3,
        "garch_1_1": 4,
        "student_t_garch": 5,
        "ar_1_garch": 5,
        "arma_1_1_garch": 6,
        "arima_1_1_0_garch": 6,
    }.get(model_id, base)
    return int(max(base, extra))


def _aic(log_likelihood: float, k: int) -> float:
    return float(-2.0 * log_likelihood + 2.0 * k) if np.isfinite(log_likelihood) else np.nan


def _bic(log_likelihood: float, k: int, n: int) -> float:
    return float(-2.0 * log_likelihood + math.log(max(n, 2)) * k) if np.isfinite(log_likelihood) else np.nan


def _hqic(log_likelihood: float, k: int, n: int) -> float:
    return (
        float(-2.0 * log_likelihood + 2.0 * k * math.log(max(math.log(max(n, 3)), 1.0)))
        if np.isfinite(log_likelihood)
        else np.nan
    )


def _target_list(target_view: str) -> list[str]:
    target = str(target_view).strip().lower()
    if target == "log_price":
        return ["log_price"]
    if target == "log_return":
        return ["log_return"]
    return ["log_return", "log_price"]


def _interleave_by_family(specs: list[CandidateSpec]) -> list[CandidateSpec]:
    family_order = ["mean", "trend", "volatility", "combined"]
    grouped: dict[str, list[CandidateSpec]] = {family: [] for family in family_order}
    extras: dict[str, list[CandidateSpec]] = {}
    for spec in specs:
        if spec.family in grouped:
            grouped[spec.family].append(spec)
        else:
            extras.setdefault(spec.family, []).append(spec)
    ordered: list[CandidateSpec] = []
    while any(grouped[family] for family in family_order):
        for family in family_order:
            if grouped[family]:
                ordered.append(grouped[family].pop(0))
    for family in sorted(extras):
        ordered.extend(extras[family])
    return ordered


def _selected_target_series(series: Mapping[str, pd.Series], target_view: str) -> dict[str, pd.Series]:
    wanted = _target_list(target_view)
    selected: dict[str, pd.Series] = {}
    for target in wanted:
        if target in series:
            clean = _clean_series(series[target])
            if not clean.empty:
                selected[target] = clean
    return selected


def _budget_origins(origins: list[int], max_origins: int) -> list[int]:
    if len(origins) <= max_origins:
        return origins
    positions = np.linspace(0, len(origins) - 1, num=max_origins, dtype=int)
    return [origins[int(pos)] for pos in sorted(set(positions))]


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return clean.sort_index()


def _empty_result_frame(specs: list[CandidateSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "target": spec.target,
                "candidate": f"{spec.model_id}:{spec.target}",
                "fit_status": "not_enough_data",
            }
            for spec in specs
        ]
    )


def _first_warning(messages: list[str]) -> str:
    for message in messages:
        if message and message != "ok":
            return message
    return ""


def _current_state(selected: dict[str, object], stage: pd.DataFrame, values: pd.Series) -> dict[str, object]:
    if not selected:
        return {}
    latest_time = values.index[-1] if not values.empty else None
    return {
        "best_current_model": selected.get("model_id"),
        "best_current_family": selected.get("family"),
        "model_uncertainty": _model_uncertainty(stage),
        "latest_closed_candle": latest_time,
        "convergence_status": selected.get("fit_status"),
        "average_fit_time_ms": selected.get("fit_time_ms"),
    }


def _model_uncertainty(stage: pd.DataFrame) -> str:
    if stage.empty or "composite_rank_score" not in stage:
        return "no-data"
    ordered = stage.sort_values("composite_rank_score", kind="mergesort")
    if len(ordered) < 2:
        return "low"
    gap = float(ordered["composite_rank_score"].iloc[1] - ordered["composite_rank_score"].iloc[0])
    if gap <= 1:
        return "high"
    if gap <= 3:
        return "moderate"
    return "low"


def _job_status(
    started: float,
    status: str,
    candidate_count: int,
    stage1_count: int,
    stage2_count: int,
) -> dict[str, object]:
    return {
        "status": status,
        "queue_size": 0,
        "cache_hit_rate": 0.0,
        "candidate_count": int(candidate_count),
        "stage1_fits": int(stage1_count),
        "stage2_fits": int(stage2_count),
        "fit_time_ms": (time.perf_counter() - started) * 1000.0,
        "updated_at": pd.Timestamp.now(tz="UTC"),
    }


def _emit_progress(
    callback: ProgressCallback | None,
    status: dict[str, object],
    **updates: object,
) -> None:
    if callback is None:
        return
    payload = {**status, **updates}
    try:
        callback(payload)
    except Exception:
        return
