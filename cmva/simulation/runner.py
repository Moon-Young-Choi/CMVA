"""Simulation data fetching, validation, and execution."""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import pandas as pd

from cmva.config import CMVAConfig
from cmva.data.candle import candles_to_frame, closed_only, normalize_candle_frame
from cmva.data.rest_client import BinanceRestClient
from cmva.data.storage import CandleStorage
from cmva.engine.interval import interval_to_timedelta
from cmva.models.lab import CandidateSpec, default_candidate_specs
from cmva.models.lab import _fit_candidate  # Reuses the existing model-lab estimators for run-scoped evaluation.
from cmva.native.backend import backend
from cmva.simulation.repository import SimulationRepository
from cmva.simulation.schemas import STATUS_LABELS_KO, SimulationStepResult
from cmva.simulation.scoring import score_origin
from cmva.simulation.spec import SimulationSpec


ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]
EPS = 1e-12


async def fetch_history_for_range(
    rest_client: BinanceRestClient,
    symbols: list[str],
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int = 1000,
) -> pd.DataFrame:
    fetched = []
    for symbol in symbols:
        fetched.extend(
            await rest_client.fetch_historical_range(
                symbol=symbol,
                interval=interval,
                start_time=start,
                end_time=end,
                limit=limit,
            )
        )
    return candles_to_frame(fetched)


async def fetch_history_for_simulation(
    spec: SimulationSpec,
    rest_client: BinanceRestClient,
    limit: int = 1000,
) -> pd.DataFrame:
    return await fetch_history_for_range(
        rest_client,
        spec.symbols,
        spec.interval,
        spec.data_start,
        spec.data_end,
        limit=limit,
    )


class SimulationRunner:
    def __init__(
        self,
        repository: SimulationRepository,
        rest_client: BinanceRestClient | None = None,
        storage: CandleStorage | None = None,
        progress_callback: ProgressCallback | None = None,
        bootstrap_limit: int = 1000,
    ) -> None:
        self.repository = repository
        self.rest_client = rest_client
        self.storage = storage
        self.progress_callback = progress_callback
        self.bootstrap_limit = bootstrap_limit

    async def run(self, spec: SimulationSpec) -> None:
        if not spec.run_id:
            raise ValueError("run_id가 없는 SimulationSpec은 실행할 수 없습니다.")
        run_id = spec.run_id
        warnings: list[str] = []
        try:
            await self._set_progress(
                run_id,
                "preparing_data",
                current_step="범위 기반 시장 데이터를 준비하는 중",
                progress_pct=0.02,
            )
            candles = await self._fetch_or_load_history(spec, warnings)
            await self._set_progress(
                run_id,
                "validating_data",
                current_step="시뮬레이션 입력 데이터 검증 중",
                progress_pct=0.06,
            )
            validation = validate_simulation_data(candles, spec)
            if warnings:
                validation.setdefault("warnings", []).extend(warnings)
            self.repository.save_data_validation(run_id, validation)
            if not validation.get("is_valid"):
                error = "시뮬레이션을 실행할 수 있는 최소 데이터 조건을 만족하지 못했습니다."
                self.repository.save_warnings(run_id, validation.get("warnings", []) + validation.get("errors", []))
                self.repository.mark_failed(run_id, error)
                await self._emit(run_id)
                return
            total_fits = _estimate_total_fits(candles, spec)
            await self._set_progress(
                run_id,
                "running",
                current_step="모델 적합과 점수 산출 실행 중",
                progress_pct=0.08,
                total_fits=total_fits,
            )
            scores: list[dict[str, Any]] = []
            all_step_records: list[dict[str, Any]] = []
            metric_series: dict[str, Any] = {}
            prior_forecasts: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            completed_fits = 0
            specs = default_candidate_specs(
                groups=spec.candidate_model_groups,
                target_view=spec.target_view,
                limit=spec.candidate_model_count,
            )
            closed = _closed_range(candles, spec)
            for symbol in spec.symbols:
                symbol_candles = closed.loc[closed["symbol"] == symbol].sort_values("open_time")
                missing_rate = float(validation.get("symbols", {}).get(symbol, {}).get("missing_rate", 0.0))
                target_series = _target_series(symbol_candles, spec.target_view)
                for target_name, series in target_series.items():
                    target_specs = [candidate for candidate in specs if candidate.target == target_name]
                    origins = generate_simulation_origins(len(series), spec.t_bars, spec.forecast_horizon_bars, spec.dt_bars)
                    for origin_position, origin_index in enumerate(origins, start=1):
                        origin_time = pd.Timestamp(series.index[origin_index])
                        origin_records: list[dict[str, Any]] = []
                        for candidate in target_specs:
                            completed_fits += 1
                            await self._set_progress(
                                run_id,
                                "running",
                                current_step="모델 적합과 점수 산출 실행 중",
                                current_origin_time=origin_time.isoformat(),
                                active_symbol=symbol,
                                active_model=f"{candidate.model_id}:{candidate.target}",
                                completed_fits=completed_fits,
                                total_fits=total_fits,
                                progress_pct=0.08 + 0.86 * (completed_fits / max(1, total_fits)),
                            )
                            origin_records.append(
                                _evaluate_candidate_at_origin(
                                    run_id=run_id,
                                    symbol=symbol,
                                    series=series,
                                    spec=spec,
                                    candidate=candidate,
                                    origin_index=origin_index,
                                )
                            )
                        ranked = _rank_origin_records(origin_records)
                        all_step_records.extend(ranked)
                        self.repository.append_step_results(run_id, ranked)
                        score = score_origin(
                            run_id=run_id,
                            symbol=symbol,
                            origin_time=origin_time.isoformat(),
                            records=ranked,
                            prior_forecasts=prior_forecasts[(symbol, target_name)],
                            scoring_method=spec.scoring_method,
                            missing_rate=missing_rate,
                        ).to_dict()
                        scores.append(score)
                        self.repository.append_score_results(run_id, [score])
                        for record in ranked:
                            if record.get("converged") and math.isfinite(float(record.get("forecast_volatility") or float("nan"))):
                                prior_forecasts[(symbol, target_name)][str(record["candidate"])].append(
                                    float(record["forecast_volatility"])
                                )
                        metric_series = build_metric_series(scores, all_step_records)
                        self.repository.save_metric_series(run_id, metric_series)
                        await self._set_progress(
                            run_id,
                            "running",
                            current_step=f"{symbol} {target_name} {origin_position}/{len(origins)} 시점 완료",
                            latest_metrics=_latest_metric_summary(ranked),
                            latest_scores=score,
                            completed_fits=completed_fits,
                            total_fits=total_fits,
                            progress_pct=0.08 + 0.86 * (completed_fits / max(1, total_fits)),
                        )
            conclusion = build_conclusion(spec, validation, scores, all_step_records)
            self.repository.save_metric_series(run_id, metric_series)
            self.repository.save_conclusion(run_id, conclusion)
            self.repository.save_warnings(run_id, sorted(set(warnings + validation.get("warnings", []))))
            self.repository.mark_completed(run_id)
            await self._emit(run_id)
        except Exception as exc:
            self.repository.mark_failed(run_id, f"시뮬레이션 실행 중 오류가 발생했습니다: {exc}")
            await self._emit(run_id)

    async def _fetch_or_load_history(self, spec: SimulationSpec, warnings: list[str]) -> pd.DataFrame:
        remote = pd.DataFrame()
        if self.rest_client is not None:
            try:
                remote = await fetch_history_for_simulation(spec, self.rest_client, limit=self.bootstrap_limit)
            except Exception as exc:
                warnings.append(f"REST 범위 조회가 실패하여 로컬 캐시를 확인했습니다: {exc}")
        if not remote.empty:
            return remote
        if self.storage is None:
            return remote
        local = self.storage.load_many(spec.symbols)
        if local.empty:
            return local
        warnings.append("원격 조회 결과가 없어 로컬 닫힌 캔들 캐시로 검증했습니다.")
        return _closed_range(local, spec)

    async def _set_progress(self, run_id: str, status: str, **updates: Any) -> None:
        payload = {
            "updated_at": pd.Timestamp.now(tz="UTC"),
            **updates,
        }
        self.repository.update_status(run_id, status, **payload)
        await self._emit(run_id)

    async def _emit(self, run_id: str) -> None:
        if self.progress_callback is None:
            return
        snapshot = self.repository.load_partial_results(run_id)
        snapshot.update(self.repository.load_run(run_id))
        result = self.progress_callback(run_id, snapshot)
        if asyncio.iscoroutine(result):
            await result


def validate_simulation_data(candles: pd.DataFrame, spec: SimulationSpec) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "is_valid": True,
        "errors": errors,
        "warnings": warnings,
        "symbols": {},
        "required": {
            "t_bars": spec.t_bars,
            "dt_bars": spec.dt_bars,
            "forecast_horizon_bars": spec.forecast_horizon_bars,
            "calibration_bars": spec.calibration_bars,
        },
    }
    if candles.empty:
        errors.append("데이터 기간에 해당하는 캔들이 없습니다.")
        report["is_valid"] = False
        return report
    data = normalize_candle_frame(candles)
    bad_interval = data.loc[data["interval"] != spec.interval]
    if not bad_interval.empty:
        errors.append("선택한 캔들 간격과 다른 데이터가 포함되어 있습니다.")
    unclosed = data.loc[~data["is_closed"]]
    if not unclosed.empty:
        errors.append("열려 있는 미완성 캔들이 포함되어 있습니다.")
    data = _closed_range(data, spec)
    if data.empty:
        errors.append("선택한 기간 안에 닫힌 캔들이 없습니다.")
        report["is_valid"] = False
        return report
    duplicates = data.duplicated(["symbol", "interval", "open_time"], keep=False)
    if duplicates.any():
        errors.append("중복된 캔들 타임스탬프가 있습니다.")
    bad_ohlc = data.loc[
        (data["high"] < data[["open", "close", "low"]].max(axis=1))
        | (data["low"] > data[["open", "close", "high"]].min(axis=1))
        | (data[["open", "high", "low", "close"]] <= 0).any(axis=1)
    ]
    if not bad_ohlc.empty:
        errors.append("유효하지 않은 OHLC 캔들이 있습니다.")
    expected_delta = interval_to_timedelta(spec.interval)
    for symbol in spec.symbols:
        group = data.loc[data["symbol"] == symbol].sort_values("open_time")
        unique_times = pd.DatetimeIndex(group["open_time"].drop_duplicates())
        expected_start = _ceil_to_interval(spec.data_start, expected_delta)
        expected_end = _floor_to_interval(spec.data_end, expected_delta)
        expected_times = (
            pd.date_range(expected_start, expected_end, freq=expected_delta, tz="UTC")
            if expected_start <= expected_end
            else pd.DatetimeIndex([], tz="UTC")
        )
        missing = expected_times.difference(unique_times)
        gap_count = int(group["open_time"].diff().gt(expected_delta).sum()) if not group.empty else 0
        expected_rows = len(expected_times)
        missing_count = int(max(len(missing), 0))
        missing_rate = float(missing_count / expected_rows) if expected_rows else 0.0
        origins = generate_simulation_origins(max(0, len(group) - 1), spec.t_bars, spec.forecast_horizon_bars, spec.dt_bars)
        row = {
            "rows": int(len(group)),
            "first_open_time": group["open_time"].iloc[0].isoformat() if not group.empty else None,
            "latest_open_time": group["open_time"].iloc[-1].isoformat() if not group.empty else None,
            "expected_rows": expected_rows,
            "missing_candle_count": missing_count,
            "missing_rate": missing_rate,
            "coverage_pct": round(100.0 * (1.0 - missing_rate), 4),
            "gap_count": gap_count,
            "origin_count": len(origins),
        }
        report["symbols"][symbol] = row
        if group.empty:
            errors.append(f"{symbol}: 데이터가 없습니다.")
        elif len(group) < spec.t_bars + spec.forecast_horizon_bars + 1:
            errors.append(f"{symbol}: T, horizon을 충족할 만큼 캔들이 충분하지 않습니다.")
        elif not origins:
            errors.append(f"{symbol}: 시뮬레이션 origin을 만들 수 없습니다.")
        if missing_count:
            warnings.append(f"{symbol}: 누락 캔들 {missing_count}개, 누락률 {missing_rate:.2%}입니다.")
        if gap_count:
            warnings.append(f"{symbol}: 캔들 간격에 공백이 {gap_count}회 감지되었습니다.")
        if spec.data_start != expected_start or spec.data_end != expected_end:
            warnings.append(
                f"{symbol}: 입력 시간이 {spec.interval} 캔들 경계와 달라 기대 범위를 "
                f"{expected_start.isoformat()} - {expected_end.isoformat()} 기준으로 검증했습니다."
            )
    report["is_valid"] = not errors
    return report


def generate_simulation_origins(
    n_observations: int,
    training_window_bars: int,
    horizon_bars: int,
    stride_bars: int,
) -> list[int]:
    n = int(n_observations)
    training = max(1, int(training_window_bars))
    horizon = max(1, int(horizon_bars))
    stride = max(1, int(stride_bars))
    if n <= training + horizon:
        return []
    return list(range(training - 1, n - horizon, stride))


def _ceil_to_interval(timestamp: pd.Timestamp, delta: pd.Timedelta) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    elapsed = ts - epoch
    remainder = elapsed.value % delta.value
    if remainder == 0:
        return ts
    return ts + pd.Timedelta(delta.value - remainder, unit="ns")


def _floor_to_interval(timestamp: pd.Timestamp, delta: pd.Timedelta) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    elapsed = ts - epoch
    return epoch + pd.Timedelta((elapsed.value // delta.value) * delta.value, unit="ns")


def build_metric_series(scores: list[dict[str, Any]], latest_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    series: dict[str, Any] = {
        "volatility_score": [],
        "expectation_score": [],
        "trend_score": [],
        "seasonality_score": [],
        "confidence": [],
        "selected_model": [],
        "forecast_vs_realized_volatility": [],
        "aic_by_model": defaultdict(list),
        "bic_by_model": defaultdict(list),
        "hqic_by_model": defaultdict(list),
        "qlike_by_model": defaultdict(list),
        "rmse_mae_by_model": defaultdict(list),
    }
    for score in scores:
        time_key = score.get("origin_time")
        symbol = score.get("symbol")
        series["volatility_score"].append({"time": time_key, "symbol": symbol, "value": score.get("volatility_score_0_100")})
        series["expectation_score"].append({"time": time_key, "symbol": symbol, "value": score.get("expectation_score_0_100")})
        series["trend_score"].append({"time": time_key, "symbol": symbol, "value": score.get("trend_score_minus100_100")})
        series["seasonality_score"].append({"time": time_key, "symbol": symbol, "value": score.get("seasonality_score_0_100")})
        series["confidence"].append({"time": time_key, "symbol": symbol, "value": score.get("volatility_confidence_0_1")})
        series["selected_model"].append({"time": time_key, "symbol": symbol, "value": score.get("selected_model_id")})
        series["forecast_vs_realized_volatility"].append(
            {
                "time": time_key,
                "symbol": symbol,
                "forecast": score.get("forecast_volatility"),
                "realized": score.get("realized_volatility"),
            }
        )
    for record in latest_records or []:
        candidate = str(record.get("candidate"))
        time_key = record.get("origin_time")
        for metric, key in (
            ("aic", "aic_by_model"),
            ("bic", "bic_by_model"),
            ("hqic", "hqic_by_model"),
            ("qlike", "qlike_by_model"),
        ):
            series[key][candidate].append({"time": time_key, "value": record.get(metric)})
        series["rmse_mae_by_model"][candidate].append(
            {"time": time_key, "rmse": record.get("rmse_loss"), "mae": record.get("mae_loss")}
        )
    return _plain_series(series)


def build_conclusion(
    spec: SimulationSpec,
    validation: dict[str, Any],
    scores: list[dict[str, Any]],
    step_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for score in scores:
        latest_by_symbol[str(score.get("symbol"))] = score
    headline_parts = []
    for symbol, score in latest_by_symbol.items():
        headline_parts.append(
            f"{symbol}: 변동성 {score.get('volatility_level_label')}, 기대값 {score.get('expectation_level_label')}, "
            f"추세 {score.get('trend_level_label')}, 계절성 {score.get('seasonality_level_label')}"
        )
    return {
        "headline_summary": " / ".join(headline_parts) if headline_parts else "완료된 점수 행이 없습니다.",
        "latest_scores": list(latest_by_symbol.values()),
        "input_summary": spec.to_dict(),
        "data_quality_summary": validation,
        "model_evaluation": _model_evaluation(step_records or []),
        "score_count": len(scores),
    }


def _model_evaluation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    frame = pd.DataFrame(records)
    rows = []
    for candidate, group in frame.groupby("candidate", dropna=False):
        ranks = pd.to_numeric(group.get("rank"), errors="coerce").dropna()
        rank_stability = 1.0
        if len(ranks) > 1:
            rank_stability = float(1.0 / (1.0 + ranks.std(ddof=0)))
        rows.append(
            {
                "candidate": str(candidate),
                "origin_count": int(len(group)),
                "convergence_rate": float(group["converged"].astype(bool).mean()) if "converged" in group else 0.0,
                "avg_aic": _finite_or_none(pd.to_numeric(group.get("aic"), errors="coerce").mean()),
                "avg_bic": _finite_or_none(pd.to_numeric(group.get("bic"), errors="coerce").mean()),
                "avg_hqic": _finite_or_none(pd.to_numeric(group.get("hqic"), errors="coerce").mean()),
                "avg_qlike": _finite_or_none(pd.to_numeric(group.get("qlike"), errors="coerce").mean()),
                "rmse": _finite_or_none(pd.to_numeric(group.get("rmse_loss"), errors="coerce").mean()),
                "mae": _finite_or_none(pd.to_numeric(group.get("mae_loss"), errors="coerce").mean()),
                "rank_stability": _finite_or_none(rank_stability),
                "composite_rank_score": _finite_or_none(
                    pd.to_numeric(group.get("composite_rank_score"), errors="coerce").mean()
                ),
            }
        )
    return sorted(rows, key=lambda row: row.get("composite_rank_score") if row.get("composite_rank_score") is not None else float("inf"))


def _evaluate_candidate_at_origin(
    run_id: str,
    symbol: str,
    series: pd.Series,
    spec: SimulationSpec,
    candidate: CandidateSpec,
    origin_index: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    values = series.to_numpy(dtype=float)
    train_start = max(0, origin_index - spec.t_bars + 1)
    train = values[train_start : origin_index + 1]
    realization_index = origin_index + spec.forecast_horizon_bars
    realized = float(values[realization_index])
    if candidate.target == "log_price":
        realized_return = float(values[realization_index] - values[origin_index])
        realized_mean = realized
    else:
        realized_return = realized
        realized_mean = realized
    fit = _fit_candidate(candidate, train)
    forecast_variance = max(float(fit.variance_forecast), EPS)
    realized_variance = max(float(realized_return * realized_return), EPS)
    qlike = float(backend.qlike(np.asarray([realized_variance]), np.asarray([forecast_variance]))[0])
    rmse_loss = abs(realized_variance - forecast_variance)
    mae_loss = abs(realized_variance - forecast_variance)
    nobs = int(np.isfinite(train).sum())
    residual_diagnostics = _residual_diagnostics(train)
    warning = _candidate_warning(candidate, fit.message)
    row = SimulationStepResult(
        run_id=run_id,
        symbol=symbol,
        origin_index=origin_index,
        origin_time=pd.Timestamp(series.index[origin_index]).isoformat(),
        realization_time=pd.Timestamp(series.index[realization_index]).isoformat(),
        train_start_time=pd.Timestamp(series.index[train_start]).isoformat(),
        train_end_time=pd.Timestamp(series.index[origin_index]).isoformat(),
        evaluation_start_time=pd.Timestamp(series.index[origin_index]).isoformat(),
        evaluation_end_time=pd.Timestamp(series.index[realization_index]).isoformat(),
        model_id=candidate.model_id,
        model_family=candidate.family,
        target=candidate.target,
        candidate=f"{candidate.model_id}:{candidate.target}",
        estimator=_estimator_for(candidate),
        params={"parameter_count": int(fit.parameter_count), "fit_message": fit.message},
        converged=bool(fit.success),
        fit_status="ok" if fit.success else "failed",
        fit_message="정상 적합" if fit.success else f"모델 적합 실패: {fit.message}",
        log_likelihood=_finite_or_none(fit.log_likelihood),
        nobs=nobs,
        parameter_count=int(fit.parameter_count),
        aic=_finite_or_none(fit.aic),
        bic=_finite_or_none(fit.bic),
        hqic=_finite_or_none(fit.hqic),
        qlike=_finite_or_none(qlike),
        rmse_loss=_finite_or_none(rmse_loss),
        mae_loss=_finite_or_none(mae_loss),
        forecast_mean=_finite_or_none(fit.mean_forecast),
        realized_mean=_finite_or_none(realized_mean),
        realized_return=_finite_or_none(realized_return),
        forecast_variance=_finite_or_none(forecast_variance),
        realized_variance=_finite_or_none(realized_variance),
        forecast_volatility=_finite_or_none(math.sqrt(forecast_variance)),
        realized_volatility=_finite_or_none(math.sqrt(realized_variance)),
        residual_diagnostics=residual_diagnostics,
        rank=None,
        composite_rank_score=None,
        fit_time_ms=(time.perf_counter() - start) * 1000.0,
        warnings=[warning] if warning else [],
    )
    return row.to_dict()


def _rank_origin_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records
    frame = pd.DataFrame(records)
    for metric, ascending in (("aic", True), ("bic", True), ("hqic", True), ("qlike", True), ("rmse_loss", True), ("mae_loss", True)):
        frame[f"{metric}_rank"] = pd.to_numeric(frame[metric], errors="coerce").rank(method="dense", ascending=ascending)
    frame["rank"] = pd.to_numeric(frame["qlike"], errors="coerce").rank(method="dense", ascending=True).fillna(len(frame)).astype(int)
    frame["composite_rank_score"] = (
        frame["aic_rank"].fillna(len(frame))
        + frame["bic_rank"].fillna(len(frame))
        + frame["hqic_rank"].fillna(len(frame))
        + 2.0 * frame["qlike_rank"].fillna(len(frame))
        + frame["rmse_loss_rank"].fillna(len(frame))
        + frame["mae_loss_rank"].fillna(len(frame))
        + np.where(frame["converged"], 0.0, len(frame))
    )
    frame = frame.sort_values(["composite_rank_score", "qlike", "aic"], kind="mergesort")
    return frame.drop(columns=[column for column in frame.columns if column.endswith("_rank")], errors="ignore").to_dict(orient="records")


def _target_series(candles: pd.DataFrame, target_view: str) -> dict[str, pd.Series]:
    close = pd.Series(
        pd.to_numeric(candles["close"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(candles["open_time"], utc=True),
        name="close",
    )
    log_price = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan).dropna().rename("log_price")
    log_return = log_price.diff().dropna().rename("log_return")
    if target_view == "log_price":
        return {"log_price": log_price}
    if target_view == "log_return":
        return {"log_return": log_return}
    return {"log_return": log_return, "log_price": log_price}


def _closed_range(candles: pd.DataFrame, spec: SimulationSpec) -> pd.DataFrame:
    if candles.empty:
        return candles
    data = closed_only(candles)
    return data.loc[
        (data["interval"] == spec.interval)
        & (data["open_time"] >= spec.data_start)
        & (data["open_time"] <= spec.data_end)
        & (data["symbol"].isin(spec.symbols))
    ].copy()


def _estimate_total_fits(candles: pd.DataFrame, spec: SimulationSpec) -> int:
    data = _closed_range(candles, spec)
    specs = default_candidate_specs(
        groups=spec.candidate_model_groups,
        target_view=spec.target_view,
        limit=spec.candidate_model_count,
    )
    total = 0
    for symbol in spec.symbols:
        targets = _target_series(data.loc[data["symbol"] == symbol].sort_values("open_time"), spec.target_view)
        for target_name, series in targets.items():
            total += len([candidate for candidate in specs if candidate.target == target_name]) * len(
                generate_simulation_origins(len(series), spec.t_bars, spec.forecast_horizon_bars, spec.dt_bars)
            )
    return max(1, total)


def _latest_metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    best = min(records, key=lambda record: record.get("composite_rank_score") or float("inf"))
    return {
        "선택 후보": best.get("candidate"),
        "AIC": best.get("aic"),
        "BIC": best.get("bic"),
        "QLIKE": best.get("qlike"),
        "예측 변동성": best.get("forecast_volatility"),
        "실현 변동성": best.get("realized_volatility"),
    }


def _residual_diagnostics(train: np.ndarray) -> dict[str, Any]:
    clean = train[np.isfinite(train)]
    if clean.size < 4:
        return {"표본수": int(clean.size), "해석": "잔차 진단을 계산할 표본이 부족합니다."}
    centered = clean - float(np.mean(clean))
    variance = float(np.var(centered, ddof=1)) if clean.size > 1 else 0.0
    autocorr = np.nan
    if clean.size > 2 and np.std(centered[:-1]) > 0 and np.std(centered[1:]) > 0:
        autocorr = float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
    return {
        "표본수": int(clean.size),
        "잔차분산": variance,
        "1차 자기상관": _finite_or_none(autocorr),
        "해석": "진단은 적합 시점 이전 학습 구간만 사용했습니다.",
    }


def _candidate_warning(candidate: CandidateSpec, message: str) -> str:
    if message and message != "ok":
        return f"{candidate.model_id}: {message}"
    if candidate.model_id in {"ma_1", "arma_1_1", "arima_1_1_0", "ar_1_garch", "arma_1_1_garch", "arima_1_1_0_garch"}:
        return "고급 모형은 현재 경량 추정/폴백 후보로 평가되었습니다."
    return ""


def _estimator_for(candidate: CandidateSpec) -> str:
    model_id = candidate.model_id
    if "garch" in model_id or model_id.startswith("arch"):
        return "MLE/QMLE 경량 변동성 추정"
    if model_id.startswith(("ar_", "ma_", "arma_", "arima")):
        return "OLS/CSS 경량 평균 추정"
    if "season" in model_id or "stl" in model_id:
        return "계절 더미/평균 경량 추정"
    if model_id == "ewma_vol":
        return "EWMA 재귀 추정"
    if model_id == "realized_vol":
        return "표본 실현 변동성 추정"
    return "폐쇄형 표본 추정"


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _plain_series(series: dict[str, Any]) -> dict[str, Any]:
    plain = {}
    for key, value in series.items():
        if isinstance(value, defaultdict):
            plain[key] = dict(value)
        else:
            plain[key] = value
    return plain


def create_runner_for_app(
    config: CMVAConfig,
    repository: SimulationRepository,
    rest_client: BinanceRestClient,
    storage: CandleStorage,
    progress_callback: ProgressCallback | None = None,
) -> SimulationRunner:
    return SimulationRunner(
        repository=repository,
        rest_client=rest_client,
        storage=storage,
        progress_callback=progress_callback,
        bootstrap_limit=config.bootstrap_limit,
    )
