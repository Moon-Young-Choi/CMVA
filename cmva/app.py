"""Top-level CMVA application orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from cmva.analysis_types import DiagnosticSnapshot
from cmva.config import CMVAConfig, ensure_artifact_dirs, load_config
from cmva.data.candle import Candle
from cmva.data.rest_client import BinanceRestClient
from cmva.data.storage import CandleStorage, save_frame
from cmva.data.validation import ValidationReport, validate_candles
from cmva.data.websocket_client import BinanceWebSocketClient
from cmva.engine.interval import latest_closed_open_time
from cmva.features import FeatureBundle, compute_feature_bundle
from cmva.forecast.volatility_forecaster import ForecastSnapshot, VolatilityForecaster
from cmva.logging_config import configure_logging
from cmva.models.garch import historical_garch_forecast
from cmva.models.diagnostics import build_method_steps, run_statistical_diagnostics
from cmva.models.lab import ModelLabResult, run_model_lab
from cmva.models.selection import ModelSelectionResult, select_volatility_model
from cmva.native.backend import backend_status
from cmva.regime.classifier import classify_regime_series
from cmva.regime.shock import compute_shock_series
from cmva.state import AppState
from cmva.time_ranges import parse_time_range, slice_by_time_range
from cmva.validation import ModelValidationResult, run_walk_forward_validation


@dataclass
class AnalysisSnapshot:
    candles: pd.DataFrame
    features: FeatureBundle
    forecast: ForecastSnapshot
    historical_forecast: pd.Series
    regimes: pd.Series
    shocks: pd.DataFrame
    model_validation: ModelValidationResult | None
    model_lab: ModelLabResult | None
    validation: ValidationReport
    diagnostics: DiagnosticSnapshot


class CMVAApplication:
    def __init__(self, config: CMVAConfig | None = None) -> None:
        self.config = config or load_config()
        ensure_artifact_dirs(self.config)
        self.state = AppState()
        self.storage = CandleStorage(self.config.cleaned_dir)
        self.rest_client = BinanceRestClient(self.config.rest_base_url)
        self.forecaster = VolatilityForecaster(
            refit_frequency=self.config.garch_refit_frequency,
            min_observations=100,
        )
        self.snapshot: AnalysisSnapshot | None = None
        self.range_model_validation: ModelValidationResult | None = None
        self.range_diagnostics: DiagnosticSnapshot | None = None
        self.model_selection: ModelSelectionResult | None = None
        self.last_model_selection_time: pd.Timestamp | None = None
        self.model_lab_job_status: dict[str, object] = {
            "status": "idle",
            "queue_size": 0,
            "cache_hit_rate": 0.0,
            "candidate_count": 0,
            "stage1_fits": 0,
            "stage2_fits": 0,
            "progress_pct": 0.0,
            "updated_at": pd.Timestamp.now(tz="UTC"),
        }

    async def bootstrap(self, fetch_remote: bool = True) -> None:
        self.state.mode = "BOOTSTRAP"
        self._set_bootstrap_progress("loading_cache", current_symbol=None, completed_symbols=0)
        self.state.log("Loading local candle cache")
        candles = self._active_interval_frame(self.storage.load_many(self.config.symbols))
        if fetch_remote:
            try:
                candles = await self.fetch_missing_history(candles)
            except Exception as exc:
                self.state.mode = "DEGRADED"
                self._set_bootstrap_progress("degraded", error=str(exc))
                self.state.log(f"Historical fetch degraded: {exc}")
        if candles.empty:
            self.state.mode = "DEGRADED"
            self.state.log("No candles available yet")
            self.state.data_status = {"rows": 0, "validation": "no data"}
            self._set_bootstrap_progress("degraded", error="no candle rows available")
            return
        self._set_bootstrap_progress("computing", current_symbol=None)
        self.recompute(candles, force_refit=True)
        if self.state.mode != "DEGRADED":
            self.state.mode = "LIVE"
            self._set_bootstrap_progress("ready", current_symbol=None, completed_symbols=len(self.config.symbols))

    async def fetch_missing_history(self, existing: pd.DataFrame) -> pd.DataFrame:
        now = latest_closed_open_time(pd.Timestamp(datetime.now(timezone.utc)), self.config.interval)
        analysis_range = parse_time_range(self.config.analysis_period)
        desired_start = (
            now - pd.Timedelta(hours=analysis_range.hours)
            if analysis_range.hours is not None
            else now - pd.Timedelta(days=self.config.historical_days)
        )
        fetched: list[Candle] = []
        fetched_count = 0
        total_symbols = len(self.config.symbols)
        self._set_bootstrap_progress(
            "fetching",
            completed_symbols=0,
            total_symbols=total_symbols,
            fetched_candles=fetched_count,
            target_start=desired_start,
            target_end=now,
        )
        for position, symbol in enumerate(self.config.symbols, start=1):
            self._set_bootstrap_progress(
                "fetching",
                current_symbol=symbol,
                completed_symbols=position - 1,
                total_symbols=total_symbols,
                fetched_candles=fetched_count,
                target_start=desired_start,
                target_end=now,
            )
            symbol_existing = existing.loc[
                (existing["symbol"] == symbol) & (existing["interval"] == self.config.interval)
            ] if not existing.empty else pd.DataFrame()
            if symbol_existing.empty:
                start = desired_start
            else:
                latest = pd.to_datetime(symbol_existing["open_time"], utc=True).max()
                start = max(latest + self.config.interval_delta, desired_start)
            if start <= now:
                self.state.log(f"Fetching {symbol} klines from {start}")
                symbol_fetched = await self.rest_client.fetch_historical_range(
                    symbol=symbol,
                    interval=self.config.interval,
                    start_time=start,
                    end_time=now,
                    limit=self.config.bootstrap_limit,
                )
                fetched.extend(symbol_fetched)
                fetched_count += len(symbol_fetched)
            self._set_bootstrap_progress(
                "fetching",
                current_symbol=symbol,
                completed_symbols=position,
                total_symbols=total_symbols,
                fetched_candles=fetched_count,
                target_start=desired_start,
                target_end=now,
            )
        if fetched:
            self._set_bootstrap_progress(
                "storing",
                current_symbol=None,
                completed_symbols=total_symbols,
                total_symbols=total_symbols,
                fetched_candles=fetched_count,
                target_start=desired_start,
                target_end=now,
            )
            self.storage.upsert_closed(fetched)
        return self._active_interval_frame(self.storage.load_many(self.config.symbols))

    def _set_bootstrap_progress(self, phase: str, **updates: object) -> None:
        progress = {
            "phase": phase,
            "interval": self.config.interval,
            "total_symbols": len(self.config.symbols),
            "completed_symbols": 0,
            "current_symbol": None,
            "fetched_candles": 0,
            **self.state.bootstrap_progress,
            **updates,
            "updated_at": pd.Timestamp.now(tz="UTC"),
        }
        progress["phase"] = phase
        self.state.bootstrap_progress = progress

    def recompute(self, candles: pd.DataFrame, force_refit: bool = False) -> AnalysisSnapshot:
        candles = self._active_interval_frame(candles)
        validation = validate_candles(candles, self.config.interval, self.config.symbols)
        features = compute_feature_bundle(
            candles,
            short_window=self.config.rolling_short_window,
            medium_window=self.config.rolling_medium_window,
            long_window=self.config.rolling_long_window,
            trend_window=self.config.window_bar_counts["trend_window"],
        )
        forecast = self.forecaster.forecast(features.returns, force_refit=force_refit)
        self._maybe_run_model_selection(features.features["basket_return"])
        historical_forecast = self._historical_forecast_series(features.features)
        regimes = classify_regime_series(
            features.features,
            historical_forecast,
            min_history=min(self.config.min_threshold_history, max(30, len(features.features) // 4)),
        )
        shocks = compute_shock_series(
            features.returns,
            historical_forecast,
            features.features["avg_pairwise_corr"],
            features.features["dispersion"],
            window=self.config.rolling_short_window,
            severe_threshold=self.config.severe_shock_threshold,
            moderate_threshold=self.config.moderate_shock_threshold,
        )
        model_validation = None
        if len(features.returns.dropna(how="all")) > 2:
            model_validation = run_walk_forward_validation(
                basket_returns=features.features["basket_return"],
                forecast_vol=historical_forecast,
                ewma_vol=features.features["ewma_vol"],
                realized_vol=features.features["market_vol"],
                regimes=regimes,
            )
        self._set_model_lab_progress(
            {
                "status": "queued",
                "queue_size": 1,
                "cache_hit_rate": 0.0,
                "candidate_count": self.config.candidate_model_count,
                "stage1_fits": 0,
                "stage2_fits": 0,
                "progress_pct": 0.0,
            }
        )
        model_lab = run_model_lab(self._model_lab_targets(features), self.config, progress_callback=self._set_model_lab_progress)
        self._set_model_lab_progress(model_lab.job_status)
        garch_params = dict(self.forecaster.basket_model._fit_result.params)
        diagnostics = run_statistical_diagnostics(
            basket_returns=features.features["basket_return"],
            forecast_vol=historical_forecast,
            ewma_vol=features.features["ewma_vol"],
            regimes=regimes,
            shocks=shocks,
            validation_losses=model_validation.losses if model_validation is not None else None,
            garch_params=garch_params,
            generated_at=pd.Timestamp.now(tz="UTC"),
            periods_per_year=self.config.periods_per_year,
        )
        diagnostics.method_steps = build_method_steps(
            features=features.features,
            forecast_vol=historical_forecast,
            regimes=regimes,
            shocks=shocks,
        )
        snapshot = AnalysisSnapshot(
            candles,
            features,
            forecast,
            historical_forecast,
            regimes,
            shocks,
            model_validation,
            model_lab,
            validation,
            diagnostics,
        )
        self.snapshot = snapshot
        self._persist_analysis(snapshot)
        self._update_state(snapshot, historical_forecast)
        return snapshot

    def process_current_candle(self, candle: Candle) -> None:
        self.state.data_status["current_candle"] = candle.to_record()

    def process_closed_candle(self, candle: Candle) -> None:
        if not candle.is_closed:
            self.process_current_candle(candle)
            return
        current = self.state.data_status.get("current_candle")
        if isinstance(current, dict) and current.get("symbol") == candle.symbol and current.get("open_time") == candle.open_time:
            self.state.data_status.pop("current_candle", None)
        if self.state.paused:
            self.state.log(f"Paused; received closed candle for {candle.symbol} at {candle.open_time}")
            return
        candles = self.storage.upsert_closed([candle])
        all_candles = self._active_interval_frame(self.storage.load_many(self.config.symbols))
        snapshot = self.recompute(all_candles, force_refit=False)
        if not candles.empty:
            self.state.log(f"Processed closed candle {candle.symbol} {candle.open_time}")

    async def stream_live(self) -> None:
        client = BinanceWebSocketClient(
            self.config.symbols,
            interval=self.config.interval,
            base_url=self.config.websocket_base_url,
        )
        self.state.websocket_status = "connecting"
        async for candle in client.stream():
            self.state.websocket_status = "connected"
            if candle.is_closed:
                self.process_closed_candle(candle)
            else:
                self.process_current_candle(candle)

    def force_refit(self) -> None:
        candles = self._active_interval_frame(self.storage.load_many(self.config.symbols))
        if not candles.empty:
            self.recompute(candles, force_refit=True)
            self.state.log("Forced GARCH refit")

    def rerun_backtest(self) -> None:
        if self.snapshot is not None:
            self.recompute(self.snapshot.candles, force_refit=False)
            self.state.log("Walk-forward validation recomputed")

    def run_model_selection(self) -> ModelSelectionResult | None:
        if self.snapshot is None:
            self.state.log("Model selection skipped: no snapshot")
            return None
        result = select_volatility_model(self.snapshot.features.features["basket_return"].dropna())
        self.model_selection = result
        self.last_model_selection_time = pd.Timestamp.now(tz="UTC")
        self.state.model_status.update(
            {
                "selected_model": result.selected_model,
                "model_selection_success": result.success,
                "model_selection_aic": result.aic,
                "model_selection_bic": result.bic,
                "model_selection_message": result.message,
                "last_model_selection_time": self.last_model_selection_time,
            }
        )
        self.state.log(f"Model selection completed: {result.selected_model} ({result.message})")
        return result

    def apply_settings(self, updates: dict[str, object], recompute: bool = False) -> None:
        allowed = {
            "symbols",
            "interval",
            "forecast_horizon_bars",
            "forecast_horizon",
            "historical_days",
            "dashboard_time_range",
            "forecast_time_range",
            "backtest_time_range",
            "allowed_time_ranges",
            "volatility_window",
            "correlation_window",
            "pca_window",
            "trend_window",
            "regime_threshold_window",
            "rolling_short_window",
            "rolling_medium_window",
            "rolling_long_window",
            "min_threshold_history",
            "garch_refit_frequency",
            "severe_shock_threshold",
            "moderate_shock_threshold",
            "shock_breadth_threshold",
            "use_cpp",
            "analysis_period",
            "training_window",
            "refit_stride_bars",
            "search_mode",
            "candidate_model_count",
            "candidate_model_groups",
            "target_view",
        }
        current = asdict(self.config)
        for key, value in updates.items():
            if key in allowed:
                current[key] = value
        self.config = CMVAConfig(**current)
        ensure_artifact_dirs(self.config)
        self.storage = CandleStorage(self.config.cleaned_dir)
        self.rest_client = BinanceRestClient(self.config.rest_base_url)
        self.forecaster.refit_frequency = self.config.garch_refit_frequency
        if recompute and self.snapshot is not None:
            self.recompute(self.snapshot.candles, force_refit=True)
            self.state.log("Settings applied and walk-forward validation recomputed")
        elif self.snapshot is not None:
            self._update_range_views(self.snapshot)
            self.state.log("Settings applied from now")
        else:
            self.state.log("Settings applied from now")

    def apply_view_ranges(self, updates: dict[str, object], recompute_backtest: bool = False) -> None:
        current = asdict(self.config)
        for key in ("dashboard_time_range", "forecast_time_range", "backtest_time_range"):
            if key in updates:
                current[key] = str(updates[key])
        self.config = CMVAConfig(**current)
        if self.snapshot is not None:
            self._update_range_views(self.snapshot)
        action = "View ranges applied"
        if recompute_backtest:
            action = "View ranges applied and backtest range recomputed"
        self.state.log(action)

    def cancel_settings(self) -> None:
        self.state.log("Settings change canceled")

    def toggle_pause(self) -> None:
        self.state.paused = not self.state.paused
        self.state.mode = "PAUSED" if self.state.paused else "LIVE"
        self.state.log("Paused live processing" if self.state.paused else "Resumed live processing")

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.state.mode,
            "latest_closed_time": self.state.latest_closed_time,
            "current_regime": self.state.current_regime,
            "current_shock_type": self.state.current_shock_type,
            "forecast_vol": self.state.forecast_vol,
            "forecast_horizon": self.config.forecast_horizon,
            "interval": self.config.interval,
            "window_bar_counts": self.config.window_bar_counts,
            "dashboard_time_range": self.config.dashboard_time_range,
            "forecast_time_range": self.config.forecast_time_range,
            "backtest_time_range": self.config.backtest_time_range,
            "websocket_status": self.state.websocket_status,
            "data_rows": self.state.data_status.get("rows"),
            "validation": self.state.data_status.get("validation"),
            "validation_issues": self.state.data_status.get("validation_issues"),
            **self.state.backtest_summary,
        }

    def _historical_forecast_series(self, features: pd.DataFrame) -> pd.Series:
        forecast = historical_garch_forecast(
            features["basket_return"],
            refit_frequency=self.config.garch_refit_frequency,
            min_observations=100,
        ).reindex(features.index)
        latest_garch = self.forecaster.basket_model.forecast_one_step(features["basket_return"].dropna())
        if not forecast.empty and latest_garch.volatility > 0:
            forecast.iloc[-1] = latest_garch.volatility
        return forecast.ffill().fillna(features["market_vol"]).rename("forecast_vol")

    def _model_lab_targets(self, features: FeatureBundle) -> dict[str, pd.Series]:
        basket_price = features.close.mean(axis=1, skipna=True)
        log_price = pd.Series(
            np.log(basket_price.where(basket_price > 0)).replace([np.inf, -np.inf], np.nan),
            index=basket_price.index,
            name="log_price",
        )
        return {
            "log_return": features.features["basket_return"].rename("log_return"),
            "log_price": log_price,
        }

    def _set_model_lab_progress(self, status: dict[str, object]) -> None:
        payload = {**self.model_lab_job_status, **status}
        if not status.get("updated_at"):
            payload["updated_at"] = pd.Timestamp.now(tz="UTC")
        if payload.get("status") in {"complete", "no_data", "not_enough_data", "no_candidates"}:
            payload["active_stage"] = payload.get("active_stage") or payload.get("status")
            payload["active_target"] = status.get("active_target")
            payload["active_candidate"] = status.get("active_candidate")
        self.model_lab_job_status = payload
        self.state.model_status.update(
            {
                "model_lab_status": payload.get("status"),
                "model_lab_queue_size": payload.get("queue_size", 0),
                "model_lab_cache_hit_rate": payload.get("cache_hit_rate", 0.0),
                "model_lab_progress_pct": payload.get("progress_pct", 0.0),
                "model_lab_active_stage": payload.get("active_stage"),
                "model_lab_active_target": payload.get("active_target"),
            }
        )

    def _update_state(self, snapshot: AnalysisSnapshot, forecast_series: pd.Series) -> None:
        features = snapshot.features.features
        current_candle = self.state.data_status.get("current_candle")
        if not features.empty:
            latest_idx = features.index[-1]
            self.state.latest_closed_time = latest_idx
            self.state.forecast_vol = float(snapshot.forecast.basket_forecast.volatility)
            self.state.current_regime = str(snapshot.regimes.iloc[-1]) if not snapshot.regimes.empty else None
            self.state.current_shock_type = (
                str(snapshot.shocks["shock_type"].iloc[-1]) if not snapshot.shocks.empty else None
            )
            self.state.push_trend("basket_return", float(features["basket_return"].iloc[-1]))
            self.state.push_trend("forecast_vol", self.state.forecast_vol)
            if "trend_strength" in features:
                self.state.push_trend("trend_strength", float(features["trend_strength"].iloc[-1]))
            if not snapshot.shocks.empty and "shock_score" in snapshot.shocks:
                self.state.push_trend("regime_score", float(snapshot.shocks["shock_score"].iloc[-1]))
        model_lab_status = self.model_lab_job_status or (snapshot.model_lab.job_status if snapshot.model_lab else {})
        self.state.model_status = {
            **snapshot.forecast.status,
            "basket_forecast_model": snapshot.forecast.basket_forecast.model_name,
            "basket_forecast_degraded": snapshot.forecast.basket_forecast.degraded,
            "refit_countdown": snapshot.forecast.refit_countdown,
            "model_lab_status": model_lab_status.get("status", "not-run"),
            "model_lab_queue_size": model_lab_status.get("queue_size", 0),
            "model_lab_cache_hit_rate": model_lab_status.get("cache_hit_rate", 0.0),
            "model_lab_progress_pct": model_lab_status.get("progress_pct", 0.0),
            "model_lab_active_stage": model_lab_status.get("active_stage"),
            "model_lab_active_target": model_lab_status.get("active_target"),
            "cpp_backend_status": backend_status(),
        }
        if snapshot.model_lab is not None:
            self.state.model_status.update(snapshot.model_lab.summary())
        if self.model_selection is not None:
            self.state.model_status.update(
                {
                    "selected_model": self.model_selection.selected_model,
                    "model_selection_success": self.model_selection.success,
                    "model_selection_aic": self.model_selection.aic,
                    "model_selection_bic": self.model_selection.bic,
                    "model_selection_message": self.model_selection.message,
                    "last_model_selection_time": self.last_model_selection_time,
                }
            )
        self.state.data_status = {
            "rows": int(len(snapshot.candles)),
            "symbols": self.config.symbols,
            "validation": "ok" if snapshot.validation.is_valid else "issues",
            "validation_issues": len(snapshot.validation.issues),
            "validation_issue_details": [asdict(issue) for issue in snapshot.validation.issues],
            "symbol_rows": snapshot.validation.symbol_rows,
        }
        if current_candle is not None:
            self.state.data_status["current_candle"] = current_candle
        if snapshot.model_validation is not None:
            self.state.backtest_summary = snapshot.model_validation.summary()
        self.state.latest_diagnostics = snapshot.diagnostics
        self.state.last_stat_test_run = snapshot.diagnostics.generated_at
        self.state.process_timeline.extend(snapshot.diagnostics.method_steps)
        self.state.process_timeline = self.state.process_timeline[-120:]
        self._update_range_views(snapshot)

    def _update_range_views(self, snapshot: AnalysisSnapshot) -> None:
        latest = snapshot.features.features.index[-1] if not snapshot.features.features.empty else None
        dashboard_features, dashboard_meta = slice_by_time_range(
            snapshot.features.features,
            self.config.dashboard_time_range,
            latest=latest,
        )
        forecast_features, forecast_meta = slice_by_time_range(
            snapshot.features.features,
            self.config.forecast_time_range,
            latest=latest,
        )
        forecast_series = snapshot.historical_forecast.reindex(forecast_features.index)
        forecast_regimes = snapshot.regimes.reindex(forecast_features.index).ffill()
        forecast_shocks = snapshot.shocks.reindex(forecast_features.index)
        backtest_features, backtest_meta = slice_by_time_range(
            snapshot.features.features,
            self.config.backtest_time_range,
            latest=latest,
        )
        backtest_forecast = snapshot.historical_forecast.reindex(backtest_features.index)
        backtest_regimes = snapshot.regimes.reindex(backtest_features.index).ffill()
        self.range_model_validation = None
        if len(backtest_features.dropna(how="all")) > 2 and "basket_return" in backtest_features:
            self.range_model_validation = run_walk_forward_validation(
                basket_returns=backtest_features["basket_return"],
                forecast_vol=backtest_forecast,
                ewma_vol=backtest_features.get("ewma_vol", pd.Series(dtype=float)),
                realized_vol=backtest_features.get("market_vol", pd.Series(dtype=float)),
                regimes=backtest_regimes,
            )
        garch_params = dict(self.forecaster.basket_model._fit_result.params)
        active_diagnostics = run_statistical_diagnostics(
            basket_returns=forecast_features.get("basket_return", pd.Series(dtype=float)),
            forecast_vol=forecast_series,
            ewma_vol=forecast_features.get("ewma_vol", pd.Series(dtype=float)),
            regimes=forecast_regimes,
            shocks=forecast_shocks,
            validation_losses=self.range_model_validation.losses if self.range_model_validation is not None else None,
            garch_params=garch_params,
            generated_at=pd.Timestamp.now(tz="UTC"),
            periods_per_year=self.config.periods_per_year,
        )
        active_diagnostics.method_steps = snapshot.diagnostics.method_steps
        self.range_diagnostics = active_diagnostics
        self.state.latest_diagnostics = active_diagnostics
        self.state.last_stat_test_run = active_diagnostics.generated_at
        self.state.range_status = {
            "data_interval": self.config.interval,
            "forecast_horizon": self.config.forecast_horizon,
            "window_bar_counts": self.config.window_bar_counts,
            "dashboard": dashboard_meta.to_record(),
            "forecast": forecast_meta.to_record(),
            "backtest": backtest_meta.to_record(),
            "available_ranges": ", ".join(value.upper() for value in self.config.allowed_time_ranges),
        }
        if self.range_model_validation is not None:
            self.state.backtest_summary = {
                "backtest_range": self.config.backtest_time_range.upper(),
                "backtest_start": backtest_meta.start,
                "backtest_end": backtest_meta.end,
                "backtest_observations": backtest_meta.actual_points,
                **self.range_model_validation.metrics,
            }
        else:
            self.state.backtest_summary = {
                "backtest_range": self.config.backtest_time_range.upper(),
                "backtest_observations": backtest_meta.actual_points,
                "status": "not enough observations",
            }

    def _persist_analysis(self, snapshot: AnalysisSnapshot) -> None:
        try:
            save_frame(snapshot.features.features, self.config.features_dir / "features.parquet")
            if snapshot.model_validation is not None:
                save_frame(snapshot.model_validation.losses, self.config.validation_dir / "model_validation_losses.parquet")
            if snapshot.model_lab is not None:
                save_frame(snapshot.model_lab.leaderboard, self.config.models_dir / "model_lab_leaderboard.parquet")
                save_frame(snapshot.model_lab.timelines, self.config.models_dir / "model_lab_timelines.parquet")
        except Exception as exc:
            self.state.log(f"Artifact persistence skipped: {exc}")

    def _maybe_run_model_selection(self, basket_returns: pd.Series) -> None:
        now = pd.Timestamp.now(tz="UTC")
        due = self.last_model_selection_time is None or now - self.last_model_selection_time >= pd.Timedelta(days=7)
        if not due:
            return
        result = select_volatility_model(basket_returns.dropna())
        self.model_selection = result
        self.last_model_selection_time = now

    def _active_interval_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        if candles.empty or "interval" not in candles:
            return candles
        return candles.loc[candles["interval"] == self.config.interval].copy()


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    configure_logging()
    from cmva.web.app import serve

    serve(CMVAApplication(), host=host, port=port, open_browser=open_browser)
