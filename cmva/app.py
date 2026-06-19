"""Top-level CMVA application orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cmva.backtest.engine import BacktestResult, run_historical_backtest
from cmva.backtest.live_paper import LivePaperBacktest
from cmva.config import CMVAConfig, ensure_artifact_dirs, load_config
from cmva.data.candle import Candle, candles_to_frame
from cmva.data.rest_client import BinanceRestClient
from cmva.data.storage import CandleStorage, save_frame
from cmva.data.validation import ValidationReport, validate_candles
from cmva.data.websocket_client import BinanceWebSocketClient
from cmva.features import FeatureBundle, compute_feature_bundle
from cmva.forecast.volatility_forecaster import ForecastSnapshot, VolatilityForecaster
from cmva.logging_config import configure_logging
from cmva.models.garch import historical_garch_forecast
from cmva.models.selection import ModelSelectionResult, select_volatility_model
from cmva.policy.regime_vol_target import RegimeVolTargetPolicy
from cmva.regime.classifier import classify_regime_series
from cmva.regime.shock import compute_shock_series
from cmva.reports.html import export_html_report
from cmva.reports.markdown import export_markdown_report, export_validation_report
from cmva.reports.plots import export_equity_svg
from cmva.state import AppState


@dataclass
class AnalysisSnapshot:
    candles: pd.DataFrame
    features: FeatureBundle
    forecast: ForecastSnapshot
    regimes: pd.Series
    shocks: pd.DataFrame
    backtest: BacktestResult | None
    validation: ValidationReport


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
        self.paper = LivePaperBacktest(
            transaction_cost_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
        )
        self.snapshot: AnalysisSnapshot | None = None
        self.model_selection: ModelSelectionResult | None = None
        self.last_model_selection_time: pd.Timestamp | None = None

    async def bootstrap(self, fetch_remote: bool = True) -> None:
        self.state.mode = "BOOTSTRAP"
        self.state.log("Loading local candle cache")
        candles = self.storage.load_many(self.config.symbols)
        if fetch_remote:
            try:
                candles = await self.fetch_missing_history(candles)
            except Exception as exc:
                self.state.mode = "DEGRADED"
                self.state.log(f"Historical fetch degraded: {exc}")
        if candles.empty:
            self.state.mode = "DEGRADED"
            self.state.log("No candles available yet")
            self.state.data_status = {"rows": 0, "validation": "no data"}
            return
        self.recompute(candles, force_refit=True)
        self.paper.set_next_exposure(self.state.target_exposure or 0.0)
        if self.state.mode != "DEGRADED":
            self.state.mode = "LIVE"

    async def fetch_missing_history(self, existing: pd.DataFrame) -> pd.DataFrame:
        now = pd.Timestamp(datetime.now(timezone.utc)).floor(self.config.interval)
        desired_start = now - pd.Timedelta(days=self.config.historical_days)
        fetched: list[Candle] = []
        for symbol in self.config.symbols:
            symbol_existing = existing.loc[existing["symbol"] == symbol] if not existing.empty else pd.DataFrame()
            if symbol_existing.empty:
                start = desired_start
            else:
                latest = pd.to_datetime(symbol_existing["open_time"], utc=True).max()
                start = max(latest + pd.Timedelta(self.config.interval), desired_start)
            if start <= now:
                self.state.log(f"Fetching {symbol} klines from {start}")
                fetched.extend(
                    await self.rest_client.fetch_historical_range(
                        symbol=symbol,
                        interval=self.config.interval,
                        start_time=start,
                        end_time=now,
                        limit=self.config.bootstrap_limit,
                    )
                )
        if fetched:
            self.storage.upsert_closed(fetched)
        return self.storage.load_many(self.config.symbols)

    def recompute(self, candles: pd.DataFrame, force_refit: bool = False) -> AnalysisSnapshot:
        validation = validate_candles(candles, self.config.interval, self.config.symbols)
        features = compute_feature_bundle(
            candles,
            short_window=self.config.rolling_short_window,
            medium_window=self.config.rolling_medium_window,
            long_window=self.config.rolling_long_window,
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
        backtest = None
        if len(features.returns.dropna(how="all")) > 2:
            backtest = run_historical_backtest(
                returns=features.returns,
                forecast_vol=historical_forecast,
                regimes=regimes,
                target_vol_per_period=self.config.target_vol_per_period,
                max_leverage=self.config.max_leverage,
                transaction_cost_bps=self.config.transaction_cost_bps,
                slippage_bps=self.config.slippage_bps,
                periods_per_year=self.config.periods_per_year,
            )
        snapshot = AnalysisSnapshot(candles, features, forecast, regimes, shocks, backtest, validation)
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
        if self.state.paused:
            self.state.log(f"Paused; received closed candle for {candle.symbol} at {candle.open_time}")
            return
        candles = self.storage.upsert_closed([candle])
        all_candles = self.storage.load_many(self.config.symbols)
        snapshot = self.recompute(all_candles, force_refit=False)
        latest_return = snapshot.features.features["basket_return"].dropna()
        if not latest_return.empty:
            self.paper.settle_closed_candle(latest_return.index[-1], float(latest_return.iloc[-1]))
            self.paper.set_next_exposure(self.state.target_exposure or 0.0)
            self.state.live_paper_pnl = self.paper.cumulative_return
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
        candles = self.storage.load_many(self.config.symbols)
        if not candles.empty:
            self.recompute(candles, force_refit=True)
            self.state.log("Forced GARCH refit")

    def rerun_backtest(self) -> None:
        if self.snapshot is not None:
            self.recompute(self.snapshot.candles, force_refit=False)
            self.state.log("Historical backtest recomputed")

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
            "historical_days",
            "rolling_short_window",
            "rolling_medium_window",
            "rolling_long_window",
            "min_threshold_history",
            "garch_refit_frequency",
            "target_annual_vol",
            "max_leverage",
            "transaction_cost_bps",
            "slippage_bps",
            "severe_shock_threshold",
            "moderate_shock_threshold",
            "shock_breadth_threshold",
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
        self.paper.transaction_cost_bps = self.config.transaction_cost_bps
        self.paper.slippage_bps = self.config.slippage_bps
        if recompute and self.snapshot is not None:
            self.recompute(self.snapshot.candles, force_refit=True)
            self.state.log("Settings applied and historical backtest recomputed")
        else:
            self.state.log("Settings applied from now")

    def cancel_settings(self) -> None:
        self.state.log("Settings change canceled")

    def toggle_pause(self) -> None:
        self.state.paused = not self.state.paused
        self.state.mode = "PAUSED" if self.state.paused else "LIVE"
        self.state.log("Paused live processing" if self.state.paused else "Resumed live processing")

    def export_report(self) -> tuple[Path, Path]:
        summary = self.summary()
        metrics = self.snapshot.backtest.metrics if self.snapshot and self.snapshot.backtest else None
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        markdown_path = self.config.reports_dir / f"cmva_report_{stamp}.md"
        html_path = self.config.reports_dir / f"cmva_report_{stamp}.html"
        plots: list[Path] = []
        if self.snapshot is not None and self.snapshot.backtest is not None:
            equity_plot = export_equity_svg(
                self.snapshot.backtest.equity,
                self.config.reports_dir / f"equity_curve_{stamp}.svg",
            )
            if equity_plot is not None:
                plots.append(equity_plot)
        export_markdown_report(markdown_path, summary, metrics, plots)
        export_html_report(html_path, summary, metrics, plots)
        if self.snapshot is not None:
            export_validation_report(self.config.reports_dir / "validation_report.md", self.snapshot.validation.to_markdown())
        self.state.log(f"Exported report {markdown_path}")
        return markdown_path, html_path

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.state.mode,
            "latest_closed_time": self.state.latest_closed_time,
            "current_regime": self.state.current_regime,
            "current_shock_type": self.state.current_shock_type,
            "forecast_vol_1h": self.state.forecast_vol_1h,
            "target_exposure": self.state.target_exposure,
            "live_paper_pnl": self.state.live_paper_pnl,
            "websocket_status": self.state.websocket_status,
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

    def _update_state(self, snapshot: AnalysisSnapshot, forecast_series: pd.Series) -> None:
        features = snapshot.features.features
        if not features.empty:
            latest_idx = features.index[-1]
            self.state.latest_closed_time = latest_idx
            self.state.forecast_vol_1h = float(snapshot.forecast.basket_forecast.volatility)
            self.state.current_regime = str(snapshot.regimes.iloc[-1]) if not snapshot.regimes.empty else None
            self.state.current_shock_type = (
                str(snapshot.shocks["shock_type"].iloc[-1]) if not snapshot.shocks.empty else None
            )
            policy = RegimeVolTargetPolicy(self.config.target_vol_per_period, self.config.max_leverage)
            self.state.target_exposure = policy.target_exposure(
                float(snapshot.forecast.basket_forecast.volatility),
                self.state.current_regime,
            )
        self.state.model_status = {
            **snapshot.forecast.status,
            "basket_forecast_model": snapshot.forecast.basket_forecast.model_name,
            "basket_forecast_degraded": snapshot.forecast.basket_forecast.degraded,
            "refit_countdown": snapshot.forecast.refit_countdown,
        }
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
        }
        if snapshot.backtest is not None:
            strategy = snapshot.backtest.metrics.get("regime_aware_vol_target", {})
            self.state.backtest_summary = {
                "backtest_cumulative_return": strategy.get("cumulative_return"),
                "backtest_max_drawdown": strategy.get("max_drawdown"),
                "backtest_sharpe": strategy.get("sharpe"),
                "backtest_turnover": strategy.get("turnover"),
            }

    def _persist_analysis(self, snapshot: AnalysisSnapshot) -> None:
        try:
            save_frame(snapshot.features.features, self.config.features_dir / "features.parquet")
            if snapshot.backtest is not None:
                save_frame(snapshot.backtest.returns, self.config.backtests_dir / "strategy_returns.parquet")
                save_frame(snapshot.backtest.equity, self.config.backtests_dir / "equity.parquet")
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


def run() -> None:
    configure_logging()
    from cmva.tui.app import CMVATuiApp

    CMVATuiApp(CMVAApplication()).run()
