"""FastAPI web application for CMVA."""

from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cmva.app import CMVAApplication
from cmva.engine.interval import bars_for_duration, interval_to_timedelta
from cmva.models.lab import default_candidate_specs, generate_rolling_origins
from cmva.native.backend import backend_status
from cmva.simulation.repository import SimulationRepository
from cmva.simulation.runner import create_runner_for_app
from cmva.simulation.snapshot import build_simulation_snapshot
from cmva.simulation.spec import SimulationSpec
from cmva.web.glossary import load_glossary


PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


def create_web_app(cmva: CMVAApplication | None = None, start_background: bool = True) -> FastAPI:
    cmva_app = cmva or CMVAApplication()
    app = FastAPI(title="CMVA", version="0.1.0", lifespan=_lifespan(cmva_app, start_background))
    app.state.cmva = cmva_app
    app.state.background_threads = []
    app.state.simulation_repository = SimulationRepository(cmva_app.config.simulations_dir)
    app.state.simulation_repository.mark_interrupted_runs()
    app.state.simulation_tasks = {}
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def setup_home(request: Request) -> HTMLResponse:
        return _render(request, "simulation_new.html", page="simulation_new")

    @app.get("/setup", response_class=HTMLResponse)
    async def setup(request: Request) -> HTMLResponse:
        return _render(request, "setup.html", page="setup")

    @app.get("/overview", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        return _render(request, "overview.html", page="overview")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_alias(request: Request) -> HTMLResponse:
        return _render(request, "overview.html", page="overview")

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/markets", response_class=HTMLResponse)
    async def markets(request: Request) -> HTMLResponse:
        return _render(request, "data_quality.html", page="data_quality")

    @app.get("/data-quality", response_class=HTMLResponse)
    async def data_quality(request: Request) -> HTMLResponse:
        return _render(request, "data_quality.html", page="data_quality")

    @app.get("/volatility", response_class=HTMLResponse)
    async def volatility(request: Request) -> HTMLResponse:
        return _render(request, "volatility.html", page="volatility")

    @app.get("/trend", response_class=HTMLResponse)
    async def trend(request: Request) -> HTMLResponse:
        return _render(request, "trend_seasonality.html", page="trend_seasonality")

    @app.get("/trend-seasonality", response_class=HTMLResponse)
    async def trend_seasonality(request: Request) -> HTMLResponse:
        return _render(request, "trend_seasonality.html", page="trend_seasonality")

    @app.get("/correlation-pca", response_class=HTMLResponse)
    async def correlation_pca(request: Request) -> HTMLResponse:
        return _render(request, "correlation_pca.html", page="correlation_pca")

    @app.get("/shock-regime", response_class=HTMLResponse)
    async def shock_regime(request: Request) -> HTMLResponse:
        return _render(request, "shock_regime.html", page="shock_regime")

    @app.get("/current-market-state", response_class=HTMLResponse)
    async def current_market_state(request: Request) -> HTMLResponse:
        return _render(request, "current_market_state.html", page="current_market_state")

    @app.get("/models", response_class=HTMLResponse)
    async def models(request: Request) -> HTMLResponse:
        return _render(request, "model_lab.html", page="model_lab")

    @app.get("/model-lab", response_class=HTMLResponse)
    async def model_lab(request: Request) -> HTMLResponse:
        return _render(request, "model_lab.html", page="model_lab")

    @app.get("/validation", response_class=HTMLResponse)
    async def validation(request: Request) -> HTMLResponse:
        return _render(request, "rolling_evaluation.html", page="rolling_evaluation")

    @app.get("/rolling-evaluation", response_class=HTMLResponse)
    async def rolling_evaluation(request: Request) -> HTMLResponse:
        return _render(request, "rolling_evaluation.html", page="rolling_evaluation")

    @app.get("/model-comparison", response_class=HTMLResponse)
    async def model_comparison(request: Request) -> HTMLResponse:
        return _render(request, "model_comparison.html", page="model_comparison")

    @app.get("/diagnostics", response_class=HTMLResponse)
    async def diagnostics(request: Request) -> HTMLResponse:
        return _render(request, "diagnostics.html", page="diagnostics")

    @app.get("/methodology", response_class=HTMLResponse)
    async def methodology(request: Request) -> HTMLResponse:
        return _render(request, "methodology.html", page="methodology")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        return _render(request, "settings.html", page="settings")

    @app.post("/settings")
    async def update_settings(request: Request) -> RedirectResponse:
        form = await request.form()
        symbols = str(form.get("symbols", ", ".join(cmva_app.config.symbols)))
        candidate_groups = form.getlist("candidate_model_groups") or [
            item.strip() for item in str(form.get("candidate_model_groups_csv", "")).split(",") if item.strip()
        ]
        cmva_app.apply_settings(
            {
                "symbols": [item.strip().upper() for item in symbols.split(",") if item.strip()],
                "interval": str(form.get("interval", cmva_app.config.interval)),
                "analysis_period": str(form.get("analysis_period", cmva_app.config.analysis_period)),
                "training_window": str(form.get("training_window", cmva_app.config.training_window)),
                "forecast_horizon_bars": _int_form(form, "forecast_horizon_bars", cmva_app.config.forecast_horizon_bars),
                "refit_stride_bars": _int_form(form, "refit_stride_bars", cmva_app.config.refit_stride_bars),
                "search_mode": str(form.get("search_mode", cmva_app.config.search_mode)),
                "candidate_model_count": _int_form(form, "candidate_model_count", cmva_app.config.candidate_model_count),
                "candidate_model_groups": candidate_groups or cmva_app.config.candidate_model_groups,
                "target_view": str(form.get("target_view", cmva_app.config.target_view)),
                "volatility_window": str(form.get("volatility_window", cmva_app.config.volatility_window)),
                "correlation_window": str(form.get("correlation_window", cmva_app.config.correlation_window)),
                "pca_window": str(form.get("pca_window", cmva_app.config.pca_window)),
                "trend_window": str(form.get("trend_window", cmva_app.config.trend_window)),
                "regime_threshold_window": str(
                    form.get("regime_threshold_window", cmva_app.config.regime_threshold_window)
                ),
                "garch_refit_frequency": _int_form(form, "garch_refit_frequency", cmva_app.config.garch_refit_frequency),
                "severe_shock_threshold": _float_form(
                    form, "severe_shock_threshold", cmva_app.config.severe_shock_threshold
                ),
                "moderate_shock_threshold": _float_form(
                    form, "moderate_shock_threshold", cmva_app.config.moderate_shock_threshold
                ),
            },
            recompute=True,
        )
        cmva_app.apply_view_ranges(
            {
                "dashboard_time_range": str(form.get("dashboard_time_range", cmva_app.config.dashboard_time_range)),
                "forecast_time_range": str(form.get("forecast_time_range", cmva_app.config.forecast_time_range)),
                "backtest_time_range": str(form.get("backtest_time_range", cmva_app.config.backtest_time_range)),
            },
            recompute_backtest=True,
        )
        return RedirectResponse(str(form.get("next_path", "/setup")), status_code=303)

    @app.get("/logs", response_class=HTMLResponse)
    async def logs(request: Request) -> HTMLResponse:
        return _render(request, "logs.html", page="logs")

    @app.get("/simulation/new", response_class=HTMLResponse)
    async def simulation_new(request: Request) -> HTMLResponse:
        return _render(request, "simulation_new.html", page="simulation_new")

    @app.get("/simulations", response_class=HTMLResponse)
    async def simulations(request: Request) -> HTMLResponse:
        return _render(request, "simulations.html", page="simulations")

    @app.post("/api/simulations")
    async def create_simulation(request: Request) -> Response:
        data = await _request_payload(request)
        try:
            spec = SimulationSpec.from_mapping(data)
        except ValueError as exc:
            return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=400)
        repository: SimulationRepository = app.state.simulation_repository
        run_id = repository.create_run(spec)
        runner = create_runner_for_app(
            cmva_app.config,
            repository,
            cmva_app.rest_client,
            cmva_app.storage,
        )
        thread = threading.Thread(target=_run_simulation_thread, args=(runner, spec), daemon=True)
        thread.start()
        app.state.simulation_tasks[run_id] = thread
        accepts = request.headers.get("accept", "")
        if "application/json" in accepts or request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"ok": True, "run_id": run_id, "redirect_url": f"/simulation/{run_id}"}, status_code=201)
        return RedirectResponse(f"/simulation/{run_id}", status_code=303)

    @app.get("/simulation/{run_id}", response_class=HTMLResponse)
    async def simulation_run(request: Request, run_id: str) -> HTMLResponse:
        snapshot = _load_simulation_or_404(app.state.simulation_repository, run_id)
        return _render(request, "simulation_run.html", page="simulation_run", simulation=snapshot)

    @app.get("/results/{run_id}", response_class=HTMLResponse)
    async def simulation_results(request: Request, run_id: str) -> HTMLResponse:
        repository: SimulationRepository = app.state.simulation_repository
        try:
            results = repository.load_final_results(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="시뮬레이션 실행을 찾을 수 없습니다.") from exc
        return _render(request, "results.html", page="simulation_results", simulation=results)

    @app.get("/api/simulations/{run_id}")
    async def api_simulation(run_id: str) -> JSONResponse:
        snapshot = _load_simulation_or_404(app.state.simulation_repository, run_id)
        return JSONResponse(_jsonable(snapshot))

    @app.get("/api/simulations/{run_id}/results")
    async def api_simulation_results(run_id: str) -> JSONResponse:
        repository: SimulationRepository = app.state.simulation_repository
        try:
            return JSONResponse(_jsonable(repository.load_final_results(run_id)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="시뮬레이션 실행을 찾을 수 없습니다.") from exc

    @app.websocket("/ws/simulations/{run_id}")
    async def simulation_ws(socket: WebSocket, run_id: str) -> None:
        repository: SimulationRepository = app.state.simulation_repository
        await socket.accept()
        try:
            while True:
                await socket.send_text(json.dumps(_jsonable(build_simulation_snapshot(repository, run_id))))
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, KeyError):
            return

    @app.get("/api/snapshot")
    async def api_snapshot() -> JSONResponse:
        return JSONResponse(_jsonable(build_snapshot(cmva_app)))

    @app.post("/api/refresh")
    async def api_refresh() -> JSONResponse:
        if cmva_app.snapshot is not None:
            cmva_app.recompute(cmva_app.snapshot.candles, force_refit=False)
        return JSONResponse(_jsonable(build_snapshot(cmva_app)))

    @app.websocket("/ws/snapshot")
    async def snapshot_ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                await socket.send_text(json.dumps(_jsonable(build_snapshot(cmva_app))))
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            return

    def _render(request: Request, template_name: str, page: str, **extra: Any) -> HTMLResponse:
        snapshot = _jsonable(build_snapshot(cmva_app))
        return templates.TemplateResponse(
            request,
            template_name,
            {
                "request": request,
                "page": page,
                "cmva": cmva_app,
                "snapshot": snapshot,
                "glossary": load_glossary(),
                "simulation_runs": app.state.simulation_repository.list_runs(),
                **extra,
            },
        )

    return app


async def _request_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return dict(await request.json())
    form = await request.form()
    payload: dict[str, Any] = dict(form)
    groups = form.getlist("candidate_model_groups")
    if groups:
        payload["candidate_model_groups"] = groups
    return payload


def _load_simulation_or_404(repository: SimulationRepository, run_id: str) -> dict[str, Any]:
    try:
        return build_simulation_snapshot(repository, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="시뮬레이션 실행을 찾을 수 없습니다.") from exc


def _run_simulation_thread(runner, spec: SimulationSpec) -> None:
    asyncio.run(runner.run(spec))


def _lifespan(cmva: CMVAApplication, start_background: bool):
    @asynccontextmanager
    async def manager(app: FastAPI) -> AsyncIterator[None]:
        if start_background:
            thread = threading.Thread(target=_run_background_worker, args=(cmva,), daemon=True)
            thread.start()
            app.state.background_threads.append(thread)
        try:
            yield
        finally:
            cmva.state.log("Web server shutdown requested")

    return manager


def serve(cmva: CMVAApplication, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"CMVA dashboard: {url}")
    uvicorn.run(create_web_app(cmva, start_background=True), host=host, port=port, log_level="info")


async def _bootstrap_and_stream(cmva: CMVAApplication) -> None:
    await cmva.bootstrap(fetch_remote=True)
    if cmva.state.mode != "DEGRADED":
        try:
            await cmva.stream_live()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cmva.state.websocket_status = "error"
            cmva.state.mode = "DEGRADED"
            cmva.state.log(f"WebSocket stopped: {exc}")


def _run_background_worker(cmva: CMVAApplication) -> None:
    try:
        asyncio.run(_bootstrap_and_stream(cmva))
    except Exception as exc:
        cmva.state.websocket_status = "error"
        cmva.state.mode = "DEGRADED"
        cmva.state.log(f"Background worker stopped: {exc}")


def build_snapshot(cmva: CMVAApplication) -> dict[str, Any]:
    latest_features = {}
    latest_shock = {}
    series: dict[str, list[dict[str, Any]]] = {}
    market_rows: list[dict[str, Any]] = []
    data_accumulation = _empty_data_accumulation()
    data_quality = _empty_data_quality()
    pipeline = _empty_pipeline()
    recent_candles: list[dict[str, Any]] = []
    validation = cmva.range_model_validation or (cmva.snapshot.model_validation if cmva.snapshot else None)
    model_lab = cmva.snapshot.model_lab if cmva.snapshot is not None else None
    model_lab_job_status = dict(getattr(cmva, "model_lab_job_status", {}) or {})
    if model_lab is not None and model_lab_job_status.get("status") == "idle":
        model_lab_job_status = dict(model_lab.job_status)
    model_lab_summary = model_lab.summary() if model_lab is not None else {}
    if model_lab_job_status:
        model_lab_summary = {**model_lab_summary, **model_lab_job_status}
    if cmva.snapshot is not None:
        features = cmva.snapshot.features.features
        if not features.empty:
            latest_features = features.iloc[-1].dropna().to_dict()
            for name in (
                "basket_return",
                "market_vol",
                "ewma_vol",
                "range_vol",
                "vol_percentile",
                "vol_z_score",
                "vol_of_vol",
                "avg_pairwise_corr",
                "avg_btc_beta",
                "pca1_share",
                "dispersion",
                "trend_slope",
                "trend_tstat",
                "trend_strength",
                "trend_autocorr",
                "up_down_ratio",
            ):
                series[name] = _series_records(features.get(name))
            series["log_return"] = _series_records(features.get("basket_return"))
            basket_price = cmva.snapshot.features.close.mean(axis=1, skipna=True)
            log_price = np.log(basket_price.where(basket_price > 0)).replace([np.inf, -np.inf], np.nan)
            series["log_price"] = _series_records(log_price.rename("log_price"))
            series["forecast_vol"] = _series_records(cmva.snapshot.historical_forecast)
            prior_forecast = cmva.snapshot.historical_forecast.reindex(features.index).shift(1).replace(0.0, np.nan)
            standardized = (features["basket_return"] / prior_forecast).replace([np.inf, -np.inf], np.nan)
            series["standardized_residual"] = _series_records(standardized)
            series["regime_state"] = _categorical_series_records(cmva.snapshot.regimes)
        if not cmva.snapshot.shocks.empty:
            latest_shock = cmva.snapshot.shocks.iloc[-1].dropna().to_dict()
            for name in ("shock_score", "shock_breadth", "rv_jump_ratio"):
                series[name] = _series_records(cmva.snapshot.shocks.get(name))
        market_rows = _latest_market_rows(cmva.snapshot.candles)
        data_accumulation = _data_accumulation(cmva.snapshot.candles, cmva.config.symbols, cmva.config.interval)
        data_quality = _data_quality(cmva.snapshot.validation)
        pipeline = _pipeline(cmva, cmva.snapshot, validation)
        recent_candles = _recent_candle_rows(cmva.snapshot.candles)
    if validation is not None:
        for name in ("garch_qlike", "ewma_qlike", "naive_qlike"):
            series[name] = _series_records(validation.losses.get(name))
        for model in ("garch", "ewma", "naive"):
            squared = validation.losses.get(f"{model}_squared_error")
            absolute = validation.losses.get(f"{model}_absolute_error")
            if squared is not None:
                series[f"{model}_rmse_loss"] = _series_records(np.sqrt(pd.to_numeric(squared, errors="coerce")))
            if absolute is not None:
                series[f"{model}_mae_loss"] = _series_records(absolute)
    return {
        "summary": cmva.summary(),
        "mode": cmva.state.mode,
        "bootstrap_progress": cmva.state.bootstrap_progress,
        "data_accumulation": data_accumulation,
        "data_quality": data_quality,
        "pipeline": pipeline,
        "setup_estimate": _setup_estimate(cmva, data_accumulation),
        "forming_candle": cmva.state.data_status.get("current_candle"),
        "data_policy": {
            "research_dataset": "closed candles only",
            "forming_candle_usage": "display only",
        },
        "recent_candles": recent_candles,
        "latest_features": latest_features,
        "latest_shock": latest_shock,
        "market_rows": market_rows,
        "series": series,
        "model_status": cmva.state.model_status,
        "cpp_backend": backend_status(),
        "model_lab": {
            "summary": model_lab_summary,
            "stage1": _frame_records(model_lab.stage1, limit=200) if model_lab is not None else [],
            "stage2": _frame_records(model_lab.stage2, limit=200) if model_lab is not None else [],
            "leaderboard": _frame_records(model_lab.leaderboard, limit=200) if model_lab is not None else [],
            "timelines": _frame_records(model_lab.timelines, limit=400) if model_lab is not None else [],
            "rank_heatmap": _heatmap_records(model_lab.rank_heatmap) if model_lab is not None else [],
            "selected": model_lab.selected if model_lab is not None else {},
            "current_state": model_lab.current_state if model_lab is not None else {},
            "job_status": model_lab_job_status,
        },
        "range_status": cmva.state.range_status,
        "validation": validation.summary() if validation is not None else {},
        "validation_losses": _frame_records(validation.losses) if validation is not None else [],
        "calibration_by_decile": _frame_records(validation.calibration_by_decile) if validation is not None else [],
        "realized_vol_by_regime": _frame_records(validation.realized_vol_by_regime) if validation is not None else [],
        "diagnostics": {
            "model": [item.to_record() for item in cmva.state.latest_diagnostics.model_tests],
            "forecast": [item.to_record() for item in cmva.state.latest_diagnostics.forecast_tests],
            "risk": [item.to_record() for item in cmva.state.latest_diagnostics.risk_tests],
            "validation": [item.to_record() for item in cmva.state.latest_diagnostics.backtest_tests],
            "regime": [item.to_record() for item in cmva.state.latest_diagnostics.regime_tests],
        },
        "logs": cmva.state.logs[-100:],
    }


def _latest_market_rows(candles: pd.DataFrame) -> list[dict[str, Any]]:
    if candles.empty:
        return []
    ordered = candles.sort_values("open_time").groupby("symbol", as_index=False).tail(1)
    return ordered.sort_values("symbol").to_dict(orient="records")


def _empty_data_accumulation() -> dict[str, Any]:
    return {
        "total_rows": 0,
        "symbols": [],
        "first_open_time": None,
        "latest_open_time": None,
        "coverage_hours": None,
        "expected_symbols": [],
        "interval": None,
    }


def _empty_data_quality() -> dict[str, Any]:
    return {
        "status": "no data",
        "is_valid": False,
        "issue_count": 0,
        "issues": [],
        "counts_by_check": {},
        "counts_by_symbol": {},
        "symbol_rows": {},
    }


def _empty_pipeline() -> dict[str, Any]:
    return {
        "data_rows": 0,
        "feature_rows": 0,
        "forecast_points": 0,
        "model": None,
        "validation_observations": 0,
    }


def _pipeline(cmva: CMVAApplication, snapshot, validation) -> dict[str, Any]:
    return {
        "data_rows": int(len(snapshot.candles)),
        "feature_rows": int(len(snapshot.features.features)),
        "forecast_points": int(snapshot.historical_forecast.dropna().shape[0]),
        "model": cmva.state.model_status.get("basket_forecast_model"),
        "validation_observations": (
            int(validation.sample_size) if validation is not None and getattr(validation, "sample_size", None) is not None else 0
        ),
    }


def _data_accumulation(candles: pd.DataFrame, expected_symbols: list[str], interval: str) -> dict[str, Any]:
    if candles.empty:
        empty = _empty_data_accumulation()
        empty["expected_symbols"] = expected_symbols
        empty["interval"] = interval
        return empty
    data = candles.sort_values(["symbol", "open_time"])
    first = pd.to_datetime(data["open_time"], utc=True).min()
    latest = pd.to_datetime(data["open_time"], utc=True).max()
    coverage_hours = None
    if pd.notna(first) and pd.notna(latest):
        coverage_hours = float((latest - first) / pd.Timedelta(hours=1))
    symbols = []
    expected_delta = interval_to_timedelta(interval)
    for symbol, group in data.groupby("symbol"):
        ordered = group.sort_values("open_time")
        unique_open_times = ordered["open_time"].drop_duplicates()
        expected_rows = None
        estimated_missing = None
        if len(unique_open_times) >= 2:
            expected_rows = int(((unique_open_times.iloc[-1] - unique_open_times.iloc[0]) / expected_delta) + 1)
            estimated_missing = max(expected_rows - len(unique_open_times), 0)
        missing_gap_count = int(ordered["open_time"].diff().gt(expected_delta).sum())
        duplicate_count = int(ordered.duplicated(["symbol", "interval", "open_time"], keep=False).sum())
        bad_ohlc = ordered.loc[
            (ordered["high"] < ordered[["open", "close", "low"]].max(axis=1))
            | (ordered["low"] > ordered[["open", "close", "high"]].min(axis=1))
            | (ordered[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ]
        symbols.append(
            {
                "symbol": symbol,
                "rows": int(len(ordered)),
                "first_open_time": ordered["open_time"].iloc[0],
                "latest_open_time": ordered["open_time"].iloc[-1],
                "closed_rows": int(ordered["is_closed"].sum()),
                "unclosed_rows": int((~ordered["is_closed"]).sum()),
                "missing_gap_count": missing_gap_count,
                "estimated_missing_candles": estimated_missing,
                "duplicate_rows": duplicate_count,
                "ohlc_issue_rows": int(len(bad_ohlc)),
            }
        )
    return {
        "total_rows": int(len(data)),
        "symbols": symbols,
        "first_open_time": first,
        "latest_open_time": latest,
        "coverage_hours": coverage_hours,
        "expected_symbols": expected_symbols,
        "interval": interval,
    }


def _data_quality(report) -> dict[str, Any]:
    if report is None:
        return _empty_data_quality()
    issues = [issue.__dict__ for issue in report.issues]
    status = "ok"
    if not report.is_valid:
        status = "issues"
    elif report.issues:
        status = "warnings"
    counts_by_check: dict[str, int] = {}
    counts_by_symbol: dict[str, int] = {}
    for issue in report.issues:
        counts_by_check[issue.check] = counts_by_check.get(issue.check, 0) + int(issue.count or 1)
        symbol = issue.symbol or "ALL"
        counts_by_symbol[symbol] = counts_by_symbol.get(symbol, 0) + int(issue.count or 1)
    return {
        "status": status,
        "is_valid": report.is_valid,
        "issue_count": len(report.issues),
        "issues": issues,
        "counts_by_check": counts_by_check,
        "counts_by_symbol": counts_by_symbol,
        "symbol_rows": report.symbol_rows,
    }


def _recent_candle_rows(candles: pd.DataFrame, limit: int = 80) -> list[dict[str, Any]]:
    if candles.empty:
        return []
    columns = ["symbol", "interval", "open_time", "open", "high", "low", "close", "volume", "is_closed"]
    available = [column for column in columns if column in candles]
    return candles.sort_values("open_time").tail(limit)[available].to_dict(orient="records")


def _series_records(series: pd.Series | None, limit: int = 160) -> list[dict[str, Any]]:
    if series is None or series.empty:
        return []
    clean = pd.to_numeric(series, errors="coerce").dropna().tail(limit)
    return [{"time": index, "value": value} for index, value in clean.items()]


def _categorical_series_records(series: pd.Series | None, limit: int = 160) -> list[dict[str, Any]]:
    if series is None or series.empty:
        return []
    clean = series.dropna().tail(limit)
    categories = {value: index for index, value in enumerate(sorted(clean.astype(str).unique()), start=1)}
    return [{"time": index, "value": categories[str(value)], "label": str(value)} for index, value in clean.items()]


def _frame_records(frame: pd.DataFrame, limit: int = 160) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    data = frame.tail(limit).reset_index()
    return data.to_dict(orient="records")


def _heatmap_records(frame: pd.DataFrame, limit_columns: int = 80) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    limited = frame.iloc[:, -limit_columns:].copy()
    records = []
    for candidate, row in limited.iterrows():
        record = {"candidate": candidate}
        record.update({str(column): value for column, value in row.items()})
        records.append(record)
    return records


def _int_form(form: Any, key: str, default: int) -> int:
    try:
        return int(form.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _float_form(form: Any, key: str, default: float) -> float:
    try:
        return float(form.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _setup_estimate(cmva: CMVAApplication, data_accumulation: dict[str, Any]) -> dict[str, object]:
    candidate_specs = default_candidate_specs(
        groups=cmva.config.candidate_model_groups,
        target_view=cmva.config.target_view,
        limit=cmva.config.candidate_model_count,
    )
    rows = int(data_accumulation.get("total_rows") or 0)
    symbol_count = max(1, len(cmva.config.symbols))
    per_symbol_rows = max(0, rows // symbol_count)
    training_bars = bars_for_duration(cmva.config.training_window, cmva.config.interval)
    origins = generate_rolling_origins(
        per_symbol_rows,
        training_window_bars=training_bars,
        horizon_bars=cmva.config.forecast_horizon_bars,
        refit_stride_bars=cmva.config.refit_stride_bars,
    )
    stage1_fits = len(candidate_specs) * min(len(origins), max(8, min(40, len(origins) // 2 or len(origins))))
    stage2_candidates = max(3, min(6, len(candidate_specs) // 2 or len(candidate_specs)))
    stage2_fits = stage2_candidates * len(origins)
    estimated_total = stage1_fits + stage2_fits
    return {
        "symbols": len(cmva.config.symbols),
        "candidate_specs": len(candidate_specs),
        "rolling_origins": len(origins),
        "stage1_fits": stage1_fits,
        "stage2_fits": stage2_fits,
        "estimated_total_fits": estimated_total,
        "training_window_bars": training_bars,
        "forecast_horizon_bars": cmva.config.forecast_horizon_bars,
        "refit_stride_bars": cmva.config.refit_stride_bars,
        "estimated_runtime": f"{max(1, estimated_total) * 5} ms to {max(1, estimated_total) * 40} ms",
        "memory_estimate": f"{max(1, rows) * max(1, len(candidate_specs)) * 8 / 1024 / 1024:.2f} MB",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if hasattr(value, "isoformat") and value.__class__.__name__ in {"Timestamp", "datetime"}:
        return value.isoformat()
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
