from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient
import pandas as pd

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig
from cmva.data.candle import Candle
from cmva.web.app import create_web_app


WEB_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_web_routes_load_with_synthetic_snapshot(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))

    for path in [
        "/",
        "/markets",
        "/volatility",
        "/trend",
        "/correlation-pca",
        "/shock-regime",
        "/models",
        "/validation",
        "/methodology",
        "/settings",
        "/logs",
    ]:
        response = client.get(path)
        assert response.status_code == 200


def test_web_routes_render_distinct_pages(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))
    expected_headings = {
        "/": "Dashboard",
        "/markets": "Markets",
        "/volatility": "Volatility",
        "/trend": "Trend",
        "/correlation-pca": "Correlation / PCA",
        "/shock-regime": "Shock & Regime",
        "/models": "Models",
        "/validation": "Backtest / Validation",
        "/methodology": "Methodology",
        "/settings": "Settings",
        "/logs": "Logs",
    }

    for path, heading in expected_headings.items():
        response = client.get(path)
        assert f"<h1>{heading}</h1>" in response.text


def test_snapshot_api_is_json_serializable(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))

    response = client.get("/api/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["interval"] == "1h"
    assert "latest_features" in payload
    assert payload["data_accumulation"]["total_rows"] == 120 * 3
    assert len(payload["data_accumulation"]["symbols"]) == 3
    assert "bootstrap_progress" in payload
    assert "data_quality" in payload
    assert "pipeline" in payload
    assert "standardized_residual" in payload["series"]
    assert "regime_state" in payload["series"]
    assert payload["data_quality"]["status"] == "ok"
    assert payload["forming_candle"] is None


def test_markets_page_exposes_data_accumulation(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))

    response = client.get("/markets")

    assert response.status_code == 200
    assert "Accumulation By Symbol" in response.text
    assert "Forming Candle" in response.text
    assert "Data Quality Issues" in response.text
    assert "Research Dataset" in response.text
    assert "Data To Validation Flow" in response.text
    assert "Recent Accumulated Candles" in response.text
    assert "BTCUSDT" in response.text


def test_snapshot_exposes_forming_candle_separately(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    cmva.process_current_candle(
        Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=pd.Timestamp("2026-01-06 00:00:00+00:00"),
            close_time=pd.Timestamp("2026-01-06 00:59:59+00:00"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
            is_closed=False,
        )
    )
    client = TestClient(create_web_app(cmva, start_background=False))

    payload = client.get("/api/snapshot").json()

    assert payload["forming_candle"]["symbol"] == "BTCUSDT"
    assert payload["data_policy"]["forming_candle_usage"] == "display only"


def test_live_script_and_websocket_snapshot_are_available(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))

    response = client.get("/")
    assert "/static/js/live.js" in response.text

    with client.websocket_connect("/ws/snapshot") as websocket:
        payload = websocket.receive_json()

    assert payload["summary"]["interval"] == "1h"


def test_routes_respond_while_background_bootstrap_is_running(tmp_path):
    class SlowBootstrapApplication(CMVAApplication):
        async def bootstrap(self, fetch_remote: bool = True) -> None:
            self.state.mode = "BOOTSTRAP"
            await asyncio.sleep(1.0)
            self.state.mode = "DEGRADED"

    cmva = SlowBootstrapApplication(CMVAConfig(data_dir=tmp_path / "data"))

    with TestClient(create_web_app(cmva, start_background=True)) as client:
        started = time.perf_counter()
        response = client.get("/validation")
        elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 0.5
