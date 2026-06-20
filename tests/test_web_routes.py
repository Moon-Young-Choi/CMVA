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
        "/simulation/new",
        "/simulations",
        "/setup",
        "/overview",
        "/data-quality",
        "/model-lab",
        "/current-market-state",
        "/trend-seasonality",
        "/diagnostics",
        "/rolling-evaluation",
        "/model-comparison",
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
        "/": "새 시뮬레이션",
        "/simulation/new": "새 시뮬레이션",
        "/simulations": "실행 중인 시뮬레이션",
        "/setup": "초기 설정",
        "/overview": "개요",
        "/data-quality": "데이터 품질",
        "/model-lab": "모델 실험실",
        "/current-market-state": "현재 시장 상태",
        "/trend-seasonality": "추세 및 계절성",
        "/diagnostics": "진단",
        "/rolling-evaluation": "롤링 평가",
        "/model-comparison": "모델 비교",
        "/markets": "데이터 품질",
        "/volatility": "변동성",
        "/trend": "추세 및 계절성",
        "/correlation-pca": "상관 / PCA",
        "/shock-regime": "쇼크 & 레짐",
        "/models": "모델 실험실",
        "/validation": "롤링 평가",
        "/methodology": "방법론",
        "/settings": "고급 설정",
        "/logs": "로그",
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
    assert "setup_estimate" in payload
    assert "model_lab" in payload
    assert "cpp_backend" in payload
    assert "standardized_residual" in payload["series"]
    assert "regime_state" in payload["series"]
    assert "log_return" in payload["series"]
    assert "log_price" in payload["series"]
    assert set(payload["model_lab"]["job_status"]["evaluated_targets"]) == {"log_return", "log_price"}
    assert payload["data_quality"]["status"] == "ok"
    assert payload["forming_candle"] is None


def test_snapshot_exposes_inflight_model_lab_status_before_results(tmp_path):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva._set_model_lab_progress(
        {
            "status": "stage1_running",
            "queue_size": 12,
            "candidate_count": 8,
            "stage1_fits": 2,
            "stage2_fits": 0,
            "active_stage": "stage1",
            "active_target": "log_return",
            "active_candidate": "ar_1:log_return",
            "completed_fits": 10,
            "total_fits": 80,
            "progress_pct": 0.125,
        }
    )
    client = TestClient(create_web_app(cmva, start_background=False))

    payload = client.get("/api/snapshot").json()

    assert payload["model_lab"]["job_status"]["status"] == "stage1_running"
    assert payload["model_lab"]["job_status"]["active_candidate"] == "ar_1:log_return"
    assert payload["model_lab"]["job_status"]["progress_pct"] == 0.125


def test_markets_page_exposes_data_accumulation(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))

    response = client.get("/data-quality")

    assert response.status_code == 200
    assert "심볼별 누적 현황" in response.text
    assert "형성 중 캔들" in response.text
    assert "데이터 품질 이슈" in response.text
    assert "연구 데이터셋" in response.text
    assert "데이터-검증 흐름" in response.text
    assert "최근 누적 캔들" in response.text
    assert "BTCUSDT" in response.text


def test_model_evaluation_pages_expose_filter_controls(tmp_path, synthetic_candles):
    cmva = CMVAApplication(CMVAConfig(symbols=WEB_SYMBOLS, interval="1h", data_dir=tmp_path / "data"))
    cmva.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(cmva, start_background=False))

    rolling = client.get("/rolling-evaluation")
    comparison = client.get("/model-comparison")

    assert "평가 필터" in rolling.text
    assert "name=\"metric\"" in rolling.text
    assert "data-table-filter=\"#rolling-timeline-table\"" in rolling.text
    assert "리더보드 필터" in comparison.text
    assert "data-table-filter=\"#model-comparison-table\"" in comparison.text


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
    assert "echarts.min.js" in response.text
    assert "/static/js/charts.js" in response.text

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
        response = client.get("/rolling-evaluation")
        elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 0.5
