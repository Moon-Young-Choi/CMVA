from __future__ import annotations

import importlib.util
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig
from cmva.time_ranges import parse_time_range, slice_by_time_range
from cmva.web.app import create_web_app


@pytest.mark.parametrize(
    ("label", "hours"),
    [
        ("1d", 24),
        ("1w", 168),
        ("1m", 30 * 24),
        ("3m", 90 * 24),
        ("6m", 180 * 24),
        ("1y", 365 * 24),
        ("12h", 12),
        ("10d", 240),
        ("all", None),
    ],
)
def test_time_range_parser(label, hours):
    parsed = parse_time_range(label)

    assert parsed.normalized == label
    assert parsed.hours == hours


def test_time_range_parser_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_time_range("soon")


def test_one_day_slice_returns_latest_24_hourly_rows():
    index = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    series = pd.Series(range(48), index=index)

    sliced, metadata = slice_by_time_range(series, "1d")

    assert len(sliced) == 24
    assert sliced.index[0] == index[-24]
    assert metadata.expected_points == 24
    assert metadata.actual_points == 24


def test_default_range_config_values():
    config = CMVAConfig()

    assert config.interval == "15m"
    assert config.forecast_horizon == "1 bar = next 15 minutes"
    assert config.dashboard_time_range == "1d"
    assert config.forecast_time_range == "1w"
    assert config.backtest_time_range == "1y"
    assert config.training_window == "30d"
    assert config.search_mode == "two_stage"


def test_range_settings_change_backtest_sample_size(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(interval="1h", data_dir=tmp_path / "data"))
    app.recompute(synthetic_candles(periods=240), force_refit=True)
    full_count = app.state.backtest_summary["backtest_observations"]

    app.apply_view_ranges(
        {
            "dashboard_time_range": "1d",
            "forecast_time_range": "1d",
            "backtest_time_range": "1d",
        },
        recompute_backtest=True,
    )

    assert full_count == 240
    assert app.state.backtest_summary["backtest_observations"] == 24
    assert app.range_model_validation is not None
    assert app.range_model_validation.sample_size <= 24


def test_forecast_diagnostics_use_selected_forecast_range(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(interval="1h", data_dir=tmp_path / "data"))
    app.recompute(synthetic_candles(periods=240), force_refit=True)
    app.apply_view_ranges(
        {
            "dashboard_time_range": "1d",
            "forecast_time_range": "1d",
            "backtest_time_range": "1w",
        }
    )

    forecast_status = app.state.range_status["forecast"]
    assert forecast_status["actual_points"] == 24
    assert app.state.latest_diagnostics.forecast_tests


def test_new_web_pages_render_range_and_model_lab_surfaces(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(interval="1h", data_dir=tmp_path / "data"))
    app.recompute(synthetic_candles(periods=120), force_refit=True)
    client = TestClient(create_web_app(app, start_background=False))

    pages = {
        "/setup": "초기 설정",
        "/overview": "개요",
        "/data-quality": "데이터 품질",
        "/model-lab": "모델 실험실",
        "/current-market-state": "현재 시장 상태",
        "/trend-seasonality": "추세 및 계절성",
        "/diagnostics": "진단",
        "/rolling-evaluation": "롤링 평가",
        "/model-comparison": "모델 비교",
    }

    for path, heading in pages.items():
        response = client.get(path)
        assert response.status_code == 200
        assert f"<h1>{heading}</h1>" in response.text


def test_legacy_tui_paths_are_removed():
    assert importlib.util.find_spec("cmva.tui") is None
    result = subprocess.run(
        [sys.executable, "-m", "cmva", "--tui"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --tui" in result.stderr
