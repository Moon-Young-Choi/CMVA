from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from rich.console import Console
from textual.containers import VerticalScroll
from textual.widgets import Input

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig
from cmva.reports.markdown import build_markdown_report
from cmva.time_ranges import parse_time_range, slice_by_time_range
from cmva.tui.app import CMVATuiApp, _parse_ranges
from cmva.tui.graphs import time_series_panel
from cmva.tui.screens import render_backtest, render_dashboard, render_stat_tests


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


def test_graph_renderer_handles_common_series_shapes():
    index = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
    cases = [
        pd.Series([1, 2, 3, 2, 5, 4, 3, 6], index=index),
        pd.Series([1] * 8, index=index),
        pd.Series([None, None, 1, None, 2, None, 3, None], index=index),
        pd.Series([-3, -2, -1, 0, 1, 0, -1, 2], index=index),
        pd.Series([1], index=index[:1]),
    ]
    console = Console(record=True, width=120)
    for pos, series in enumerate(cases):
        console.print(time_series_panel(series, f"case {pos}", "all"))

    text = console.export_text()
    assert "case 0" in text
    assert "latest=" in text


def test_default_range_config_values():
    config = CMVAConfig()

    assert config.interval == "1h"
    assert config.forecast_horizon == "1h"
    assert config.dashboard_time_range == "1d"
    assert config.forecast_time_range == "1w"
    assert config.backtest_time_range == "1y"


def test_range_settings_change_backtest_sample_size(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
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
    assert app.range_backtest is not None
    assert len(app.range_backtest.returns) == 24


def test_forecast_diagnostics_use_selected_forecast_range(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
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


def test_dashboard_and_backtest_render_range_labels_and_graphs(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
    app.recompute(synthetic_candles(periods=240), force_refit=True)
    console = Console(record=True, width=160)

    console.print(render_dashboard(app))
    dashboard = console.export_text()
    assert "Dashboard range" in dashboard
    assert "Forecast horizon" in dashboard
    assert "Basket Return" in dashboard

    console = Console(record=True, width=160)
    console.print(render_backtest(app))
    backtest = console.export_text()
    assert "Equity Curve" in backtest
    assert "Drawdown" in backtest
    assert "backtest_observations" in backtest


def test_stat_tests_render_active_forecast_range(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
    app.recompute(synthetic_candles(periods=240), force_refit=True)
    console = Console(record=True, width=160)

    console.print(render_stat_tests(app))
    text = console.export_text()

    assert "Active Diagnostic Window" in text
    assert "Forecast diagnostics range=1W" in text


def test_settings_tab_exposes_range_controls(tmp_path):
    app = CMVATuiApp(CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports")))

    async def noop_bootstrap() -> None:
        return None

    app._bootstrap = noop_bootstrap  # type: ignore[method-assign]

    async def run() -> None:
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#settings_scroll", VerticalScroll)
            ranges = pilot.app.query_one("#settings_ranges", Input)
            assert ranges.value == "1d, 1w, 1y"

    asyncio.run(run())


def test_range_form_parser_rejects_invalid_values():
    assert _parse_ranges("1d, 1w, all") == ("1d", "1w", "all")
    with pytest.raises(ValueError):
        _parse_ranges("1d, soon, all")


def test_report_contains_selected_ranges_and_horizon_policy():
    markdown = build_markdown_report(
        {
            "mode": "LIVE",
            "forecast_horizon": "1h",
            "dashboard_time_range": "1d",
            "forecast_time_range": "1w",
            "backtest_time_range": "1y",
        }
    )

    assert "Horizon and range policy" in markdown
    assert "`forecast_horizon`: 1h" in markdown
    assert "`backtest_time_range`: 1y" in markdown
