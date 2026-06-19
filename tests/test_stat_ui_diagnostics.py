from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from rich.console import Console
from textual.containers import VerticalScroll

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig
from cmva.models.diagnostics import run_statistical_diagnostics
from cmva.regime.shock import compute_shock_series
from cmva.tui.app import CMVATuiApp
from cmva.tui.screens import render_methodology, render_stat_tests
from cmva.validation import run_walk_forward_validation


def test_shock_score_uses_prior_forecast_only():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    returns = pd.DataFrame({"BTCUSDT": [0.0, 0.10, 0.10], "ETHUSDT": [0.0, 0.10, 0.10]}, index=index)
    forecast = pd.Series([0.05, 1000.0, 1000.0], index=index)
    avg_corr = pd.Series([0.0, 0.0, 0.0], index=index)
    dispersion = pd.Series([0.0, 0.0, 0.0], index=index)

    shocks = compute_shock_series(returns, forecast, avg_corr, dispersion, window=2)

    assert shocks["shock_score"].iloc[1] == pytest.approx(2.0)


def test_statistical_diagnostics_are_structured_and_graceful():
    index = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    returns = pd.Series([0.001 * ((idx % 5) - 2) for idx in range(80)], index=index)
    forecast = pd.Series(0.003, index=index)
    ewma = pd.Series(0.004, index=index)
    regimes = pd.Series(["QUIET_CORRELATED"] * 80, index=index)
    shocks = pd.DataFrame({"shock_type": ["NORMAL"] * 80}, index=index)
    validation = run_walk_forward_validation(returns, forecast, ewma, forecast, regimes)

    diagnostics = run_statistical_diagnostics(
        returns,
        forecast,
        ewma,
        regimes,
        shocks,
        validation.losses,
        garch_params={"alpha[1]": 0.05, "beta[1]": 0.90, "nu": 8.0},
    )

    assert diagnostics.model_tests
    assert diagnostics.forecast_tests
    assert diagnostics.backtest_tests
    assert any(result.p_value is not None for result in diagnostics.model_tests + diagnostics.risk_tests)
    assert all(result.sample_size >= 0 for result in diagnostics.all_tests)


def test_validation_diagnostics_are_deterministic():
    index = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    returns = pd.Series([0.001, -0.0005, 0.0002, 0.0007] * 20, index=index)
    forecast = pd.Series(0.003, index=index)
    regimes = pd.Series(["ASSET_SPECIFIC"] * 80, index=index)
    shocks = pd.DataFrame({"shock_type": ["NORMAL"] * 80}, index=index)
    validation = run_walk_forward_validation(returns, forecast, forecast, forecast, regimes)

    first = run_statistical_diagnostics(returns, forecast, forecast, regimes, shocks, validation.losses)
    second = run_statistical_diagnostics(returns, forecast, forecast, regimes, shocks, validation.losses)
    first_stats = [(result.name, result.statistic, result.p_value) for result in first.backtest_tests]
    second_stats = [(result.name, result.statistic, result.p_value) for result in second.backtest_tests]

    assert first_stats == second_stats


def test_methodology_and_stat_tabs_render_math_and_pvalues(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(interval="1h", data_dir=tmp_path / "data"))
    app.recompute(synthetic_candles(periods=80), force_refit=True)
    console = Console(record=True, width=160)

    console.print(render_methodology(app))
    methodology = console.export_text()
    assert "r_t = log(C_t / C_{t-1})" in methodology
    assert "sigma_{t|t-1}" in methodology

    console = Console(record=True, width=160)
    console.print(render_stat_tests(app))
    stat_tests = console.export_text()
    assert "Model Diagnostics" in stat_tests
    assert "p" in stat_tests
    assert "Decision" in stat_tests


def test_settings_tab_is_wrapped_in_vertical_scroll(tmp_path):
    app = CMVATuiApp(CMVAApplication(CMVAConfig(data_dir=tmp_path / "data")))

    async def noop_bootstrap() -> None:
        return None

    app._bootstrap = noop_bootstrap  # type: ignore[method-assign]

    async def run() -> None:
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#settings_scroll", VerticalScroll)

    asyncio.run(run())

