from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from rich.console import Console
from textual.containers import VerticalScroll

from cmva.analysis_types import DiagnosticSnapshot, StatTestResult
from cmva.app import CMVAApplication
from cmva.backtest.engine import run_historical_backtest
from cmva.config import CMVAConfig
from cmva.models.diagnostics import run_statistical_diagnostics
from cmva.regime.shock import compute_shock_series
from cmva.reports.markdown import build_markdown_report
from cmva.tui.app import CMVATuiApp
from cmva.tui.screens import render_methodology, render_stat_tests


def test_shock_score_uses_prior_forecast_only():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    returns = pd.DataFrame({"BTCUSDT": [0.0, 0.10, 0.10], "ETHUSDT": [0.0, 0.10, 0.10]}, index=index)
    forecast = pd.Series([0.05, 1000.0, 1000.0], index=index)
    avg_corr = pd.Series([0.0, 0.0, 0.0], index=index)
    dispersion = pd.Series([0.0, 0.0, 0.0], index=index)

    shocks = compute_shock_series(returns, forecast, avg_corr, dispersion, window=2)

    assert shocks["shock_score"].iloc[1] == pytest.approx(2.0)


def test_backtest_costs_are_shifted_with_applied_exposure():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    returns = pd.DataFrame({"BTCUSDT": [0.0, 0.02, 0.0], "ETHUSDT": [0.0, 0.02, 0.0]}, index=index)
    forecast = pd.Series([0.001, 0.001, 0.001], index=index)
    regimes = pd.Series(["ASSET_SPECIFIC"] * 3, index=index)

    result = run_historical_backtest(returns, forecast, regimes, 0.01, 1.0, 50.0, 50.0)
    strategy = result.returns["regime_aware_vol_target"]

    assert strategy.iloc[0] == pytest.approx(0.0)
    assert strategy.iloc[1] == pytest.approx(0.01)


def test_statistical_diagnostics_are_structured_and_graceful():
    index = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    returns = pd.Series([0.001 * ((idx % 5) - 2) for idx in range(80)], index=index)
    forecast = pd.Series(0.003, index=index)
    ewma = pd.Series(0.004, index=index)
    regimes = pd.Series(["QUIET_CORRELATED"] * 80, index=index)
    shocks = pd.DataFrame({"shock_type": ["NORMAL"] * 80}, index=index)
    backtest_returns = pd.DataFrame(
        {
            "regime_aware_vol_target": returns * 0.5,
            "equal_weight_buy_hold": returns,
        },
        index=index,
    )

    diagnostics = run_statistical_diagnostics(
        returns,
        forecast,
        ewma,
        regimes,
        shocks,
        backtest_returns,
        garch_params={"alpha[1]": 0.05, "beta[1]": 0.90, "nu": 8.0},
    )

    assert diagnostics.model_tests
    assert diagnostics.forecast_tests
    assert diagnostics.backtest_tests
    assert any(result.p_value is not None for result in diagnostics.model_tests + diagnostics.risk_tests)
    assert all(result.sample_size >= 0 for result in diagnostics.all_tests)


def test_bootstrap_diagnostics_are_deterministic():
    index = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    returns = pd.Series([0.001, -0.0005, 0.0002, 0.0007] * 20, index=index)
    forecast = pd.Series(0.003, index=index)
    regimes = pd.Series(["ASSET_SPECIFIC"] * 80, index=index)
    shocks = pd.DataFrame({"shock_type": ["NORMAL"] * 80}, index=index)
    backtest_returns = pd.DataFrame({"regime_aware_vol_target": returns}, index=index)

    first = run_statistical_diagnostics(returns, forecast, forecast, regimes, shocks, backtest_returns)
    second = run_statistical_diagnostics(returns, forecast, forecast, regimes, shocks, backtest_returns)
    first_ci = [result.interpretation for result in first.backtest_tests if result.name == "Bootstrap Sharpe CI"][0]
    second_ci = [result.interpretation for result in second.backtest_tests if result.name == "Bootstrap Sharpe CI"][0]

    assert first_ci == second_ci


def test_methodology_and_stat_tabs_render_math_and_pvalues(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
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
    app = CMVATuiApp(CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports")))

    async def noop_bootstrap() -> None:
        return None

    app._bootstrap = noop_bootstrap  # type: ignore[method-assign]

    async def run() -> None:
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#settings_scroll", VerticalScroll)

    asyncio.run(run())


def test_report_contains_math_diagnostics_and_limitations():
    diagnostics = DiagnosticSnapshot(
        model_tests=[
            StatTestResult(
                name="Example test",
                null_hypothesis="no issue",
                formula="Q = n(n+2) sum",
                statistic=1.23,
                p_value=0.45,
                decision="pass diagnostic",
                sample_size=100,
                interpretation="example",
            )
        ]
    )

    markdown = build_markdown_report({"mode": "LIVE"}, diagnostics=diagnostics)

    assert "Mathematical methodology" in markdown
    assert "Statistical diagnostics" in markdown
    assert "p_value" in markdown
    assert "Signals are shifted by one period" in markdown
