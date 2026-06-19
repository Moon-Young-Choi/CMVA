"""Textual application for CMVA."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Static, TabbedContent, TabPane

from cmva.app import CMVAApplication
from cmva.time_ranges import normalize_time_range
from cmva.tui.bindings import BINDINGS
from cmva.tui.screens import (
    render_backtest,
    render_dashboard,
    render_data,
    render_features,
    render_logs,
    render_methodology,
    render_models,
    render_process,
    render_regime,
    render_settings,
    render_stat_tests,
)
from cmva.tui.theme import CSS


class CMVATuiApp(App):
    CSS = CSS
    BINDINGS: ClassVar = BINDINGS

    def __init__(self, cmva: CMVAApplication) -> None:
        super().__init__()
        self.cmva = cmva
        self._tasks: list[asyncio.Task] = []
        self._tab_ids = [
            "dashboard",
            "data",
            "features",
            "models",
            "methodology",
            "stat_tests",
            "regime",
            "backtest",
            "process",
            "settings",
            "logs",
        ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="tabs"):
            with TabPane("Dashboard", id="dashboard"):
                yield Static(id="dashboard_content", classes="panel")
            with TabPane("Data", id="data"):
                yield Static(id="data_content", classes="panel")
            with TabPane("Features", id="features"):
                yield Static(id="features_content", classes="panel")
            with TabPane("Models", id="models"):
                yield Button("Run model selection", id="models_select")
                yield Static(id="models_content", classes="panel")
            with TabPane("Methodology", id="methodology"):
                with VerticalScroll(classes="scroll_panel"):
                    yield Static(id="methodology_content", classes="panel")
            with TabPane("Stat Tests", id="stat_tests"):
                with VerticalScroll(classes="scroll_panel"):
                    yield Static(id="stat_tests_content", classes="panel")
            with TabPane("Regime", id="regime"):
                yield Static(id="regime_content", classes="panel")
            with TabPane("Backtest", id="backtest"):
                yield Static(id="backtest_content", classes="panel")
            with TabPane("Process", id="process"):
                with VerticalScroll(classes="scroll_panel"):
                    yield Static(id="process_content", classes="panel")
            with TabPane("Settings", id="settings"):
                with VerticalScroll(classes="scroll_panel", id="settings_scroll"):
                    yield Label("Symbols", classes="field_label")
                    yield Input(value=", ".join(self.cmva.config.symbols), id="settings_symbols")
                    yield Label("Primary interval", classes="field_label")
                    yield Input(value=self.cmva.config.interval, id="settings_interval")
                    yield Label("Forecast horizon bars", classes="field_label")
                    yield Input(value=str(self.cmva.config.forecast_horizon_bars), id="settings_horizon")
                    yield Label("Rolling windows: volatility, correlation, PCA, trend, regime threshold", classes="field_label")
                    yield Input(
                        value=(
                            f"{self.cmva.config.volatility_window}, {self.cmva.config.correlation_window}, "
                            f"{self.cmva.config.pca_window}, {self.cmva.config.trend_window}, "
                            f"{self.cmva.config.regime_threshold_window}"
                        ),
                        id="settings_windows",
                    )
                    yield Label("GARCH refit frequency, severe shock, moderate shock", classes="field_label")
                    yield Input(
                        value=f"{self.cmva.config.garch_refit_frequency}, {self.cmva.config.severe_shock_threshold}, {self.cmva.config.moderate_shock_threshold}",
                        id="settings_model",
                    )
                    yield Button("Apply from now", id="settings_apply_now")
                    yield Button("Recompute validation", id="settings_recompute")
                    yield Label("Dashboard, forecast, backtest ranges", classes="field_label")
                    yield Input(
                        value=f"{self.cmva.config.dashboard_time_range}, {self.cmva.config.forecast_time_range}, {self.cmva.config.backtest_time_range}",
                        id="settings_ranges",
                    )
                    yield Label("Preset ranges: 1d, 1w, 1m, 3m, 6m, 1y, all. Custom examples: 12h, 10d, 4w", classes="field_label")
                    yield Button("Apply view ranges", id="settings_apply_ranges")
                    yield Button("Recompute validation range", id="settings_recompute_range")
                    yield Button("Cancel", id="settings_cancel")
                    yield Static(id="settings_content", classes="panel")
            with TabPane("Logs", id="logs"):
                with VerticalScroll(classes="scroll_panel"):
                    yield Static(id="logs_content", classes="panel")
        yield Footer()

    async def on_mount(self) -> None:
        self.refresh_views()
        self.set_interval(1.0, self.refresh_views)
        self._tasks.append(asyncio.create_task(self._bootstrap()))

    async def on_unmount(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _bootstrap(self) -> None:
        await self.cmva.bootstrap(fetch_remote=True)
        self.refresh_views()
        if self.cmva.state.mode != "DEGRADED":
            self._tasks.append(asyncio.create_task(self._stream_live()))

    async def _stream_live(self) -> None:
        try:
            await self.cmva.stream_live()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.cmva.state.websocket_status = "error"
            self.cmva.state.mode = "DEGRADED"
            self.cmva.state.log(f"WebSocket stopped: {exc}")

    def refresh_views(self) -> None:
        widgets = {
            "dashboard_content": render_dashboard(self.cmva),
            "data_content": render_data(self.cmva),
            "features_content": render_features(self.cmva),
            "models_content": render_models(self.cmva),
            "methodology_content": render_methodology(self.cmva),
            "stat_tests_content": render_stat_tests(self.cmva),
            "regime_content": render_regime(self.cmva),
            "backtest_content": render_backtest(self.cmva),
            "process_content": render_process(self.cmva),
            "settings_content": render_settings(self.cmva),
            "logs_content": render_logs(self.cmva),
        }
        for widget_id, renderable in widgets.items():
            try:
                self.query_one(f"#{widget_id}", Static).update(renderable)
            except Exception:
                pass

    def action_previous_tab(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        active = tabs.active or self._tab_ids[0]
        idx = self._tab_ids.index(active) if active in self._tab_ids else 0
        tabs.active = self._tab_ids[(idx - 1) % len(self._tab_ids)]

    def action_next_tab(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        active = tabs.active or self._tab_ids[0]
        idx = self._tab_ids.index(active) if active in self._tab_ids else 0
        tabs.active = self._tab_ids[(idx + 1) % len(self._tab_ids)]

    def action_toggle_pause(self) -> None:
        self.cmva.toggle_pause()
        self.refresh_views()

    def action_refresh_now(self) -> None:
        if self.cmva.snapshot is not None:
            self.cmva.recompute(self.cmva.snapshot.candles, force_refit=False)
        self.cmva.state.log("Manual refresh")
        self.refresh_views()

    def action_force_refit(self) -> None:
        self.cmva.force_refit()
        self.refresh_views()

    def action_rerun_backtest(self) -> None:
        self.cmva.rerun_backtest()
        self.refresh_views()

    def action_quit(self) -> None:
        self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings_apply_now":
            self._apply_settings_from_form(recompute=False)
        elif event.button.id == "settings_recompute":
            self._apply_settings_from_form(recompute=True)
        elif event.button.id == "settings_cancel":
            self._reset_settings_form()
            self.cmva.cancel_settings()
            self.refresh_views()
        elif event.button.id == "models_select":
            self.cmva.run_model_selection()
            self.refresh_views()
        elif event.button.id == "settings_apply_ranges":
            self._apply_ranges_from_form(recompute_backtest=False)
        elif event.button.id == "settings_recompute_range":
            self._apply_ranges_from_form(recompute_backtest=True)

    def _apply_settings_from_form(self, recompute: bool) -> None:
        try:
            updates = self._settings_form_values()
        except ValueError as exc:
            self.cmva.state.log(f"Settings error: {exc}")
            self.refresh_views()
            return
        self.cmva.apply_settings(updates, recompute=recompute)
        self._reset_settings_form()
        self.refresh_views()

    def _apply_ranges_from_form(self, recompute_backtest: bool) -> None:
        try:
            updates = self._range_form_values()
        except ValueError as exc:
            self.cmva.state.log(f"Settings error: {exc}")
            self.refresh_views()
            return
        self.cmva.apply_view_ranges(updates, recompute_backtest=recompute_backtest)
        self._reset_settings_form()
        self.refresh_views()

    def _settings_form_values(self) -> dict[str, object]:
        symbols = [
            symbol.strip().upper()
            for symbol in self.query_one("#settings_symbols", Input).value.split(",")
            if symbol.strip()
        ]
        if not symbols:
            raise ValueError("at least one symbol is required")
        volatility_window, correlation_window, pca_window, trend_window, regime_threshold_window = _parse_strings(
            self.query_one("#settings_windows", Input).value,
            5,
            "duration windows",
        )
        refit_frequency, severe_shock, moderate_shock = _parse_mixed_model_settings(
            self.query_one("#settings_model", Input).value
        )
        horizon_bars = int(self.query_one("#settings_horizon", Input).value)
        if horizon_bars <= 0:
            raise ValueError("forecast horizon bars must be positive")
        return {
            "symbols": symbols,
            "interval": self.query_one("#settings_interval", Input).value.strip(),
            "forecast_horizon_bars": horizon_bars,
            "volatility_window": volatility_window,
            "correlation_window": correlation_window,
            "pca_window": pca_window,
            "trend_window": trend_window,
            "regime_threshold_window": regime_threshold_window,
            "garch_refit_frequency": refit_frequency,
            "severe_shock_threshold": severe_shock,
            "moderate_shock_threshold": moderate_shock,
        }

    def _range_form_values(self) -> dict[str, object]:
        dashboard_range, forecast_range, backtest_range = _parse_ranges(
            self.query_one("#settings_ranges", Input).value
        )
        return {
            "dashboard_time_range": dashboard_range,
            "forecast_time_range": forecast_range,
            "backtest_time_range": backtest_range,
        }

    def _reset_settings_form(self) -> None:
        self.query_one("#settings_symbols", Input).value = ", ".join(self.cmva.config.symbols)
        self.query_one("#settings_interval", Input).value = self.cmva.config.interval
        self.query_one("#settings_horizon", Input).value = str(self.cmva.config.forecast_horizon_bars)
        self.query_one("#settings_windows", Input).value = (
            f"{self.cmva.config.volatility_window}, {self.cmva.config.correlation_window}, "
            f"{self.cmva.config.pca_window}, {self.cmva.config.trend_window}, "
            f"{self.cmva.config.regime_threshold_window}"
        )
        self.query_one("#settings_model", Input).value = (
            f"{self.cmva.config.garch_refit_frequency}, {self.cmva.config.severe_shock_threshold}, {self.cmva.config.moderate_shock_threshold}"
        )
        self.query_one("#settings_ranges", Input).value = (
            f"{self.cmva.config.dashboard_time_range}, {self.cmva.config.forecast_time_range}, {self.cmva.config.backtest_time_range}"
        )


def _parse_strings(raw: str, expected: int, label: str) -> tuple[str, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{label} expects {expected} comma-separated values")
    return tuple(values)


def _parse_mixed_model_settings(raw: str) -> tuple[int, float, float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("model settings expects 3 comma-separated values")
    refit_frequency = int(values[0])
    severe_shock = float(values[1])
    moderate_shock = float(values[2])
    if refit_frequency <= 0 or severe_shock <= 0 or moderate_shock <= 0:
        raise ValueError("model settings values must be positive")
    return refit_frequency, severe_shock, moderate_shock


def _parse_ranges(raw: str) -> tuple[str, str, str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("range settings expects 3 comma-separated values")
    return tuple(normalize_time_range(value) for value in values)  # type: ignore[return-value]
