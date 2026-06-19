"""Textual application for CMVA."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.widgets import Button, Footer, Header, Input, Label, Static, TabbedContent, TabPane

from cmva.app import CMVAApplication
from cmva.tui.bindings import BINDINGS
from cmva.tui.screens import (
    render_backtest,
    render_dashboard,
    render_data,
    render_features,
    render_logs,
    render_models,
    render_regime,
    render_settings,
)
from cmva.tui.theme import CSS


class CMVATuiApp(App):
    CSS = CSS
    BINDINGS: ClassVar = BINDINGS

    def __init__(self, cmva: CMVAApplication) -> None:
        super().__init__()
        self.cmva = cmva
        self._tasks: list[asyncio.Task] = []
        self._tab_ids = ["dashboard", "data", "features", "models", "regime", "backtest", "settings", "logs"]

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
            with TabPane("Regime", id="regime"):
                yield Static(id="regime_content", classes="panel")
            with TabPane("Backtest", id="backtest"):
                yield Static(id="backtest_content", classes="panel")
            with TabPane("Settings", id="settings"):
                yield Label("Symbols", classes="field_label")
                yield Input(value=", ".join(self.cmva.config.symbols), id="settings_symbols")
                yield Label("Rolling windows: short, medium, long", classes="field_label")
                yield Input(
                    value=f"{self.cmva.config.rolling_short_window}, {self.cmva.config.rolling_medium_window}, {self.cmva.config.rolling_long_window}",
                    id="settings_windows",
                )
                yield Label("Target annual vol, max leverage", classes="field_label")
                yield Input(
                    value=f"{self.cmva.config.target_annual_vol}, {self.cmva.config.max_leverage}",
                    id="settings_risk",
                )
                yield Label("Cost bps, slippage bps", classes="field_label")
                yield Input(
                    value=f"{self.cmva.config.transaction_cost_bps}, {self.cmva.config.slippage_bps}",
                    id="settings_costs",
                )
                yield Label("GARCH refit frequency, severe shock, moderate shock", classes="field_label")
                yield Input(
                    value=f"{self.cmva.config.garch_refit_frequency}, {self.cmva.config.severe_shock_threshold}, {self.cmva.config.moderate_shock_threshold}",
                    id="settings_model",
                )
                yield Button("Apply from now", id="settings_apply_now")
                yield Button("Recompute historical backtest", id="settings_recompute")
                yield Button("Cancel", id="settings_cancel")
                yield Static(id="settings_content", classes="panel")
            with TabPane("Logs", id="logs"):
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
            "regime_content": render_regime(self.cmva),
            "backtest_content": render_backtest(self.cmva),
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

    def action_export_report(self) -> None:
        markdown_path, html_path = self.cmva.export_report()
        self.cmva.state.log(f"Report ready: {markdown_path}, {html_path}")
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

    def _settings_form_values(self) -> dict[str, object]:
        symbols = [
            symbol.strip().upper()
            for symbol in self.query_one("#settings_symbols", Input).value.split(",")
            if symbol.strip()
        ]
        if not symbols:
            raise ValueError("at least one symbol is required")
        short_window, medium_window, long_window = _parse_ints(
            self.query_one("#settings_windows", Input).value,
            3,
            "rolling windows",
        )
        target_annual_vol, max_leverage = _parse_floats(
            self.query_one("#settings_risk", Input).value,
            2,
            "risk settings",
        )
        cost_bps, slippage_bps = _parse_floats(
            self.query_one("#settings_costs", Input).value,
            2,
            "cost settings",
        )
        refit_frequency, severe_shock, moderate_shock = _parse_mixed_model_settings(
            self.query_one("#settings_model", Input).value
        )
        return {
            "symbols": symbols,
            "rolling_short_window": short_window,
            "rolling_medium_window": medium_window,
            "rolling_long_window": long_window,
            "target_annual_vol": target_annual_vol,
            "max_leverage": max_leverage,
            "transaction_cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
            "garch_refit_frequency": refit_frequency,
            "severe_shock_threshold": severe_shock,
            "moderate_shock_threshold": moderate_shock,
        }

    def _reset_settings_form(self) -> None:
        self.query_one("#settings_symbols", Input).value = ", ".join(self.cmva.config.symbols)
        self.query_one("#settings_windows", Input).value = (
            f"{self.cmva.config.rolling_short_window}, {self.cmva.config.rolling_medium_window}, {self.cmva.config.rolling_long_window}"
        )
        self.query_one("#settings_risk", Input).value = (
            f"{self.cmva.config.target_annual_vol}, {self.cmva.config.max_leverage}"
        )
        self.query_one("#settings_costs", Input).value = (
            f"{self.cmva.config.transaction_cost_bps}, {self.cmva.config.slippage_bps}"
        )
        self.query_one("#settings_model", Input).value = (
            f"{self.cmva.config.garch_refit_frequency}, {self.cmva.config.severe_shock_threshold}, {self.cmva.config.moderate_shock_threshold}"
        )


def _parse_ints(raw: str, expected: int, label: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{label} expects {expected} comma-separated values")
    parsed = tuple(int(value) for value in values)
    if any(value <= 0 for value in parsed):
        raise ValueError(f"{label} values must be positive")
    return parsed


def _parse_floats(raw: str, expected: int, label: str) -> tuple[float, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{label} expects {expected} comma-separated values")
    parsed = tuple(float(value) for value in values)
    if any(value < 0 for value in parsed):
        raise ValueError(f"{label} values must be nonnegative")
    return parsed


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
