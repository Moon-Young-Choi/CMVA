"""TUI tab content builders."""

from __future__ import annotations

from rich.table import Table

from cmva.app import CMVAApplication
from cmva.tui.widgets import backtest_panel, dashboard_table, diagnostics_panel, key_value_table, methodology_panel, process_table


def render_dashboard(app: CMVAApplication) -> Table:
    return dashboard_table(app)


def render_data(app: CMVAApplication) -> Table:
    return key_value_table("Data", app.state.data_status)


def render_features(app: CMVAApplication) -> Table:
    if app.snapshot is None or app.snapshot.features.features.empty:
        return key_value_table("Features", {"status": "no features"})
    latest = app.snapshot.features.features.iloc[-1].to_dict()
    return key_value_table("Latest Features", latest)


def render_models(app: CMVAApplication) -> Table:
    return key_value_table("Models", app.state.model_status)


def render_methodology(app: CMVAApplication):
    return methodology_panel(app)


def render_stat_tests(app: CMVAApplication):
    return diagnostics_panel(app.state.latest_diagnostics, app.state.range_status)


def render_regime(app: CMVAApplication) -> Table:
    if app.snapshot is None:
        return key_value_table("Regime", {"current": app.state.current_regime})
    values = {"current_regime": app.state.current_regime, "current_shock": app.state.current_shock_type}
    if not app.snapshot.shocks.empty:
        values.update(app.snapshot.shocks.iloc[-1].to_dict())
    return key_value_table("Regime", values)


def render_backtest(app: CMVAApplication) -> Table:
    return backtest_panel(app)


def render_process(app: CMVAApplication) -> Table:
    return process_table(app.state.process_timeline)


def render_settings(app: CMVAApplication) -> Table:
    return key_value_table(
        "Settings",
        {
            "symbols": ", ".join(app.config.symbols),
            "interval": app.config.interval,
            "forecast_horizon": app.config.forecast_horizon,
            "historical_days": app.config.historical_days,
            "dashboard_time_range": app.config.dashboard_time_range,
            "forecast_time_range": app.config.forecast_time_range,
            "backtest_time_range": app.config.backtest_time_range,
            "allowed_time_ranges": ", ".join(app.config.allowed_time_ranges),
            "duration_windows": (
                f"vol={app.config.volatility_window}, corr={app.config.correlation_window}, "
                f"pca={app.config.pca_window}, trend={app.config.trend_window}, "
                f"regime={app.config.regime_threshold_window}"
            ),
            "window_bar_counts": app.config.window_bar_counts,
            "garch_refit_frequency": app.config.garch_refit_frequency,
        },
    )


def render_logs(app: CMVAApplication) -> str:
    return "\n".join(app.state.logs[-40:]) or "No logs yet."
