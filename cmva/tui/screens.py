"""TUI tab content builders."""

from __future__ import annotations

from rich.table import Table

from cmva.app import CMVAApplication
from cmva.tui.widgets import dashboard_table, key_value_table


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


def render_regime(app: CMVAApplication) -> Table:
    if app.snapshot is None:
        return key_value_table("Regime", {"current": app.state.current_regime})
    values = {"current_regime": app.state.current_regime, "current_shock": app.state.current_shock_type}
    if not app.snapshot.shocks.empty:
        values.update(app.snapshot.shocks.iloc[-1].to_dict())
    return key_value_table("Regime", values)


def render_backtest(app: CMVAApplication) -> Table:
    values = dict(app.state.backtest_summary)
    if app.snapshot and app.snapshot.backtest:
        values["average_exposure_by_regime"] = app.snapshot.backtest.average_exposure_by_regime
        values["return_by_regime"] = app.snapshot.backtest.return_by_regime
    return key_value_table("Backtest", values or {"status": "no backtest"})


def render_settings(app: CMVAApplication) -> Table:
    return key_value_table(
        "Settings",
        {
            "symbols": ", ".join(app.config.symbols),
            "interval": app.config.interval,
            "historical_days": app.config.historical_days,
            "rolling_windows": f"{app.config.rolling_short_window}, {app.config.rolling_medium_window}, {app.config.rolling_long_window}",
            "target_annual_vol": app.config.target_annual_vol,
            "max_leverage": app.config.max_leverage,
            "transaction_cost_bps": app.config.transaction_cost_bps,
            "slippage_bps": app.config.slippage_bps,
            "garch_refit_frequency": app.config.garch_refit_frequency,
        },
    )


def render_logs(app: CMVAApplication) -> str:
    return "\n".join(app.state.logs[-40:]) or "No logs yet."
