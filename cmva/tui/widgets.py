"""Reusable TUI rendering helpers."""

from __future__ import annotations

from rich.table import Table

from cmva.app import CMVAApplication


def key_value_table(title: str, values: dict[str, object]) -> Table:
    table = Table(title=title, expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in values.items():
        table.add_row(str(key), _format_value(value))
    return table


def dashboard_table(app: CMVAApplication) -> Table:
    return key_value_table(
        "CMVA Dashboard",
        {
            "Mode": app.state.mode,
            "Latest closed candle": app.state.latest_closed_time,
            "Universe": ", ".join(app.config.symbols),
            "WebSocket": app.state.websocket_status,
            "Current regime": app.state.current_regime,
            "Shock status": app.state.current_shock_type,
            "Basket forecast vol 1h": _percent(app.state.forecast_vol_1h),
            "Target exposure next hour": _multiple(app.state.target_exposure),
            "Historical cumulative return": _percent(app.state.backtest_summary.get("backtest_cumulative_return")),
            "Historical max drawdown": _percent(app.state.backtest_summary.get("backtest_max_drawdown")),
            "Live paper PnL": _percent(app.state.live_paper_pnl),
        },
    )


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _percent(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _multiple(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return str(value)
