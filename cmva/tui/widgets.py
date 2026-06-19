"""Reusable TUI rendering helpers."""

from __future__ import annotations

import math

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cmva.analysis_types import DiagnosticSnapshot, MethodStep, StatTestResult
from cmva.app import CMVAApplication


def key_value_table(title: str, values: dict[str, object]) -> Table:
    table = Table(title=title, expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in values.items():
        table.add_row(str(key), _format_value(value))
    return table


def dashboard_table(app: CMVAApplication) -> Group:
    summary = key_value_table(
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
    trends = Table(title="Live Trends", expand=True)
    trends.add_column("Series", style="cyan", no_wrap=True)
    trends.add_column("Sparkline", style="white")
    trends.add_column("Latest", style="white", no_wrap=True)
    trend_specs = {
        "basket_return": "Basket return",
        "forecast_vol": "Forecast vol",
        "target_exposure": "Exposure",
        "regime_score": "Regime/shock score",
    }
    for key, label in trend_specs.items():
        values = app.state.trend_buffers.get(key, [])
        trends.add_row(label, _sparkline(values), _format_latest(values))
    status = Panel(
        Text.from_markup(
            f"[bold]Pipeline[/bold] {app.state.mode} | WebSocket {app.state.websocket_status} | "
            f"Last stat tests {app.state.last_stat_test_run or '-'}"
        ),
        title="Status",
        expand=True,
    )
    return Group(summary, trends, status)


def methodology_panel(app: CMVAApplication) -> Group:
    cutoff = app.state.latest_closed_time
    latest = {}
    if app.snapshot is not None and not app.snapshot.features.features.empty:
        latest = app.snapshot.features.features.iloc[-1].to_dict()
    formula_table = Table(title="Mathematical Methodology", expand=True)
    formula_table.add_column("Step", style="cyan", no_wrap=True)
    formula_table.add_column("Formula / rule", style="white")
    formula_table.add_column("Current value", style="white", no_wrap=True)
    formula_table.add_column("Cutoff", style="white", no_wrap=True)
    formula_table.add_row("Return", "r_t = log(C_t / C_{t-1})", _format_value(latest.get("basket_return")), str(cutoff or "-"))
    formula_table.add_row("Realized vol", "sqrt(mean(r_i^2)) over rolling window", _format_value(latest.get("market_vol")), str(cutoff or "-"))
    formula_table.add_row("EWMA vol", "sqrt(EWMA(r_t^2))", _format_value(latest.get("ewma_vol")), str(cutoff or "-"))
    formula_table.add_row("Avg corr", "mean(pairwise rolling corr)", _format_value(latest.get("avg_pairwise_corr")), str(cutoff or "-"))
    formula_table.add_row("BTC beta", "cov(r_i, r_BTC) / var(r_BTC)", _format_value(latest.get("avg_btc_beta")), str(cutoff or "-"))
    formula_table.add_row("PCA1 share", "max eigenvalue(cov) / sum eigenvalues(cov)", _format_value(latest.get("pca1_share")), str(cutoff or "-"))
    formula_table.add_row("GARCH", "r_t = mu + eps_t; h_t = omega + alpha eps^2 + beta h", _percent(app.state.forecast_vol_1h), str(cutoff or "-"))
    formula_table.add_row("Shock", "|r_t| / sigma_{t|t-1}", app.state.current_shock_type or "-", str(cutoff or "-"))
    formula_table.add_row("Regime", "expanding thresholds through t only", app.state.current_regime or "-", str(cutoff or "-"))
    formula_table.add_row("Exposure", "clip(target_vol / sigma * multiplier, 0, max_leverage)", _multiple(app.state.target_exposure), str(cutoff or "-"))
    formula_table.add_row("Backtest", "R_{t+1} = w_t r_{t+1} - c |w_t - w_{t-1}|", "shifted", str(cutoff or "-"))

    steps = process_table(app.state.latest_diagnostics.method_steps[-20:] or app.state.process_timeline[-20:])
    note = Panel(
        "All formulas use closed-candle data only. Forecast, shock, and backtest rows explicitly use shifted signals where required.",
        title="Look-ahead discipline",
        expand=True,
    )
    return Group(formula_table, note, steps)


def diagnostics_panel(snapshot: DiagnosticSnapshot) -> Group:
    return Group(
        _test_table("Model Diagnostics", snapshot.model_tests),
        _test_table("Forecast Evaluation", snapshot.forecast_tests),
        _test_table("Risk Coverage", snapshot.risk_tests),
        _test_table("Backtest Inference", snapshot.backtest_tests),
        _test_table("Regime / Shock Validation", snapshot.regime_tests),
    )


def process_table(steps: list[MethodStep]) -> Table:
    table = Table(title="Moment-by-moment Process", expand=True)
    table.add_column("Time", style="white", no_wrap=True)
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Formula / action", style="white")
    table.add_column("Output", style="white")
    table.add_column("Cutoff", style="white", no_wrap=True)
    table.add_column("Look-ahead", style="green", no_wrap=True)
    if not steps:
        table.add_row("-", "waiting", "no process events yet", "-", "-", "-")
        return table
    for step in steps[-40:]:
        table.add_row(
            str(step.timestamp or "-"),
            step.stage,
            step.formula_id,
            _format_value(step.output),
            str(step.data_cutoff or "-"),
            step.lookahead_status,
        )
    return table


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.6f}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items())
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


def _format_latest(values: list[float]) -> str:
    if not values:
        return "-"
    return _format_value(values[-1])


def _sparkline(values: list[float], width: int = 48) -> str:
    if not values:
        return "-"
    values = values[-width:]
    clean = [float(value) for value in values if not math.isnan(float(value)) and not math.isinf(float(value))]
    if not clean:
        return "-"
    low = min(clean)
    high = max(clean)
    chars = " .:-=+*#%@"
    if high == low:
        return chars[len(chars) // 2] * len(values)
    line = []
    for value in values:
        position = (float(value) - low) / (high - low)
        idx = max(0, min(len(chars) - 1, int(round(position * (len(chars) - 1)))))
        line.append(chars[idx])
    return "".join(line)


def _test_table(title: str, tests: list[StatTestResult]) -> Table:
    table = Table(title=title, expand=True)
    table.add_column("Test", style="cyan", no_wrap=True)
    table.add_column("Null / formula", style="white")
    table.add_column("Stat", style="white", no_wrap=True)
    table.add_column("p", style="white", no_wrap=True)
    table.add_column("Decision", style="white", no_wrap=True)
    table.add_column("n", style="white", no_wrap=True)
    table.add_column("Interpretation", style="white")
    if not tests:
        table.add_row("-", "no diagnostics yet", "-", "-", "-", "-", "-")
        return table
    for result in tests:
        table.add_row(
            result.name,
            f"{result.null_hypothesis}\n{result.formula}",
            _format_value(result.statistic),
            _format_pvalue(result.p_value),
            result.decision,
            str(result.sample_size),
            result.interpretation or result.limitations,
        )
    return table


def _format_pvalue(value: float | None) -> str:
    if value is None or pd_isna(value):
        return "-"
    return f"{float(value):.4f}"


def pd_isna(value: object) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False
