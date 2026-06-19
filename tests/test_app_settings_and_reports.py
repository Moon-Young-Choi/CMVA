from __future__ import annotations

import pandas as pd

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig
from cmva.reports.plots import export_equity_svg, plots_available


def test_apply_settings_updates_runtime_config(tmp_path):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
    app.apply_settings(
        {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "target_annual_vol": 0.15,
            "max_leverage": 1.2,
            "transaction_cost_bps": 3.0,
            "slippage_bps": 1.0,
            "garch_refit_frequency": 12,
        }
    )

    assert app.config.symbols == ["BTCUSDT", "ETHUSDT"]
    assert app.config.target_annual_vol == 0.15
    assert app.forecaster.refit_frequency == 12
    assert app.paper.transaction_cost_bps == 3.0
    assert app.paper.slippage_bps == 1.0


def test_model_selection_records_garch_status(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"))
    snapshot = app.recompute(synthetic_candles(periods=40), force_refit=True)
    result = app.run_model_selection()

    assert snapshot is app.snapshot
    assert result is not None
    assert result.selected_model == "garch"
    assert app.state.model_status["selected_model"] == "garch"


def test_report_equity_svg_export(tmp_path):
    equity = pd.DataFrame(
        {"regime_aware_vol_target": [1.0, 1.02, 0.99, 1.05]},
        index=pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
    )
    path = export_equity_svg(equity, tmp_path / "equity.svg")

    assert plots_available()
    assert path is not None
    assert path.exists()
    assert "<polyline" in path.read_text(encoding="utf-8")
