from __future__ import annotations

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig


def test_apply_settings_updates_runtime_config(tmp_path):
    app = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data"))
    app.apply_settings(
        {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "interval": "5m",
            "forecast_horizon_bars": 3,
            "volatility_window": "24h",
            "correlation_window": "7d",
            "pca_window": "30d",
            "trend_window": "24h",
            "regime_threshold_window": "90d",
            "garch_refit_frequency": 12,
        }
    )

    assert app.config.symbols == ["BTCUSDT", "ETHUSDT"]
    assert app.config.interval == "5m"
    assert app.config.forecast_horizon == "3 bars = next 15 minutes"
    assert app.config.window_bar_counts["volatility_window"] == 288
    assert app.forecaster.refit_frequency == 12


def test_model_selection_records_garch_status(tmp_path, synthetic_candles):
    app = CMVAApplication(CMVAConfig(interval="1h", data_dir=tmp_path / "data"))
    snapshot = app.recompute(synthetic_candles(periods=40), force_refit=True)
    result = app.run_model_selection()

    assert snapshot is app.snapshot
    assert result is not None
    assert result.selected_model == "garch"
    assert app.state.model_status["selected_model"] == "garch"

