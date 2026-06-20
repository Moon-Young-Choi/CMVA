"""Configuration loading for CMVA."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cmva.engine.interval import (
    bars_for_duration,
    describe_horizon,
    interval_to_timedelta,
    normalize_interval,
    periods_per_year,
)
from cmva.time_ranges import DEFAULT_ALLOWED_TIME_RANGES, normalize_time_range

@dataclass
class CMVAConfig:
    symbols: list[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    )
    interval: str = "1h"
    analysis_period: str = "1y"
    training_window: str = "30d"
    forecast_horizon_bars: int = 1
    forecast_horizon: str = "1 bar"
    refit_stride_bars: int = 1
    search_mode: str = "two_stage"
    candidate_model_count: int = 16
    candidate_model_groups: list[str] = field(
        default_factory=lambda: ["mean", "trend", "volatility", "combined"]
    )
    target_view: str = "both"
    historical_days: int = 365
    bootstrap_limit: int = 1000
    dashboard_time_range: str = "1d"
    forecast_time_range: str = "1w"
    backtest_time_range: str = "1y"
    allowed_time_ranges: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_TIME_RANGES))
    volatility_window: str = "24h"
    correlation_window: str = "7d"
    pca_window: str = "30d"
    trend_window: str = "24h"
    regime_threshold_window: str = "90d"
    rolling_short_window: int = 24
    rolling_medium_window: int = 168
    rolling_long_window: int = 720
    min_threshold_history: int = 720
    garch_refit_frequency: int = 672
    severe_shock_threshold: float = 3.0
    moderate_shock_threshold: float = 2.0
    shock_breadth_threshold: float = 0.60
    use_cpp: bool = True
    data_dir: Path = Path("data")
    rest_base_url: str = "https://data-api.binance.vision"
    websocket_base_url: str = "wss://data-stream.binance.vision"

    def __post_init__(self) -> None:
        symbols = [symbol.upper().strip() for symbol in self.symbols if symbol.strip()]
        if not symbols:
            raise ValueError("at least one symbol is required")
        self.symbols = symbols
        self.interval = normalize_interval(self.interval)
        self.analysis_period = normalize_time_range(self.analysis_period)
        self.forecast_horizon_bars = int(self.forecast_horizon_bars)
        if self.forecast_horizon_bars <= 0:
            raise ValueError("forecast_horizon_bars must be positive")
        self.forecast_horizon = describe_horizon(self.interval, self.forecast_horizon_bars)
        self.refit_stride_bars = max(1, int(self.refit_stride_bars))
        self.search_mode = str(self.search_mode).strip().lower().replace("-", "_")
        if self.search_mode not in {"two_stage", "fast", "detailed"}:
            raise ValueError("search_mode must be one of: two_stage, fast, detailed")
        self.candidate_model_count = max(1, int(self.candidate_model_count))
        self.candidate_model_groups = [
            str(group).strip().lower()
            for group in self.candidate_model_groups
            if str(group).strip()
        ]
        if not self.candidate_model_groups:
            self.candidate_model_groups = ["mean", "trend", "volatility", "combined"]
        self.target_view = str(self.target_view).strip().lower()
        if self.target_view not in {"log_price", "log_return", "both"}:
            raise ValueError("target_view must be one of: log_price, log_return, both")
        self.rolling_short_window = bars_for_duration(self.volatility_window, self.interval)
        self.rolling_medium_window = bars_for_duration(self.correlation_window, self.interval)
        self.rolling_long_window = bars_for_duration(self.pca_window, self.interval)
        self.min_threshold_history = bars_for_duration(self.regime_threshold_window, self.interval)
        self.garch_refit_frequency = max(1, self.garch_refit_frequency)
        self.dashboard_time_range = normalize_time_range(self.dashboard_time_range)
        self.forecast_time_range = normalize_time_range(self.forecast_time_range)
        self.backtest_time_range = normalize_time_range(self.backtest_time_range)
        self.allowed_time_ranges = [normalize_time_range(value) for value in self.allowed_time_ranges]
        self.data_dir = Path(self.data_dir)

    @property
    def periods_per_year(self) -> int:
        return periods_per_year(self.interval)

    @property
    def interval_delta(self):
        return interval_to_timedelta(self.interval)

    @property
    def interval_seconds(self) -> int:
        return int(self.interval_delta.total_seconds())

    @property
    def window_bar_counts(self) -> dict[str, int]:
        return {
            "volatility_window": self.rolling_short_window,
            "correlation_window": self.rolling_medium_window,
            "pca_window": self.rolling_long_window,
            "trend_window": bars_for_duration(self.trend_window, self.interval),
            "regime_threshold_window": self.min_threshold_history,
        }

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cleaned_dir(self) -> Path:
        return self.data_dir / "cleaned"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def validation_dir(self) -> Path:
        return self.data_dir / "validation"


def load_config(path: str | Path = "cmva.toml") -> CMVAConfig:
    config_path = Path(path)
    if not config_path.exists():
        return CMVAConfig()
    with config_path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)
    return CMVAConfig(**raw)


def ensure_artifact_dirs(config: CMVAConfig) -> None:
    for path in (
        config.raw_dir,
        config.cleaned_dir,
        config.features_dir,
        config.models_dir,
        config.validation_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
