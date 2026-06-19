"""Configuration loading for CMVA."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class CMVAConfig:
    symbols: list[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    )
    interval: str = "1h"
    historical_days: int = 365
    bootstrap_limit: int = 1000
    rolling_short_window: int = 24
    rolling_medium_window: int = 168
    rolling_long_window: int = 720
    min_threshold_history: int = 720
    garch_refit_frequency: int = 24
    target_annual_vol: float = 0.20
    max_leverage: float = 1.5
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    severe_shock_threshold: float = 3.0
    moderate_shock_threshold: float = 2.0
    shock_breadth_threshold: float = 0.60
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    rest_base_url: str = "https://data-api.binance.vision"
    websocket_base_url: str = "wss://data-stream.binance.vision"

    def __post_init__(self) -> None:
        symbols = [symbol.upper().strip() for symbol in self.symbols if symbol.strip()]
        if not symbols:
            raise ValueError("at least one symbol is required")
        self.symbols = symbols
        self.data_dir = Path(self.data_dir)
        self.reports_dir = Path(self.reports_dir)

    @property
    def periods_per_year(self) -> int:
        if self.interval != "1h":
            return 365 * 24
        return 365 * 24

    @property
    def target_vol_per_period(self) -> float:
        return self.target_annual_vol / math.sqrt(self.periods_per_year)

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
    def backtests_dir(self) -> Path:
        return self.data_dir / "backtests"


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
        config.backtests_dir,
        config.reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
