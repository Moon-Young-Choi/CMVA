from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_candles():
    def build(
        symbols: list[str] | None = None,
        periods: int = 48,
        start: str = "2026-01-01 00:00:00+00:00",
        closed: bool = True,
    ) -> pd.DataFrame:
        selected = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        index = pd.date_range(start=start, periods=periods, freq="1h", tz="UTC")
        rows = []
        for symbol_idx, symbol in enumerate(selected):
            base = 100.0 + symbol_idx * 10.0
            prices = base * np.exp(np.linspace(0.0, 0.01, periods) + 0.002 * np.sin(np.arange(periods) / 3))
            for idx, timestamp in enumerate(index):
                close = float(prices[idx])
                open_price = float(prices[idx - 1]) if idx else close
                high = max(open_price, close) * 1.001
                low = min(open_price, close) * 0.999
                rows.append(
                    {
                        "symbol": symbol,
                        "interval": "1h",
                        "open_time": timestamp,
                        "close_time": timestamp + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": 1000.0 + idx,
                        "is_closed": closed,
                    }
                )
        return pd.DataFrame(rows)

    return build
