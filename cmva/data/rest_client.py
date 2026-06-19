"""Binance Spot REST client."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

from cmva.data.candle import Candle


class BinanceRestClient:
    def __init__(self, base_url: str = "https://api.binance.com", timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        params: dict[str, object] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = _timestamp_ms(start_time)
        if end_time is not None:
            params["endTime"] = _timestamp_ms(end_time)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v3/klines", params=params)
            response.raise_for_status()
            rows = response.json()
        return [Candle.from_rest_kline(symbol, interval, row) for row in rows]

    async def fetch_historical_range(
        self,
        symbol: str,
        interval: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        limit: int = 1000,
    ) -> list[Candle]:
        candles: list[Candle] = []
        cursor = pd.Timestamp(start_time).tz_convert("UTC") if start_time.tzinfo else pd.Timestamp(start_time).tz_localize("UTC")
        end = pd.Timestamp(end_time).tz_convert("UTC") if end_time.tzinfo else pd.Timestamp(end_time).tz_localize("UTC")
        while cursor <= end:
            batch = await self.fetch_klines(symbol, interval, cursor, end, limit)
            if not batch:
                break
            candles.extend(batch)
            next_open = batch[-1].open_time + pd.Timedelta(milliseconds=1)
            if next_open <= cursor:
                break
            cursor = next_open
            if len(batch) < min(limit, 1000):
                break
        return candles


def _timestamp_ms(value: pd.Timestamp | datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return int(ts.timestamp() * 1000)
