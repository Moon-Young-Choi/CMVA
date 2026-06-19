"""Binance Spot WebSocket kline client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets

from cmva.data.candle import Candle


class BinanceWebSocketClient:
    def __init__(
        self,
        symbols: list[str],
        interval: str = "1h",
        base_url: str = "wss://stream.binance.com:9443",
    ) -> None:
        self.symbols = [symbol.lower() for symbol in symbols]
        self.interval = interval
        self.base_url = base_url.rstrip("/")

    @property
    def stream_url(self) -> str:
        streams = "/".join(f"{symbol}@kline_{self.interval}" for symbol in self.symbols)
        return f"{self.base_url}/stream?streams={streams}"

    async def stream(self) -> AsyncIterator[Candle]:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.stream_url, ping_interval=20, ping_timeout=60) as socket:
                    backoff = 1.0
                    async for message in socket:
                        yield Candle.from_websocket_payload(json.loads(message))
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
