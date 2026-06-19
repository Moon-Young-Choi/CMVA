"""Local parquet storage for candles and derived frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmva.data.candle import Candle, candles_to_frame, closed_only, normalize_candle_frame


class CandleStorage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def symbol_path(self, symbol: str) -> Path:
        return self.root / f"{symbol.upper()}.parquet"

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        path = self.symbol_path(symbol)
        if not path.exists():
            return candles_to_frame([])
        return normalize_candle_frame(pd.read_parquet(path))

    def load_many(self, symbols: list[str]) -> pd.DataFrame:
        frames = [self.load_symbol(symbol) for symbol in symbols]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return candles_to_frame([])
        return normalize_candle_frame(pd.concat(frames, ignore_index=True))

    def upsert_closed(self, candles: list[Candle]) -> pd.DataFrame:
        closed = [candle for candle in candles if candle.is_closed]
        if not closed:
            return candles_to_frame([])
        written: list[pd.DataFrame] = []
        for symbol in sorted({candle.symbol for candle in closed}):
            incoming = candles_to_frame([candle for candle in closed if candle.symbol == symbol])
            existing = self.load_symbol(symbol)
            combined = normalize_candle_frame(pd.concat([existing, incoming], ignore_index=True))
            combined = combined.drop_duplicates(["symbol", "interval", "open_time"], keep="last")
            combined = closed_only(combined)
            combined.to_parquet(self.symbol_path(symbol), index=False)
            written.append(combined)
        return normalize_candle_frame(pd.concat(written, ignore_index=True))


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=True)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
