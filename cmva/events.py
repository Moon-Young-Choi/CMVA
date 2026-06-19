"""Simple application event types."""

from __future__ import annotations

from dataclasses import dataclass

from cmva.data.candle import Candle


@dataclass(frozen=True)
class CandleEvent:
    candle: Candle


@dataclass(frozen=True)
class CommandEvent:
    name: str
