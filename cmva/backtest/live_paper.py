"""Incremental live paper backtest state."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class LivePaperBacktest:
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    pending_exposure: float = 0.0
    last_exposure: float = 0.0
    returns: list[dict[str, object]] = field(default_factory=list)

    def set_next_exposure(self, exposure: float) -> None:
        self.pending_exposure = float(exposure)

    def settle_closed_candle(self, timestamp: pd.Timestamp, basket_return: float) -> float:
        active_exposure = self.pending_exposure
        turnover = abs(active_exposure - self.last_exposure)
        cost = turnover * ((self.transaction_cost_bps + self.slippage_bps) / 10000.0)
        net_return = active_exposure * float(basket_return) - cost
        self.returns.append(
            {
                "timestamp": timestamp,
                "exposure": active_exposure,
                "basket_return": float(basket_return),
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
            }
        )
        self.last_exposure = active_exposure
        return net_return

    @property
    def cumulative_return(self) -> float:
        value = 1.0
        for row in self.returns:
            value *= 1.0 + float(row["net_return"])
        return value - 1.0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.returns)
