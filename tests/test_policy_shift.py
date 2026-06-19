from __future__ import annotations

import pandas as pd
import pytest

from cmva.backtest.live_paper import LivePaperBacktest
from cmva.policy.regime_vol_target import RegimeVolTargetPolicy


def test_live_paper_settles_previous_exposure_only_after_next_candle_closes():
    paper = LivePaperBacktest(transaction_cost_bps=0.0, slippage_bps=0.0)
    paper.set_next_exposure(1.0)
    assert paper.returns == []
    net = paper.settle_closed_candle(pd.Timestamp("2026-01-01T01:00:00Z"), 0.05)
    assert net == 0.05
    assert paper.cumulative_return == pytest.approx(0.05)


def test_regime_policy_multiplier_reduces_systemic_exposure():
    policy = RegimeVolTargetPolicy(target_vol_per_period=0.01, max_leverage=2.0)
    asset_specific = policy.target_exposure(0.01, "ASSET_SPECIFIC")
    systemic = policy.target_exposure(0.01, "SYSTEMIC_RISK")
    assert systemic < asset_specific
