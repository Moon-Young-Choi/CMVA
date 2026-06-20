from __future__ import annotations

import pandas as pd
import pytest

from cmva.simulation.scoring import (
    expectation_label,
    information_criterion_weights,
    score_origin,
    seasonality_label,
    trend_label,
    volatility_label,
)
from cmva.simulation.spec import SimulationSpec


def _base_spec_payload() -> dict[str, object]:
    return {
        "run_name": "테스트 실행",
        "symbols": "btcusdt, ETHUSDT, btcusdt",
        "interval": "1h",
        "data_start": "2026-01-01T00:00:00Z",
        "data_end": "2026-01-04T00:00:00Z",
        "T": "24h",
        "dT": "6 bars",
        "S": "10",
        "s_role": "scenario_count",
        "forecast_horizon_bars": 1,
    }


def test_simulation_spec_requires_t_dt_s():
    for field in ("T", "dT", "S"):
        payload = _base_spec_payload()
        payload[field] = ""
        with pytest.raises(ValueError):
            SimulationSpec.from_mapping(payload)


def test_simulation_spec_validation_and_bar_conversion():
    spec = SimulationSpec.from_mapping(_base_spec_payload())

    assert spec.symbols == ["BTCUSDT", "ETHUSDT"]
    assert spec.interval == "1h"
    assert spec.t_bars == 24
    assert spec.dt_bars == 6
    assert spec.data_start == pd.Timestamp("2026-01-01T00:00:00Z")


def test_simulation_spec_rejects_invalid_interval_and_range():
    payload = _base_spec_payload()
    payload["interval"] = "2x"
    with pytest.raises(ValueError):
        SimulationSpec.from_mapping(payload)

    payload = _base_spec_payload()
    payload["data_start"] = payload["data_end"]
    with pytest.raises(ValueError):
        SimulationSpec.from_mapping(payload)


def test_information_criterion_weights_sum_to_one():
    records = [
        {"candidate": "a", "bic": 10.0, "aic": 9.0, "forecast_volatility": 0.1, "converged": True},
        {"candidate": "b", "bic": 12.0, "aic": 8.0, "forecast_volatility": 0.2, "converged": True},
    ]

    bic_weights = information_criterion_weights(records, "bic_weighted_percentile")
    aic_weights = information_criterion_weights(records, "aic_weighted_percentile")

    assert sum(bic_weights.values()) == pytest.approx(1.0)
    assert sum(aic_weights.values()) == pytest.approx(1.0)
    assert bic_weights["a"] > bic_weights["b"]
    assert aic_weights["b"] > aic_weights["a"]


def test_score_ranges_and_labels_are_bounded():
    records = [
        {
            "candidate": "garch_1_1:log_return",
            "model_id": "garch_1_1",
            "model_family": "volatility",
            "estimator": "MLE/QMLE",
            "forecast_volatility": 0.04,
            "forecast_mean": 0.002,
            "converged": True,
            "bic": 10.0,
            "aic": 9.0,
            "qlike": 0.5,
            "rmse_loss": 0.1,
            "mae_loss": 0.1,
            "realized_volatility": 0.03,
            "realized_return": 0.01,
        },
        {
            "candidate": "seasonal_ar_1:log_return",
            "model_id": "seasonal_ar_1",
            "model_family": "trend",
            "estimator": "계절 더미",
            "forecast_volatility": 0.02,
            "forecast_mean": -0.001,
            "converged": True,
            "bic": 11.0,
            "aic": 10.0,
            "qlike": 0.7,
            "rmse_loss": 0.2,
            "mae_loss": 0.2,
            "realized_volatility": 0.03,
            "realized_return": 0.01,
        },
    ]

    score = score_origin(
        run_id="run",
        symbol="BTCUSDT",
        origin_time="2026-01-02T00:00:00Z",
        records=records,
        prior_forecasts={"garch_1_1:log_return": [0.01, 0.02, 0.03], "seasonal_ar_1:log_return": [0.01]},
        scoring_method="bic_weighted_percentile",
    )

    assert 0 <= score.volatility_score_0_100 <= 100
    assert 0 <= score.expectation_score_0_100 <= 100
    assert -100 <= score.trend_score_minus100_100 <= 100
    assert 0 <= score.seasonality_score_0_100 <= 100
    assert score.volatility_confidence_0_1 < 1.0
    assert volatility_label(85) == "극단적 변동성"
    assert expectation_label(10) == "강한 하방 기대"
    assert trend_label(-70) == "강한 하락 추세"
    assert seasonality_label(10) == "계절성 거의 없음"
