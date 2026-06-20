from __future__ import annotations

import asyncio

import pytest

from cmva.data.candle import Candle
from cmva.simulation.repository import SimulationRepository
from cmva.simulation.runner import SimulationRunner, generate_simulation_origins, validate_simulation_data
from cmva.simulation.spec import SimulationSpec


class FakeRestClient:
    def __init__(self, frame):
        self.frame = frame
        self.requests = []

    async def fetch_historical_range(self, symbol, interval, start_time, end_time, limit=1000):
        self.requests.append((symbol, interval, start_time, end_time))
        rows = self.frame.loc[
            (self.frame["symbol"] == symbol)
            & (self.frame["interval"] == interval)
            & (self.frame["open_time"] >= start_time)
            & (self.frame["open_time"] <= end_time)
        ]
        return [
            Candle(
                symbol=row.symbol,
                interval=row.interval,
                open_time=row.open_time,
                close_time=row.close_time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                is_closed=row.is_closed,
            )
            for row in rows.itertuples(index=False)
        ]


def _spec(**overrides) -> SimulationSpec:
    payload = {
        "run_name": "runner",
        "symbols": ["BTCUSDT"],
        "interval": "1h",
        "data_start": "2026-01-01T00:00:00Z",
        "data_end": "2026-01-03T00:00:00Z",
        "T": "12 bars",
        "dT": "6 bars",
        "S": "1.0",
        "s_role": "initial_scale",
        "forecast_horizon_bars": 1,
        "candidate_model_groups": ["mean", "volatility", "trend"],
        "candidate_model_count": 6,
        "target_view": "log_return",
        "scoring_method": "bic_weighted_percentile",
        **overrides,
    }
    return SimulationSpec.from_mapping(payload)


def test_validate_simulation_data_detects_missing_and_unclosed(synthetic_candles):
    spec = _spec()
    frame = synthetic_candles(symbols=["BTCUSDT"], periods=40, interval="1h")
    frame = frame.drop(frame.index[5]).reset_index(drop=True)

    report = validate_simulation_data(frame, spec)

    assert report["is_valid"]
    assert report["symbols"]["BTCUSDT"]["missing_candle_count"] >= 1
    assert report["warnings"]

    frame.loc[0, "is_closed"] = False
    report = validate_simulation_data(frame, spec)
    assert not report["is_valid"]
    assert any("미완성" in error for error in report["errors"])


def test_repository_create_save_load_run(tmp_path):
    repo = SimulationRepository(tmp_path / "sims")
    spec = _spec()
    run_id = repo.create_run(spec)

    repo.append_step_results(run_id, [{"run_id": run_id, "candidate": "naive_mean:log_return", "origin_time": "t"}])
    repo.append_score_results(run_id, [{"run_id": run_id, "symbol": "BTCUSDT", "origin_time": "t"}])
    repo.mark_completed(run_id)

    loaded = repo.load_final_results(run_id)
    assert loaded["spec"]["T"] == "12 bars"
    assert loaded["progress"]["status"] == "completed"
    assert loaded["step_results"][0]["candidate"] == "naive_mean:log_return"
    assert loaded["scores"][0]["symbol"] == "BTCUSDT"


def test_runner_generates_origins_without_lookahead_and_records_results(tmp_path, synthetic_candles):
    frame = synthetic_candles(symbols=["BTCUSDT"], periods=48, interval="1h")
    spec = _spec()
    repo = SimulationRepository(tmp_path / "sims")
    run_id = repo.create_run(spec)
    rest = FakeRestClient(frame)
    runner = SimulationRunner(repo, rest_client=rest)

    asyncio.run(runner.run(spec))

    assert rest.requests
    results = repo.load_final_results(run_id)
    assert results["progress"]["status"] == "completed"
    assert results["scores"]
    assert results["step_results"]
    first = results["step_results"][0]
    assert first["train_end_time"] <= first["origin_time"]
    assert first["realization_time"] > first["origin_time"]
    assert {"estimator", "params", "aic", "bic", "hqic", "qlike", "rmse_loss", "mae_loss"} <= set(first)


def test_generate_simulation_origins_from_t_and_dt():
    assert generate_simulation_origins(30, training_window_bars=10, horizon_bars=2, stride_bars=5) == [9, 14, 19, 24]
