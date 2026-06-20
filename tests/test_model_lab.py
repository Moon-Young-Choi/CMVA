from __future__ import annotations

import pandas as pd
from importlib import import_module

from cmva.config import CMVAConfig
from cmva.models.lab import default_candidate_specs, generate_rolling_origins, model_lab_cache_key, run_model_lab, run_two_stage_search
from cmva.native.python_backend import PythonBackend


model_lab = import_module("cmva.models.lab")


def test_rolling_origin_generation_uses_only_history_through_origin():
    origins = generate_rolling_origins(20, training_window_bars=8, horizon_bars=2, refit_stride_bars=3)

    assert origins[0] == 7
    assert all(origin + 2 < 20 for origin in origins)
    assert origins == [7, 10, 13, 16]


def test_two_stage_search_records_information_criteria_and_losses():
    index = pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC")
    returns = pd.Series([0.001 * ((idx % 7) - 3) for idx in range(120)], index=index)
    config = CMVAConfig(
        interval="1h",
        training_window="24h",
        candidate_model_count=8,
        candidate_model_groups=["mean", "volatility"],
        target_view="log_return",
    )

    result = run_two_stage_search(returns, config)

    assert not result.stage1.empty
    assert not result.stage2.empty
    assert not result.leaderboard.empty
    assert {"aic", "bic", "hqic", "avg_rmse", "avg_mae", "avg_qlike"} <= set(result.stage1.columns)
    assert result.leaderboard.iloc[0]["rank"] == 1
    assert result.job_status["status"] == "complete"


def test_candidate_specs_include_both_targets_when_requested():
    specs = default_candidate_specs(
        groups=["mean", "trend", "volatility", "combined"],
        target_view="both",
        limit=10,
    )

    targets = {spec.target for spec in specs}
    families = {spec.family for spec in specs}
    assert targets == {"log_return", "log_price"}
    assert {"mean", "trend", "volatility", "combined"} <= families
    assert len(specs) == 10


def test_two_stage_search_is_deterministic_on_synthetic_data():
    index = pd.date_range("2026-01-01", periods=90, freq="1h", tz="UTC")
    returns = pd.Series([0.0005, -0.001, 0.0002, 0.0009, -0.0003] * 18, index=index)
    config = CMVAConfig(interval="1h", training_window="18h", candidate_model_count=10, target_view="log_return")

    first = run_two_stage_search(returns, config)
    second = run_two_stage_search(returns, config)

    assert first.selected["candidate"] == second.selected["candidate"]
    pd.testing.assert_series_equal(first.leaderboard["model_id"], second.leaderboard["model_id"])


def test_run_model_lab_evaluates_distinct_log_return_and_log_price_targets():
    index = pd.date_range("2026-01-01", periods=96, freq="1h", tz="UTC")
    log_return = pd.Series([0.001 * ((idx % 6) - 3) for idx in range(96)], index=index)
    log_price = log_return.cumsum() + 5.0
    config = CMVAConfig(
        interval="1h",
        training_window="24h",
        candidate_model_count=10,
        candidate_model_groups=["mean", "volatility"],
        target_view="both",
    )

    result = run_model_lab({"log_return": log_return, "log_price": log_price}, config)

    assert set(result.stage1["target"]) == {"log_return", "log_price"}
    assert set(result.leaderboard["target"]) == {"log_return", "log_price"}
    assert result.job_status["evaluated_targets"] == ["log_return", "log_price"]
    assert result.data_hash == model_lab.hash_time_series(pd.DataFrame({"log_return": log_return, "log_price": log_price}))


def test_model_lab_progress_callback_reports_running_and_complete_states():
    index = pd.date_range("2026-01-01", periods=72, freq="1h", tz="UTC")
    log_return = pd.Series([0.001 * ((idx % 6) - 3) for idx in range(72)], index=index)
    log_price = log_return.cumsum() + 5.0
    config = CMVAConfig(
        interval="1h",
        training_window="18h",
        candidate_model_count=8,
        candidate_model_groups=["mean", "volatility"],
        target_view="both",
    )
    updates: list[dict[str, object]] = []

    result = run_model_lab({"log_return": log_return, "log_price": log_price}, config, progress_callback=updates.append)

    statuses = [str(update["status"]) for update in updates]
    assert result.job_status["status"] == "complete"
    assert "queued" in statuses
    assert any(status.startswith("stage1") for status in statuses)
    assert any(status.startswith("stage2") for status in statuses)
    assert updates[-1]["status"] == "complete"
    assert updates[-1]["progress_pct"] == 1.0
    assert any(update.get("active_target") == "log_return" for update in updates)
    assert any(update.get("active_target") == "log_price" for update in updates)


def test_model_lab_cache_key_changes_with_config_or_data():
    index = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
    returns = pd.Series([0.001 * idx for idx in range(20)], index=index)
    config = CMVAConfig(interval="1h")
    changed_config = CMVAConfig(interval="1h", training_window="14d")
    changed_returns = returns.copy()
    changed_returns.iloc[-1] += 0.01

    assert model_lab_cache_key(config, returns) != model_lab_cache_key(changed_config, returns)
    assert model_lab_cache_key(config, returns) != model_lab_cache_key(config, changed_returns)


def test_two_stage_search_dispatches_numerical_work_to_backend(monkeypatch):
    class SpyBackend(PythonBackend):
        def __init__(self):
            self.calls = {
                "qlike": 0,
                "rmse": 0,
                "mae": 0,
                "forecast_bias": 0,
                "forecast_realized_correlation": 0,
                "rank_stability": 0,
                "ar_fit": 0,
                "ewma_variance": 0,
                "arch_likelihood": 0,
                "garch_likelihood": 0,
                "student_t_garch_likelihood": 0,
            }

        def qlike(self, realized_variance, forecast_variance):
            self.calls["qlike"] += 1
            return super().qlike(realized_variance, forecast_variance)

        def rmse(self, actual, forecast):
            self.calls["rmse"] += 1
            return super().rmse(actual, forecast)

        def mae(self, actual, forecast):
            self.calls["mae"] += 1
            return super().mae(actual, forecast)

        def forecast_bias(self, actual, forecast):
            self.calls["forecast_bias"] += 1
            return super().forecast_bias(actual, forecast)

        def forecast_realized_correlation(self, forecast, realized):
            self.calls["forecast_realized_correlation"] += 1
            return super().forecast_realized_correlation(forecast, realized)

        def rank_stability(self, ranks):
            self.calls["rank_stability"] += 1
            return super().rank_stability(ranks)

        def ar_fit(self, values, p):
            self.calls["ar_fit"] += 1
            return super().ar_fit(values, p)

        def ewma_variance(self, returns, span):
            self.calls["ewma_variance"] += 1
            return super().ewma_variance(returns, span)

        def arch_likelihood(self, values, omega, alpha):
            self.calls["arch_likelihood"] += 1
            return super().arch_likelihood(values, omega, alpha)

        def garch_likelihood(self, values, omega, alpha, beta):
            self.calls["garch_likelihood"] += 1
            return super().garch_likelihood(values, omega, alpha, beta)

        def student_t_garch_likelihood(self, values, omega, alpha, beta, nu):
            self.calls["student_t_garch_likelihood"] += 1
            return super().student_t_garch_likelihood(values, omega, alpha, beta, nu)

    spy = SpyBackend()
    monkeypatch.setattr(model_lab, "backend", spy)
    index = pd.date_range("2026-01-01", periods=90, freq="1h", tz="UTC")
    returns = pd.Series([0.001 * ((idx % 9) - 4) for idx in range(90)], index=index)
    config = CMVAConfig(
        interval="1h",
        training_window="18h",
        candidate_model_count=12,
        candidate_model_groups=["mean", "volatility"],
        target_view="log_return",
    )

    result = run_two_stage_search(returns, config)

    assert result.job_status["status"] == "complete"
    for key in spy.calls:
        assert spy.calls[key] > 0, key
