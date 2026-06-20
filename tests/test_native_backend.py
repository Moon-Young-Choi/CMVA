from __future__ import annotations

import importlib.util
from importlib import import_module
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import cmva.features.correlation as correlation_features
import cmva.features.returns as return_features
import cmva.features.trend as trend_features
import cmva.features.volatility as volatility_features
from cmva.features import compute_feature_bundle
from cmva.native.python_backend import PythonBackend


beta_features = import_module("cmva.features.rolling_beta")


def test_python_backend_reference_kernels_are_deterministic():
    backend = PythonBackend()
    prices = np.array([100.0, 101.0, 99.0, 102.0, 103.0])

    log_returns = backend.log_returns(prices)
    rolling = backend.rolling_variance(log_returns, 3, min_periods=2)
    ewma = backend.ewma_variance(log_returns, 3)
    lb = backend.ljung_box_statistic(np.nan_to_num(log_returns), 2)

    assert np.isnan(log_returns[0])
    assert np.isfinite(rolling[-1])
    assert np.isfinite(ewma[-1])
    assert lb["sample_size"] == 5


def test_backend_selection_prefers_cpp_when_available_and_honors_fallback_env():
    code = (
        "import json; "
        "from cmva.native.backend import backend_status; "
        "print(json.dumps(backend_status(), sort_keys=True))"
    )
    default = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    default_status = json.loads(default.stdout)
    forced_env = {**os.environ, "CMVA_USE_CPP": "0"}
    forced = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=forced_env,
    )
    forced_status = json.loads(forced.stdout)

    if importlib.util.find_spec("cmva_cpp") is not None:
        assert default_status["active"] == "cpp"
        assert default_status["cpp_available"] is True
    assert forced_status["active"] == "python"
    assert forced_status["python_fallback"] is True


def test_feature_bundle_routes_core_kernels_through_backend(monkeypatch, synthetic_candles):
    class SpyBackend(PythonBackend):
        def __init__(self):
            self.calls = {
                "compute_log_returns": 0,
                "realized_volatility": 0,
                "ewma_variance": 0,
                "range_based_volatility_array": 0,
                "rolling_average_correlation": 0,
                "rolling_ols_slope": 0,
                "rolling_ols_tstat": 0,
                "rolling_covariance": 0,
            }

        def compute_log_returns(self, close_matrix):
            self.calls["compute_log_returns"] += 1
            return super().compute_log_returns(close_matrix)

        def realized_volatility(self, returns, window):
            self.calls["realized_volatility"] += 1
            return super().realized_volatility(returns, window)

        def ewma_variance(self, returns, span):
            self.calls["ewma_variance"] += 1
            return super().ewma_variance(returns, span)

        def range_based_volatility_array(self, high, low):
            self.calls["range_based_volatility_array"] += 1
            return super().range_based_volatility_array(high, low)

        def rolling_average_correlation(self, returns, window):
            self.calls["rolling_average_correlation"] += 1
            return super().rolling_average_correlation(returns, window)

        def rolling_ols_slope(self, values, window, min_periods=None):
            self.calls["rolling_ols_slope"] += 1
            return super().rolling_ols_slope(values, window, min_periods)

        def rolling_ols_tstat(self, values, window, min_periods=None):
            self.calls["rolling_ols_tstat"] += 1
            return super().rolling_ols_tstat(values, window, min_periods)

        def rolling_covariance(self, left, right, window, min_periods=None):
            self.calls["rolling_covariance"] += 1
            return super().rolling_covariance(left, right, window, min_periods)

    spy = SpyBackend()
    monkeypatch.setattr(return_features, "backend", spy)
    monkeypatch.setattr(volatility_features, "backend", spy)
    monkeypatch.setattr(correlation_features, "backend", spy)
    monkeypatch.setattr(trend_features, "backend", spy)
    monkeypatch.setattr(beta_features, "backend", spy)

    bundle = compute_feature_bundle(synthetic_candles(periods=40), short_window=6, medium_window=12, long_window=18)

    assert not bundle.features.empty
    assert spy.calls["compute_log_returns"] >= 1
    assert spy.calls["realized_volatility"] >= 3
    assert spy.calls["ewma_variance"] >= 1
    assert spy.calls["range_based_volatility_array"] >= 1
    assert spy.calls["rolling_average_correlation"] >= 1
    assert spy.calls["rolling_ols_slope"] >= 1
    assert spy.calls["rolling_ols_tstat"] >= 1
    assert spy.calls["rolling_covariance"] >= 1


@pytest.mark.skipif(importlib.util.find_spec("cmva_cpp") is None, reason="cmva_cpp extension is not built")
def test_cpp_backend_matches_python_reference_for_deterministic_kernels():
    import cmva_cpp

    backend = PythonBackend()
    prices = np.array([100.0, 101.0, 99.0, 102.0, 103.0], dtype=float)
    returns = np.nan_to_num(backend.log_returns(prices))

    np.testing.assert_allclose(cmva_cpp.log_returns(prices), backend.log_returns(prices), equal_nan=True)
    np.testing.assert_allclose(cmva_cpp.rolling_std(returns, 3, 2), backend.rolling_standard_deviation(returns, 3, 2), equal_nan=True)
    np.testing.assert_allclose(cmva_cpp.ewma_variance(returns, 3), backend.ewma_variance(returns, 3), equal_nan=True)
    assert np.isfinite(cmva_cpp.garch_likelihood(returns, 1e-6, 0.05, 0.90))
