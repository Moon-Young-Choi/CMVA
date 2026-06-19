from __future__ import annotations

import pandas as pd

from cmva.models.garch import GarchVolatilityModel


def test_garch_failed_fit_is_handled_gracefully():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])
    model = GarchVolatilityModel(min_observations=100)
    fit = model.fit(returns)
    forecast = model.forecast_one_step(returns)
    assert not fit.success
    assert forecast.degraded
    assert forecast.volatility > 0
