from __future__ import annotations

from cmva.regime.classifier import ASSET_SPECIFIC, IDIOSYNCRATIC_HIGH_VOL, SYSTEMIC_RISK, classify_regime


THRESHOLDS = {
    "forecast_vol_high": 0.02,
    "market_vol_high": 0.02,
    "corr_high": 0.6,
    "corr_low": 0.2,
    "pca_high": 0.7,
    "pca_low": 0.4,
    "dispersion_high": 0.01,
}


def test_regime_classifier_assigns_systemic_risk():
    assert classify_regime(0.03, 0.03, 0.8, 0.8, 0.005, THRESHOLDS) == SYSTEMIC_RISK


def test_regime_classifier_assigns_idiosyncratic_high_vol():
    assert classify_regime(0.03, 0.03, 0.1, 0.5, 0.02, THRESHOLDS) == IDIOSYNCRATIC_HIGH_VOL


def test_regime_classifier_assigns_asset_specific():
    assert classify_regime(0.01, 0.01, 0.1, 0.2, 0.005, THRESHOLDS) == ASSET_SPECIFIC
