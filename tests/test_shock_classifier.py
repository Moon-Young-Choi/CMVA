from __future__ import annotations

from cmva.regime.shock import (
    IDIOSYNCRATIC_VOL_SHOCK,
    MODERATE_SHOCK,
    SYSTEMIC_VOL_SHOCK,
    VOL_REGIME_BUILDUP,
    classify_shock,
)


def test_shock_classifier_systemic_example():
    assert (
        classify_shock(shock_score=4.0, shock_breadth=0.8, avg_corr=0.7, dispersion=0.01)
        == SYSTEMIC_VOL_SHOCK
    )


def test_shock_classifier_idiosyncratic_example():
    assert (
        classify_shock(
            shock_score=4.0,
            shock_breadth=0.2,
            avg_corr=0.1,
            dispersion=0.05,
            dispersion_high_threshold=0.02,
        )
        == IDIOSYNCRATIC_VOL_SHOCK
    )


def test_shock_classifier_regime_buildup_and_moderate():
    assert classify_shock(shock_score=1.0, rv_jump_ratio=2.5) == VOL_REGIME_BUILDUP
    assert classify_shock(shock_score=2.2, rv_jump_ratio=1.0) == MODERATE_SHOCK
