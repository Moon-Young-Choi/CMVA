from __future__ import annotations

from cmva.web.glossary import load_glossary


def test_glossary_contains_required_tooltip_keys():
    glossary = load_glossary()
    required = {
        "volatility",
        "realized_volatility",
        "ewma",
        "garch_forecast",
        "shock_score",
        "average_correlation",
        "pca1_share",
        "dispersion",
        "rolling_beta",
        "trend_slope",
        "trend_tstat",
        "qlike",
        "rmse",
        "mae",
        "ljung_box",
        "kupiec",
        "walk_forward",
        "lookahead_bias",
        "closed_candle",
        "forming_candle",
        "bootstrap_progress",
        "validation_issue",
    }

    assert required <= set(glossary)
    assert all(glossary[key]["short"] for key in required)
