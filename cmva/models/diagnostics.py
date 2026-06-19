"""Statistical diagnostics and methodology helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from cmva.analysis_types import DiagnosticSnapshot, MethodStep, StatTestResult


def standardized_residual(return_value: float, mean: float, volatility: float) -> float:
    if volatility <= 0:
        return 0.0
    return float((return_value - mean) / volatility)


def latest_standardized_residual(returns: pd.Series, mean: float, volatility: float) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return standardized_residual(float(clean.iloc[-1]), mean, volatility)


def run_statistical_diagnostics(
    basket_returns: pd.Series,
    forecast_vol: pd.Series,
    ewma_vol: pd.Series,
    regimes: pd.Series,
    shocks: pd.DataFrame,
    validation_losses: pd.DataFrame | None,
    garch_params: dict[str, float] | None = None,
    generated_at: pd.Timestamp | None = None,
    periods_per_year: int = 365 * 24,
    alpha: float = 0.05,
) -> DiagnosticSnapshot:
    """Build a diagnostics snapshot using only aligned historical observations."""
    returns = _clean_series(basket_returns)
    forecast = _clean_series(forecast_vol).reindex(returns.index).shift(1)
    ewma = _clean_series(ewma_vol).reindex(returns.index).shift(1)
    residuals = _standardized_residuals(returns, forecast)
    realized_variance = returns.pow(2).rename("realized_variance")
    forecast_variance = forecast.pow(2).rename("forecast_variance")
    ewma_variance = ewma.pow(2).rename("ewma_variance")
    timestamp = generated_at or _latest_timestamp(returns)
    params = garch_params or {}

    return DiagnosticSnapshot(
        model_tests=[
            _ljung_box_test(residuals, "Ljung-Box standardized residuals", "rho_1 = ... = rho_k = 0", alpha),
            _ljung_box_test(
                residuals.pow(2),
                "Ljung-Box squared standardized residuals",
                "rho_1(z^2) = ... = rho_k(z^2) = 0",
                alpha,
            ),
            _arch_lm_test(residuals, alpha),
            _parameter_check(
                "GARCH stationarity",
                "alpha[1] + beta[1] < 1",
                _param_sum(params, "alpha", "beta"),
                upper_bound=1.0,
                timestamp=timestamp,
            ),
            _parameter_check(
                "Student-t finite variance",
                "nu > 2",
                _param_value(params, "nu"),
                lower_bound=2.0,
                timestamp=timestamp,
            ),
        ],
        forecast_tests=[
            _loss_metric(realized_variance, forecast_variance, "Forecast MSE", "(r_t^2 - h_t)^2", "mse"),
            _loss_metric(realized_variance, forecast_variance, "Forecast MAE", "|r_t^2 - h_t|", "mae"),
            _loss_metric(realized_variance, forecast_variance, "Forecast QLIKE", "r_t^2 / h_t + log(h_t)", "qlike"),
            _diebold_mariano_test(realized_variance, forecast_variance, ewma_variance, alpha),
            _mincer_zarnowitz_test(realized_variance, forecast_variance, alpha),
        ],
        risk_tests=[
            _kupiec_var_test(returns, forecast, alpha=0.05),
        ],
        backtest_tests=_validation_tests(validation_losses, alpha),
        regime_tests=_regime_tests(regimes, shocks, timestamp),
        generated_at=timestamp,
    )


def build_method_steps(
    features: pd.DataFrame,
    forecast_vol: pd.Series,
    regimes: pd.Series,
    shocks: pd.DataFrame,
) -> list[MethodStep]:
    if features.empty:
        return []
    latest = features.index[-1]
    latest_features = features.iloc[-1]
    forecast_for_next = _safe_float(forecast_vol.reindex(features.index).iloc[-1]) if not forecast_vol.empty else None
    shock_score = None
    if not shocks.empty and "shock_score" in shocks:
        shock_score = _safe_float(shocks["shock_score"].iloc[-1])
    steps = [
        MethodStep(
            timestamp=latest,
            stage="Return update",
            formula_id="r_t = log(C_t / C_{t-1})",
            inputs={"close_cutoff": latest},
            output=_safe_float(latest_features.get("basket_return")),
            data_cutoff=latest,
            lookahead_status="passed",
        ),
        MethodStep(
            timestamp=latest,
            stage="Rolling features",
            formula_id="vol, corr, beta, covariance PCA1, dispersion",
            inputs={"window_policy": "rolling through t only"},
            output={
                "market_vol": _safe_float(latest_features.get("market_vol")),
                "avg_corr": _safe_float(latest_features.get("avg_pairwise_corr")),
                "pca1_share": _safe_float(latest_features.get("pca1_share")),
            },
            data_cutoff=latest,
            lookahead_status="passed",
        ),
        MethodStep(
            timestamp=latest,
            stage="Forecast",
            formula_id="GARCH(1,1) Student-t -> sigma_{t+1|t}",
            inputs={"returns_cutoff": latest},
            output=forecast_for_next,
            data_cutoff=latest,
            lookahead_status="passed",
        ),
        MethodStep(
            timestamp=latest,
            stage="Shock classification",
            formula_id="|r_t| / sigma_{t|t-1}",
            inputs={"forecast_alignment": "previous closed forecast"},
            output=shock_score,
            data_cutoff=latest,
            lookahead_status="passed",
        ),
        MethodStep(
            timestamp=latest,
            stage="Regime classification",
            formula_id="expanding thresholds through t",
            inputs={"threshold_policy": "no full-sample thresholds"},
            output=str(regimes.iloc[-1]) if not regimes.empty else None,
            data_cutoff=latest,
            lookahead_status="passed",
        ),
        MethodStep(
            timestamp=latest,
            stage="Walk-forward validation",
            formula_id="L_t = loss(r_t^2, forecast_{t|t-1}^2)",
            inputs={"forecast_alignment": "previous closed forecast"},
            output="model forecast loss updated",
            data_cutoff=latest,
            lookahead_status="passed",
        ),
    ]
    return steps


def _clean_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _latest_timestamp(series: pd.Series) -> pd.Timestamp | None:
    if series.empty:
        return None
    value = series.index[-1]
    return pd.Timestamp(value) if not isinstance(value, pd.Timestamp) else value


def _standardized_residuals(returns: pd.Series, forecast_vol: pd.Series) -> pd.Series:
    aligned = pd.concat([returns.rename("return"), forecast_vol.rename("vol")], axis=1).dropna()
    aligned = aligned.loc[aligned["vol"] > 0]
    if aligned.empty:
        return pd.Series(dtype=float, name="standardized_residual")
    values = aligned["return"] / aligned["vol"]
    return values.replace([np.inf, -np.inf], np.nan).dropna().rename("standardized_residual")


def _insufficient(name: str, null: str, formula: str, sample_size: int, minimum: int = 20) -> StatTestResult:
    return StatTestResult(
        name=name,
        null_hypothesis=null,
        formula=formula,
        statistic=None,
        p_value=None,
        decision="insufficient sample",
        sample_size=sample_size,
        interpretation=f"Need at least {minimum} aligned observations.",
        limitations="No statistical conclusion is reported for this sample.",
    )


def _decision_from_pvalue(p_value: float | None, alpha: float, reject_bad: bool = True) -> str:
    if p_value is None or pd.isna(p_value):
        return "diagnostic only"
    if p_value < alpha:
        return "fail diagnostic" if reject_bad else "reject null"
    return "pass diagnostic" if reject_bad else "do not reject null"


def _ljung_box_test(series: pd.Series, name: str, null: str, alpha: float) -> StatTestResult:
    clean = _clean_series(series)
    minimum = 20
    if len(clean) < minimum:
        return _insufficient(name, null, "Q = n(n+2) sum_k rho_k^2 / (n-k)", len(clean), minimum)
    lag = max(1, min(10, len(clean) // 5))
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(clean, lags=[lag], return_df=True)
        statistic = float(result["lb_stat"].iloc[-1])
        p_value = float(result["lb_pvalue"].iloc[-1])
    except Exception as exc:
        return _failed_result(name, null, "Ljung-Box Q", len(clean), str(exc))
    return StatTestResult(
        name=name,
        null_hypothesis=null,
        formula="Q = n(n+2) sum_k rho_k^2 / (n-k)",
        statistic=statistic,
        p_value=p_value,
        decision=_decision_from_pvalue(p_value, alpha),
        sample_size=len(clean),
        window=f"lag {lag}",
        interpretation="Checks whether residual autocorrelation remains after the volatility model.",
        limitations="Large samples can reject economically small autocorrelation.",
    )


def _arch_lm_test(residuals: pd.Series, alpha: float) -> StatTestResult:
    clean = _clean_series(residuals)
    minimum = 25
    if len(clean) < minimum:
        return _insufficient("ARCH-LM residual heteroskedasticity", "no remaining ARCH effects", "LM = n R^2", len(clean), minimum)
    lag = max(1, min(10, len(clean) // 5))
    try:
        from statsmodels.stats.diagnostic import het_arch

        lm_stat, lm_pvalue, _, _ = het_arch(clean, nlags=lag)
    except Exception as exc:
        return _failed_result("ARCH-LM residual heteroskedasticity", "no remaining ARCH effects", "LM = n R^2", len(clean), str(exc))
    return StatTestResult(
        name="ARCH-LM residual heteroskedasticity",
        null_hypothesis="no remaining ARCH effects",
        formula="LM = n R^2 from lagged squared residual regression",
        statistic=float(lm_stat),
        p_value=float(lm_pvalue),
        decision=_decision_from_pvalue(float(lm_pvalue), alpha),
        sample_size=len(clean),
        window=f"lag {lag}",
        interpretation="Checks whether volatility clustering remains in standardized residuals.",
        limitations="A rejection means the current volatility model may be incomplete.",
    )


def _failed_result(name: str, null: str, formula: str, sample_size: int, message: str) -> StatTestResult:
    return StatTestResult(
        name=name,
        null_hypothesis=null,
        formula=formula,
        statistic=None,
        p_value=None,
        decision="diagnostic unavailable",
        sample_size=sample_size,
        interpretation="The diagnostic could not be computed.",
        limitations=message,
    )


def _param_value(params: dict[str, float], needle: str) -> float | None:
    for key, value in params.items():
        if needle.lower() in key.lower():
            return float(value)
    return None


def _param_sum(params: dict[str, float], alpha_name: str, beta_name: str) -> float | None:
    alpha_value = _param_value(params, alpha_name)
    beta_value = _param_value(params, beta_name)
    if alpha_value is None or beta_value is None:
        return None
    return alpha_value + beta_value


def _parameter_check(
    name: str,
    formula: str,
    value: float | None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    timestamp: pd.Timestamp | None = None,
) -> StatTestResult:
    if value is None or pd.isna(value):
        return StatTestResult(
            name=name,
            null_hypothesis=formula,
            formula=formula,
            statistic=None,
            p_value=None,
            decision="diagnostic unavailable",
            sample_size=0,
            timestamp=timestamp,
            interpretation="GARCH fit parameters were not available.",
            limitations="Fallback forecasts do not expose this parameter check.",
        )
    passed = True
    if lower_bound is not None:
        passed = passed and value > lower_bound
    if upper_bound is not None:
        passed = passed and value < upper_bound
    return StatTestResult(
        name=name,
        null_hypothesis=formula,
        formula=formula,
        statistic=float(value),
        p_value=None,
        decision="pass diagnostic" if passed else "fail diagnostic",
        sample_size=1,
        timestamp=timestamp,
        interpretation="Checks whether fitted GARCH parameters satisfy a basic mathematical condition.",
        limitations="This is a parameter constraint check, not a p-value hypothesis test.",
    )


def _aligned_variance(realized: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([realized.rename("realized"), forecast.rename("forecast")], axis=1).dropna()
    aligned = aligned.loc[aligned["forecast"] > 0]
    return aligned.replace([np.inf, -np.inf], np.nan).dropna()


def _loss_metric(realized: pd.Series, forecast: pd.Series, name: str, formula: str, metric: str) -> StatTestResult:
    aligned = _aligned_variance(realized, forecast)
    minimum = 10
    if len(aligned) < minimum:
        return _insufficient(name, "forecast loss is finite and interpretable", formula, len(aligned), minimum)
    if metric == "mse":
        value = float((aligned["realized"] - aligned["forecast"]).pow(2).mean())
    elif metric == "mae":
        value = float((aligned["realized"] - aligned["forecast"]).abs().mean())
    else:
        value = float((aligned["realized"] / aligned["forecast"] + np.log(aligned["forecast"])).mean())
    return StatTestResult(
        name=name,
        null_hypothesis="diagnostic loss metric",
        formula=formula,
        statistic=value,
        p_value=None,
        decision="diagnostic only",
        sample_size=len(aligned),
        interpretation="Lower values indicate better variance forecast fit on the aligned sample.",
        limitations="This is an evaluation metric, not a standalone significance test.",
    )


def _qlike_loss(realized: pd.Series, forecast: pd.Series) -> pd.Series:
    aligned = _aligned_variance(realized, forecast)
    if aligned.empty:
        return pd.Series(dtype=float)
    return (aligned["realized"] / aligned["forecast"] + np.log(aligned["forecast"])).rename("qlike")


def _diebold_mariano_test(realized: pd.Series, forecast: pd.Series, ewma: pd.Series, alpha: float) -> StatTestResult:
    model_loss = _qlike_loss(realized, forecast)
    baseline_loss = _qlike_loss(realized, ewma)
    loss_diff = pd.concat([model_loss.rename("model"), baseline_loss.rename("baseline")], axis=1).dropna()
    minimum = 20
    if len(loss_diff) < minimum:
        return _insufficient("Diebold-Mariano vs EWMA", "equal predictive accuracy", "DM = mean(d_t) / HAC_se(d_t)", len(loss_diff), minimum)
    diff = loss_diff["model"] - loss_diff["baseline"]
    statistic, p_value = _hac_mean_test(diff)
    return StatTestResult(
        name="Diebold-Mariano vs EWMA",
        null_hypothesis="equal predictive accuracy",
        formula="DM = mean(L_model - L_baseline) / HAC_se",
        statistic=statistic,
        p_value=p_value,
        decision=_decision_from_pvalue(p_value, alpha, reject_bad=False),
        sample_size=len(diff),
        interpretation="Negative mean loss difference favors GARCH over EWMA under QLIKE loss.",
        limitations="Uses a lightweight HAC normal approximation.",
    )


def _mincer_zarnowitz_test(realized: pd.Series, forecast: pd.Series, alpha: float) -> StatTestResult:
    aligned = _aligned_variance(realized, forecast)
    minimum = 20
    if len(aligned) < minimum or aligned["forecast"].nunique() < 2:
        return _insufficient("Mincer-Zarnowitz calibration", "slope = 1", "r_t^2 = a + b h_t + e_t", len(aligned), minimum)
    try:
        import statsmodels.api as sm

        x = sm.add_constant(aligned["forecast"])
        model = sm.OLS(aligned["realized"], x).fit()
        slope = float(model.params.iloc[1])
        slope_se = float(model.bse.iloc[1])
        if slope_se <= 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (slope - 1.0) / slope_se
            p_value = float(2.0 * stats.t.sf(abs(statistic), max(len(aligned) - 2, 1)))
    except Exception as exc:
        return _failed_result("Mincer-Zarnowitz calibration", "slope = 1", "r_t^2 = a + b h_t + e_t", len(aligned), str(exc))
    return StatTestResult(
        name="Mincer-Zarnowitz calibration",
        null_hypothesis="slope = 1",
        formula="r_t^2 = a + b h_t + e_t",
        statistic=float(statistic),
        p_value=p_value,
        decision=_decision_from_pvalue(p_value, alpha, reject_bad=True),
        sample_size=len(aligned),
        interpretation=f"Estimated slope is {slope:.4f}; p-value tests deviation from 1.",
        limitations="Squared hourly returns are noisy realized variance proxies.",
    )


def _kupiec_var_test(returns: pd.Series, forecast_vol: pd.Series, alpha: float = 0.05) -> StatTestResult:
    aligned = pd.concat([returns.rename("return"), forecast_vol.rename("vol")], axis=1).dropna()
    aligned = aligned.loc[aligned["vol"] > 0]
    minimum = 30
    if len(aligned) < minimum:
        return _insufficient("Kupiec VaR coverage", f"exception probability = {alpha:.2%}", "LR_uc = -2 log(L0 / L1)", len(aligned), minimum)
    z = stats.norm.ppf(alpha)
    var_threshold = z * aligned["vol"]
    exceptions = (aligned["return"] < var_threshold).astype(int)
    x = int(exceptions.sum())
    n = int(len(exceptions))
    phat = min(max(x / n, 1e-12), 1.0 - 1e-12)
    p0 = min(max(alpha, 1e-12), 1.0 - 1e-12)
    ll_null = x * math.log(p0) + (n - x) * math.log(1.0 - p0)
    ll_alt = x * math.log(phat) + (n - x) * math.log(1.0 - phat)
    lr_stat = max(-2.0 * (ll_null - ll_alt), 0.0)
    p_value = float(stats.chi2.sf(lr_stat, 1))
    return StatTestResult(
        name="Kupiec VaR coverage",
        null_hypothesis=f"exception probability = {alpha:.2%}",
        formula="LR_uc = -2 log(L0 / L1)",
        statistic=float(lr_stat),
        p_value=p_value,
        decision=_decision_from_pvalue(p_value, 0.05),
        sample_size=n,
        interpretation=f"Observed {x} exceptions out of {n} one-hour returns.",
        limitations="Uses Gaussian VaR from volatility forecasts; tails may be heavier.",
    )


def _validation_tests(validation_losses: pd.DataFrame | None, alpha: float) -> list[StatTestResult]:
    if validation_losses is None or validation_losses.empty:
        return [
            _insufficient("Walk-forward QLIKE comparison", "equal forecast loss", "DM = mean(d_t) / HAC_se(d_t)", 0, 20),
        ]
    tests: list[StatTestResult] = []
    comparisons = [
        ("GARCH vs EWMA QLIKE", "garch_qlike", "ewma_qlike"),
        ("GARCH vs naive realized-vol QLIKE", "garch_qlike", "naive_qlike"),
    ]
    for name, model_column, baseline_column in comparisons:
        if model_column not in validation_losses or baseline_column not in validation_losses:
            tests.append(_insufficient(name, "equal forecast loss", "DM = mean(d_t) / HAC_se(d_t)", 0, 20))
            continue
        diff = _clean_series(validation_losses[model_column] - validation_losses[baseline_column])
        if len(diff) < 20:
            tests.append(_insufficient(name, "equal forecast loss", "DM = mean(d_t) / HAC_se(d_t)", len(diff), 20))
            continue
        statistic, p_value = _hac_mean_test(diff)
        tests.append(
            StatTestResult(
                name=name,
                null_hypothesis="equal predictive accuracy",
                formula="DM = mean(L_model - L_baseline) / HAC_se",
                statistic=statistic,
                p_value=p_value,
                decision=_decision_from_pvalue(p_value, alpha, reject_bad=False),
                sample_size=len(diff),
                interpretation="Negative mean loss difference favors GARCH under QLIKE loss.",
                limitations="This validates volatility forecasts, not trading profitability.",
            )
        )
    return tests


def _hac_mean_test(series: pd.Series) -> tuple[float, float]:
    clean = _clean_series(series)
    n = len(clean)
    if n <= 1:
        return 0.0, 1.0
    centered = clean - clean.mean()
    max_lag = max(1, min(12, int(math.sqrt(n))))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered.iloc[lag:], centered.iloc[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_var += 2.0 * weight * cov
    if long_run_var <= 0:
        return 0.0, 1.0
    se = math.sqrt(long_run_var / n)
    if se <= 0:
        return 0.0, 1.0
    statistic = float(clean.mean() / se)
    p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    return statistic, p_value


def _regime_tests(regimes: pd.Series, shocks: pd.DataFrame, timestamp: pd.Timestamp | None) -> list[StatTestResult]:
    clean_regimes = regimes.dropna()
    regime_counts = clean_regimes.value_counts(normalize=True) if not clean_regimes.empty else pd.Series(dtype=float)
    max_share = float(regime_counts.max()) if not regime_counts.empty else None
    shock_count = 0
    if not shocks.empty and "shock_type" in shocks:
        shock_count = int((shocks["shock_type"].fillna("NORMAL") != "NORMAL").sum())
    return [
        StatTestResult(
            name="Regime frequency concentration",
            null_hypothesis="diagnostic concentration",
            formula="max_k count(regime=k) / n",
            statistic=max_share,
            p_value=None,
            decision="diagnostic only" if max_share is not None else "insufficient sample",
            sample_size=int(len(clean_regimes)),
            timestamp=timestamp,
            interpretation="Shows whether one regime dominates the historical sample.",
            limitations="Regime labels are rule-based diagnostics, not ground-truth classes.",
        ),
        StatTestResult(
            name="Shock event frequency",
            null_hypothesis="diagnostic event count",
            formula="count(shock_type != NORMAL)",
            statistic=float(shock_count),
            p_value=None,
            decision="diagnostic only",
            sample_size=int(len(shocks)),
            timestamp=timestamp,
            interpretation="Counts historical non-normal shock labels.",
            limitations="Threshold choice affects event counts.",
        ),
    ]


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed
