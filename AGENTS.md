# CMVA Agent Guide

Build CMVA as a local browser-based crypto market state analytics and model validation dashboard.

Primary command:

```bash
python -m cmva
```

or:

```bash
cmva
```

This command starts a local FastAPI/Uvicorn server bound to `127.0.0.1` and opens:

```text
http://127.0.0.1:8765
```

Use `python -m cmva --no-browser` for headless runs and `python -m cmva --tui` only as a legacy fallback.

## Product Scope

CMVA is not a trading bot.

Do not implement:

- strategy generation
- order execution
- fill simulation
- PnL attribution
- arbitrage opportunity monitoring
- private API keys
- exchange account access
- futures, margin, or leverage trading
- portfolio exposure policy
- target exposure
- paper PnL

Core objective:

CMVA analyzes the current crypto market state using public Binance Spot market data. It computes volatility, trend, correlation, PCA common-risk structure, shock labels, and market regimes using selected closed candle intervals. It validates analysis models using walk-forward model backtesting and statistical diagnostics. Trading strategy design is explicitly out of scope.

## Data And Interval Policy

- Use Binance Spot public klines.
- The candle interval must be user-selectable.
- Supported MVP intervals: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`.
- Default interval: `15m`.
- Advanced `1s` support may exist later but is not required in MVP.
- Use REST klines for historical bootstrap.
- Use WebSocket `kline_<interval>` streams for live updates.

Closed candle rule:

- Current unclosed candles may be displayed.
- Only closed candles may enter storage, validation, feature calculation, model fitting, forecast generation, shock detection, regime classification, and model validation.
- The WebSocket kline `x` field must determine whether the candle is closed.
- Never forward-fill missing prices silently.

Interval policy:

- Do not hardcode `1h`.
- All rolling windows must be defined in time durations, such as `24h`, `7d`, `30d`, `90d`.
- Convert duration windows to bar counts based on the selected interval.
- Display the conversion in the UI.

## Analysis Modules

Volatility:

- realized volatility
- EWMA volatility
- GARCH forecast volatility
- range-based volatility
- volatility percentile
- standardized residual

Trend:

- rolling OLS slope of log price
- slope t-stat
- volatility-adjusted trend strength
- trend consistency
- rolling autocorrelation

Correlation / PCA:

- rolling average pairwise correlation
- rolling BTC beta
- PCA1 explained variance share
- PCA loadings
- cross-sectional dispersion

Shock / Regime:

- shock score
- shock breadth
- realized volatility jump ratio
- current shock label
- current regime label
- regime transition matrix

Data accumulation:

- total closed candle rows
- symbol-level row counts
- symbol-level first/latest closed candle time
- active interval coverage
- recent accumulated candle table
- validation issue counts

## Models And Validation

Models:

- Naive realized volatility baseline
- EWMA volatility baseline
- GARCH(1,1)-Student-t

Forecast definition:

- Forecast means next-period volatility or variance, not price direction.
- If interval is `15m` and horizon is `1 bar`, the forecast means next 15m volatility.
- If interval is `1h` and horizon is `1 bar`, the forecast means next 1h volatility.

Backtesting definition:

- Backtesting means model validation, not trading strategy PnL.
- Do not compute strategy returns.
- Do not compute portfolio weights.
- Do not compute target exposure.
- Do not compute paper PnL.
- Validate whether volatility forecasts, trend classifications, shock labels, and regime labels explain subsequent market behavior.

Validation metrics:

- RMSE
- MAE
- QLIKE
- forecast-realized volatility correlation
- calibration by forecast decile
- realized volatility by regime
- post-shock realized volatility path
- residual autocorrelation diagnostics
- model comparison against EWMA and naive baselines

Diagnostic results must include null hypothesis, formula, statistic, p-value when applicable, decision, sample size, interpretation, and limitations. Never present p-values as investment proof.

## C++ Acceleration

- Use Python for orchestration, web server, I/O, charts, settings, and UI state.
- Use C++ for computational kernels when profiling shows a bottleneck.
- Use pybind11 and scikit-build-core.
- Provide Python fallback implementation.
- C++ and Python results must match within numerical tolerance.
- Initial C++ candidates: log returns, rolling mean/std, EWMA variance, realized volatility, rolling covariance/correlation, rolling OLS slope/t-stat, forecast loss metrics.

## UI Contract

Use FastAPI + Jinja2 templates + vanilla JavaScript. Browser WebSocket can push live dashboard updates. The Textual TUI is not the primary UX.

Pages:

```text
Dashboard
Markets
Volatility
Trend
Correlation / PCA
Shock & Regime
Models
Backtest / Validation
Methodology
Settings
Logs
```

Tooltip requirement:

- Implement tooltip components across the UI.
- A tooltip appears on mouse hover and keyboard focus.
- Use tooltips for nontrivial metrics, models, diagnostics, buttons, and chart legends.
- Manage tooltip content through `cmva/web/glossary.yaml`.
- Each tooltip should include a short explanation and, when useful, a formula.
- Link detailed explanations to the Methodology page.
- Do not hide essential information only inside tooltips.

## Tests

Tests must run offline with synthetic data.

Required coverage:

- selected interval propagates to REST and WebSocket clients
- duration windows convert correctly to bar windows
- unclosed candles are excluded from analysis
- duplicate timestamps are detected
- invalid OHLC candles are rejected
- rolling features do not use future data
- volatility forecasts are shifted correctly
- walk-forward validation does not use full-sample thresholds
- Python numerical backend behavior is tested; C++ kernels must match Python reference outputs when added
- tooltip glossary keys exist for nontrivial UI metrics
- web routes load successfully
- data accumulation UI state is exposed
