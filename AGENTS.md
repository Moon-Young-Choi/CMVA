# CMVA Agent Guide

이 문서는 CMVA를 수정하는 Codex/agent가 반드시 지켜야 하는 현재 프로젝트 기준입니다. CMVA는 여러 CLI 명령어 묶음이 아니라, `python -m cmva` 또는 `cmva` 하나로 실행되는 Textual 기반 interactive terminal research app입니다.

## 1. Product Mission

CMVA, Crypto Market Volatility Analysis, 는 Binance Spot 공개 1시간봉 데이터를 사용해 암호화폐 시장의 변동성, 충격, 국면, simulated exposure, backtest 성과를 실시간으로 관찰하는 연구용 TUI 앱입니다.

핵심 원칙:

- 실제 주문, private key, 계좌 접근, 선물/레버리지 거래는 절대 포함하지 않는다.
- closed candle만 연구 계산에 사용한다.
- forecast는 가격 방향 예측이 아니라 다음 1시간 변동성 예측이다.
- 모든 feature, forecast, regime, shock, exposure, backtest는 look-ahead bias 없이 계산한다.
- Dashboard/Forecast/Backtest time range는 서로 독립적으로 설정 가능해야 한다.

## 2. Runtime Contract

실행:

```bash
./run_cmva.sh
```

또는:

```bash
.venv/bin/python -m cmva
```

설치된 환경에서는:

```bash
cmva
```

검증:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q cmva tests
python3 -m cmva --help
```

현재 기본 데이터 정책:

- Binance Spot public data only
- candle interval: `1h`
- forecast horizon: `1h`
- historical bootstrap: `365` days
- default universe: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`
- dashboard range: `1d`
- forecast diagnostic range: `1w`
- backtest evaluation range: `1y`

## 3. Non-Negotiable Invariants

Closed-candle discipline:

- REST historical klines are closed candles.
- WebSocket current candles may be displayed, but must not be persisted or used in calculations until Binance kline `x=True`.
- Never forward-fill missing prices silently.

No-look-ahead discipline:

- Feature at `t` uses data through `t`.
- Forecast used for shock scoring must be `sigma_{t|t-1}` against `r_t`.
- Exposure decided at `t` is applied to return at `t+1`.
- Backtest cost timing is `R_{t+1} = w_t r_{t+1} - c |w_t - w_{t-1}|`.
- Rolling/expanding thresholds must not use full-sample future data.

Forecast meaning:

```text
forecast_vol_1h = sigma_{t+1|t}
```

This is volatility, not price direction.

Range meaning:

- `forecast_horizon` is the mathematical prediction horizon and remains `1h`.
- `dashboard_time_range` is the visual Dashboard graph window.
- `forecast_time_range` is the forecast/stat-test display and diagnostic window.
- `backtest_time_range` is the historical backtest/statistical evaluation sample.

Supported ranges:

```text
1d, 1w, 1m, 3m, 6m, 1y, all
```

Custom examples such as `12h`, `10d`, and `4w` are valid.

## 4. Component Map

Top-level orchestration:

- `cmva/__main__.py`: user entrypoint and helpful dependency/venv guidance.
- `cmva/app.py`: application orchestration, bootstrap, recompute, range-specific views, report export.
- `cmva/config.py`: defaults and config loading.
- `cmva/state.py`: app state shared by services and TUI.
- `cmva/analysis_types.py`: `StatTestResult`, `MethodStep`, `DiagnosticSnapshot`.
- `cmva/time_ranges.py`: range parsing, normalization, slicing, and range metadata.
- `cmva/logging_config.py`: file-based logging that must not pollute TUI output.

Data layer:

- `cmva/data/rest_client.py`: Binance Spot REST klines.
- `cmva/data/websocket_client.py`: Binance kline stream.
- `cmva/data/candle.py`: `Candle` dataclass and frame conversion.
- `cmva/data/storage.py`: parquet storage under `data/cleaned`.
- `cmva/data/validation.py`: continuity, duplicates, OHLC, volume, coverage checks.

Feature layer:

- `cmva/features/returns.py`: close matrix, log returns, equal-weight basket return.
- `cmva/features/volatility.py`: realized and EWMA volatility.
- `cmva/features/correlation.py`: rolling average pairwise correlation.
- `cmva/features/rolling_beta.py`: rolling BTC beta.
- `cmva/features/pca.py`: rolling covariance PCA1 share.
- `cmva/features/dispersion.py`: cross-sectional dispersion.

Model and diagnostics layer:

- `cmva/models/garch.py`: GARCH(1,1) Student-t with EWMA fallback.
- `cmva/models/diagnostics.py`: Ljung-Box, ARCH-LM, forecast loss, DM, Mincer-Zarnowitz, Kupiec, HAC, bootstrap diagnostics.
- `cmva/models/selection.py`: model selection status, currently GARCH only.
- `cmva/forecast/volatility_forecaster.py`: asset and basket forecast service.

Regime and policy layer:

- `cmva/regime/shock.py`: `NORMAL`, `MODERATE_SHOCK`, `IDIOSYNCRATIC_VOL_SHOCK`, `SYSTEMIC_VOL_SHOCK`, `VOL_REGIME_BUILDUP`.
- `cmva/regime/classifier.py`: market regime classifier.
- `cmva/regime/thresholds.py`: rolling/expanding thresholds.
- `cmva/policy/vol_target.py`: naive volatility target exposure.
- `cmva/policy/regime_vol_target.py`: regime-aware exposure multiplier policy.

Backtest layer:

- `cmva/backtest/engine.py`: historical walk-forward backtest.
- `cmva/backtest/live_paper.py`: live paper settlement timing.
- `cmva/backtest/benchmarks.py`: BTC and equal-weight benchmarks.
- `cmva/backtest/costs.py`: turnover and cost helpers.
- `cmva/backtest/metrics.py`: performance metrics and drawdown.

TUI layer:

- `cmva/tui/app.py`: Textual app composition, settings form, buttons, refresh loop.
- `cmva/tui/screens.py`: tab render dispatch.
- `cmva/tui/widgets.py`: dashboard, methodology, diagnostics, backtest, process renderables.
- `cmva/tui/graphs.py`: terminal-native time-series graph panels.
- `cmva/tui/bindings.py`: keyboard shortcuts.
- `cmva/tui/theme.py`: TUI CSS.

Report layer:

- `cmva/reports/markdown.py`: Markdown report with methodology, diagnostics, process, range policy.
- `cmva/reports/html.py`: HTML report derived from the Markdown content.
- `cmva/reports/plots.py`: simple SVG equity export.

Tests:

- Tests must not require internet.
- Use synthetic candles and mocked data.
- Keep `pytest` green after every behavior change.

## 5. TUI Tabs And UX Contract

Tabs:

```text
Dashboard | Data | Features | Models | Methodology | Stat Tests | Regime | Backtest | Process | Settings | Logs
```

Dashboard must show:

- latest closed candle
- WebSocket status
- current regime and shock
- forecast vol 1h
- target exposure
- live paper PnL
- data interval, forecast horizon, dashboard range, forecast range, backtest range
- graph panels for basket return, forecast volatility, target exposure, and shock/regime score

Methodology must show:

- formulas for log return, rolling volatility, EWMA, correlation, beta, covariance PCA1, GARCH, shock score, exposure, backtest return
- Range Policy explaining interval, forecast horizon, display ranges, and evaluation ranges
- look-ahead discipline
- process steps

Stat Tests must show:

- active forecast diagnostic range start/end/sample count
- model diagnostics
- forecast evaluation
- risk coverage
- backtest inference
- regime/shock validation

Backtest must show:

- selected backtest range start/end/sample count
- cumulative return, max drawdown, Sharpe, turnover
- equity curve graph
- drawdown graph
- strategy vs benchmark return graph

Settings must be scrollable and expose:

- symbols
- rolling windows
- target annual vol and max leverage
- transaction cost and slippage
- GARCH refit frequency and shock thresholds
- dashboard, forecast, and backtest ranges as three comma-separated values

Logs must be scrollable and must be the user-facing place for runtime messages. `httpx` and WebSocket library logs should not print over the TUI.

Keyboard bindings:

```text
left/right  tab navigation
space       pause/resume
r           refresh
f           force GARCH refit
b           rerun backtest
e           export report
q           quit
```

## 6. Statistical Methodology Contract

Use percent returns internally for GARCH numerical stability, then convert forecasts back to decimal volatility.

GARCH:

```text
r_t = mu + eps_t
eps_t = sigma_t z_t
h_t = omega + alpha eps_{t-1}^2 + beta h_{t-1}
forecast = sigma_{t+1|t}
```

Shock score:

```text
shock_score_t = |r_t| / sigma_{t|t-1}
```

Target exposure:

```text
base_exposure_t = target_vol_per_hour / forecast_vol_{t+1|t}
target_exposure_t = clip(base_exposure_t * regime_multiplier, 0, max_leverage)
```

Backtest return:

```text
R_{t+1} = w_t r_{t+1} - cost * |w_t - w_{t-1}|
```

Regime multipliers:

```text
SYSTEMIC_RISK           0.20
IDIOSYNCRATIC_HIGH_VOL  0.50
ASSET_SPECIFIC          1.00
QUIET_CORRELATED        0.70
```

Diagnostic results must include:

- null hypothesis
- formula
- statistic
- p-value when applicable
- decision
- sample size
- interpretation
- limitations

Never present p-values as investment proof. They are diagnostics only.

## 7. Artifacts And Git Hygiene

Generated artifacts live under:

```text
data/raw
data/cleaned
data/features
data/models
data/backtests
reports
logs
```

Generated data, reports, logs, and caches should remain gitignored.

Do not revert user changes unless explicitly requested. If the worktree is dirty, inspect first and preserve unrelated changes.

## 8. Acceptance Criteria

CMVA is acceptable when:

1. `python3 -m cmva --help` works.
2. `python -m cmva`, `.venv/bin/python -m cmva`, or `./run_cmva.sh` opens the TUI.
3. The app fetches/stores Binance Spot public 1h closed candles.
4. Current unclosed candles can be displayed but are excluded from research calculations.
5. Dashboard, Methodology, Stat Tests, Backtest, Process, Settings, and Logs tabs render useful content.
6. Dashboard/Forecast/Backtest ranges can be changed independently.
7. Backtest metrics are recomputed for the selected backtest range.
8. Statistical diagnostics handle insufficient samples gracefully.
9. Reports include methodology, diagnostics, range policy, limitations, and no-trading disclaimers.
10. All tests pass offline.

