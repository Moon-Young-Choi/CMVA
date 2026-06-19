# CMVA

CMVA, Crypto Market Volatility Analysis, is a localhost web research dashboard for public crypto market-state analytics and model validation.

It uses Binance Spot public candles to analyze volatility, trend, correlation/PCA common-risk structure, shock labels, and regimes at a user-selected candle interval. Backtesting in CMVA means walk-forward model validation, not trading strategy PnL simulation.

CMVA does not place orders, generate trading strategies, request private API keys, access exchange accounts, or implement futures/margin/leverage trading.
The product surface is the local UI itself: accumulated closed-candle data, validation state, and statistical market-state analysis.

## Run

Recommended:

```bash
./run_cmva.sh
```

Manual setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m cmva
```

Installed command:

```bash
cmva
```

By default CMVA starts a local FastAPI/Uvicorn server and opens:

```text
http://127.0.0.1:8765
```

Headless run:

```bash
python -m cmva --no-browser
```

Legacy Textual fallback:

```bash
python -m cmva --tui
```

## Data Policy

- Binance Spot public market data only.
- Default candle interval: `15m`.
- Supported MVP intervals: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`.
- REST klines provide historical bootstrap.
- WebSocket kline streams update live state.
- Only closed candles enter storage, features, models, shock/regime classification, diagnostics, and validation.

## Interval And Window Policy

Rolling windows are configured as time durations and converted to bars for the selected interval.

Defaults:

- volatility window: `24h`
- correlation window: `7d`
- PCA window: `30d`
- trend window: `24h`
- regime threshold window: `90d`
- forecast horizon: `1 bar`

Example:

```text
Interval: 15m
Forecast horizon: 1 bar = next 15 minutes
Volatility window: 24h = 96 bars
```

## Validation

CMVA compares volatility models with walk-forward forecast loss:

- GARCH(1,1)-Student-t
- EWMA volatility
- naive previous realized volatility

Metrics include RMSE, MAE, QLIKE, forecast-realized correlation, calibration by decile, realized volatility by regime, and residual diagnostics.

## Data Accumulation UI

The Markets page shows accumulated closed-candle rows, symbol-level first/latest candle times, coverage, validation issue counts, latest closed candles, and recent raw candle rows. The goal is to let the user inspect the dataset CMVA is building before interpreting model outputs.

## Verify

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q cmva tests
python3 -m cmva --help
```
