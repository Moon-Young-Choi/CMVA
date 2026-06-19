# CMVA

CMVA — Crypto Market Volatility Analysis — is an interactive terminal research app for public crypto market data.

Run it with:

```bash
python3 -m cmva
```

or after installation:

```bash
cmva
```

CMVA fetches Binance Spot public 1h candles from Binance public market-data endpoints, uses only closed candles for research calculations, computes rolling market-risk features, forecasts volatility with GARCH, classifies shock/regime state, and evaluates simulated volatility-targeting policies. It is not a trading bot, does not place orders, and does not require private API keys.
