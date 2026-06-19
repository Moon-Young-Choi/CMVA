# CMVA

CMVA — Crypto Market Volatility Analysis — is an interactive terminal research app for public crypto market data.

## Run

Recommended:

```bash
./run_cmva.sh
```

The script creates/uses `.venv`, installs dependencies if needed, then opens the TUI.

If you prefer manual setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m cmva
```

After activating the virtual environment, the console command also works:

```bash
source .venv/bin/activate
cmva
```

If `python3 -m cmva` fails with `ModuleNotFoundError: No module named 'pandas'`, you are using the system Python instead of the project virtual environment. Use `./run_cmva.sh` or `.venv/bin/python -m cmva`.

CMVA fetches Binance Spot public 1h candles from Binance public market-data endpoints, uses only closed candles for research calculations, computes rolling market-risk features, forecasts volatility with GARCH, classifies shock/regime state, and evaluates simulated volatility-targeting policies. It is not a trading bot, does not place orders, and does not require private API keys.
