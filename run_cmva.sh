#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "[CMVA] Creating local virtual environment at .venv"
  if ! "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"; then
    "$PYTHON_BIN" -m venv --without-pip "$ROOT_DIR/.venv"
  fi
fi

if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "[CMVA] Installing pip inside .venv"
  GET_PIP="/tmp/cmva-get-pip.py"
  if command -v wget >/dev/null 2>&1; then
    wget -q https://bootstrap.pypa.io/get-pip.py -O "$GET_PIP"
  else
    "$PYTHON_BIN" - <<'PY'
from urllib.request import urlretrieve
urlretrieve("https://bootstrap.pypa.io/get-pip.py", "/tmp/cmva-get-pip.py")
PY
  fi
  "$VENV_PY" "$GET_PIP"
fi

if ! "$VENV_PY" - <<'PY' >/dev/null 2>&1
import arch
import fastapi
import httpx
import jinja2
import numpy
import pandas
import pyarrow
import multipart
import scipy
import sklearn
import statsmodels
import textual
import uvicorn
import websockets
import yaml
PY
then
  echo "[CMVA] Installing project dependencies"
  "$VENV_PY" -m pip install -e '.[dev]'
fi

exec "$VENV_PY" -m cmva "$@"
