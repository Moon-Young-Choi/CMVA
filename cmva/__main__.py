"""Command entrypoint for CMVA."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("CMVA - Crypto Market Volatility Analysis")
        print("")
        print("Usage:")
        print("  python -m cmva")
        print("  cmva")
        print("")
        print("This opens the interactive terminal research app.")
        return
    _reexec_with_local_venv_if_available()
    try:
        from cmva.app import run
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required dependency"
        print(f"CMVA cannot start because `{missing}` is not installed in this Python environment.")
        print("")
        print("Use one of these commands from the project root:")
        print("  ./run_cmva.sh")
        print("  .venv/bin/python -m cmva")
        print("")
        print("If `.venv` does not exist yet, run:")
        print("  ./run_cmva.sh")
        raise SystemExit(1) from None
    run()


def _reexec_with_local_venv_if_available() -> None:
    if sys.prefix != sys.base_prefix:
        return
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists() and os.environ.get("CMVA_NO_VENV_REEXEC") != "1":
        os.execv(str(venv_python), [str(venv_python), "-m", "cmva", *sys.argv[1:]])


if __name__ == "__main__":
    main()
