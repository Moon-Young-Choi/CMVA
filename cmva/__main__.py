"""Command entrypoint for CMVA."""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cmva",
        description="CMVA - localhost crypto market state analytics and model validation dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local web server.")
    parser.add_argument("--port", default=8765, type=int, help="Port for the local web server.")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser.")
    parser.add_argument("--tui", action="store_true", help="Open the legacy Textual TUI fallback instead of the web dashboard.")
    args = parser.parse_args()
    _reexec_with_local_venv_if_available()
    try:
        from cmva.app import run, run_tui
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
    if args.tui:
        run_tui()
    else:
        run(host=args.host, port=args.port, open_browser=not args.no_browser)


def _reexec_with_local_venv_if_available() -> None:
    if sys.prefix != sys.base_prefix:
        return
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists() and os.environ.get("CMVA_NO_VENV_REEXEC") != "1":
        os.execv(str(venv_python), [str(venv_python), "-m", "cmva", *sys.argv[1:]])


if __name__ == "__main__":
    main()
