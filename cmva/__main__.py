"""Command entrypoint for CMVA."""

from __future__ import annotations

import sys

from cmva.app import run


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
    run()


if __name__ == "__main__":
    main()
