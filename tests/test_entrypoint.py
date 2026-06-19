from __future__ import annotations

import subprocess
import sys


def test_help_does_not_launch_tui():
    result = subprocess.run(
        [sys.executable, "-m", "cmva", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert "interactive terminal research app" in result.stdout
