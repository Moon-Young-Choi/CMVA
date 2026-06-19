from __future__ import annotations

import subprocess
import sys


def test_help_does_not_launch_server():
    result = subprocess.run(
        [sys.executable, "-m", "cmva", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert "localhost crypto market state analytics" in result.stdout
    assert "--no-browser" in result.stdout
