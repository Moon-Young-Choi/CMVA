"""Select C++ numerical kernels when available, otherwise use Python fallback."""

from __future__ import annotations

import os

from cmva.native.python_backend import PythonBackend


USE_CPP = False
backend = PythonBackend()

if os.environ.get("CMVA_USE_CPP", "0") == "1":
    try:
        import cmva_cpp  # type: ignore

        backend = cmva_cpp  # type: ignore[assignment]
        USE_CPP = True
    except ImportError:
        backend = PythonBackend()
        USE_CPP = False
