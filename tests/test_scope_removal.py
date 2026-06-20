from __future__ import annotations

import importlib.util

from fastapi.testclient import TestClient

from cmva.web.app import create_web_app


def test_obsolete_scope_modules_are_absent():
    absent_modules = [
        "cmva.tui",
        "cmva.reports",
        "cmva.exports",
        "cmva.strategy",
        "cmva.policy",
        "cmva.execution",
        "cmva.fills",
        "cmva.portfolio",
        "cmva.pnl",
    ]

    for module_name in absent_modules:
        assert importlib.util.find_spec(module_name) is None


def test_report_and_export_routes_are_absent():
    client = TestClient(create_web_app(start_background=False))

    for path in ["/reports", "/report", "/exports", "/export", "/api/export", "/api/reports"]:
        assert client.get(path).status_code == 404
