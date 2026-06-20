from __future__ import annotations

from fastapi.testclient import TestClient

from cmva.app import CMVAApplication
from cmva.config import CMVAConfig
from cmva.web.app import create_web_app


def test_simulation_routes_and_required_input_validation(tmp_path):
    cmva = CMVAApplication(CMVAConfig(data_dir=tmp_path / "data"))
    client = TestClient(create_web_app(cmva, start_background=False))

    assert client.get("/simulation/new").status_code == 200
    assert "새 시뮬레이션" in client.get("/simulation/new").text
    assert client.get("/simulations").status_code == 200

    response = client.post(
        "/api/simulations",
        json={
            "run_name": "bad",
            "symbols": "BTCUSDT",
            "interval": "1h",
            "data_start": "2026-01-01T00:00:00Z",
            "data_end": "2026-01-02T00:00:00Z",
            "T": "12h",
            "dT": "1 bars",
        },
        headers={"accept": "application/json"},
    )

    assert response.status_code == 400
    assert "S는 필수" in response.text


def test_simulation_live_js_and_tooltip_do_not_reload_pages():
    live_js = open("cmva/web/static/js/live.js", encoding="utf-8").read()
    simulation_js = open("cmva/web/static/js/simulation.js", encoding="utf-8").read()
    tooltip_js = open("cmva/web/static/js/tooltip.js", encoding="utf-8").read()
    tooltip_css = open("cmva/web/static/css/tooltip.css", encoding="utf-8").read()

    assert "window.location.reload" not in live_js
    assert "scheduleReload" not in live_js
    assert "window.location.reload" not in simulation_js
    assert "activeAnchor" in tooltip_js
    assert "hoveringTooltip" in tooltip_js
    assert "pointer-events: auto" in tooltip_css
