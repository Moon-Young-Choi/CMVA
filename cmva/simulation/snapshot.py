"""Snapshot assembly for simulation API and WebSocket payloads."""

from __future__ import annotations

from typing import Any

from cmva.simulation.repository import SimulationRepository


def build_simulation_snapshot(repository: SimulationRepository, run_id: str, limit: int = 80) -> dict[str, Any]:
    run = repository.load_run(run_id)
    partial = repository.load_partial_results(run_id, limit=limit)
    return {
        "run_id": run_id,
        "spec": run.get("spec", {}),
        "progress": run.get("progress", {}),
        "data_validation": run.get("data_validation", {}),
        "warnings": run.get("warnings", []),
        **partial,
    }
