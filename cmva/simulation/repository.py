"""Filesystem-backed simulation run repository."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmva.simulation.schemas import STATUS_LABELS_KO
from cmva.simulation.spec import SimulationSpec


class SimulationRepository:
    def __init__(self, root: Path | str = Path("data") / "simulations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, spec: SimulationSpec) -> str:
        run_id = spec.run_id or _new_run_id()
        spec.run_id = run_id
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        self._write_json(run_dir / "spec.json", spec.to_dict())
        self.save_progress(
            run_id,
            {
                "run_id": run_id,
                "status": "pending",
                "status_label": STATUS_LABELS_KO["pending"],
                "progress_pct": 0.0,
                "current_step": "대기 중",
                "completed_fits": 0,
                "total_fits": 0,
                "updated_at": pd.Timestamp.now(tz="UTC"),
            },
        )
        self._write_json(run_dir / "warnings.json", [])
        return run_id

    def run_dir(self, run_id: str) -> Path:
        return self.root / str(run_id)

    def update_status(self, run_id: str, status: str, **updates: Any) -> None:
        progress = self.load_progress(run_id)
        progress.update(
            {
                "run_id": run_id,
                "status": status,
                "status_label": STATUS_LABELS_KO.get(status, status),
                "updated_at": pd.Timestamp.now(tz="UTC"),
                **updates,
            }
        )
        self.save_progress(run_id, progress)

    def save_data_validation(self, run_id: str, validation: dict[str, Any]) -> None:
        self._write_json(self.run_dir(run_id) / "data_validation.json", validation)

    def append_step_results(self, run_id: str, records: list[dict[str, Any]]) -> None:
        self._append_frame(self.run_dir(run_id) / "step_results.parquet", records)
        self._append_frame(self.run_dir(run_id) / "model_fits.parquet", records)

    def append_score_results(self, run_id: str, records: list[dict[str, Any]]) -> None:
        self._append_frame(self.run_dir(run_id) / "scores.parquet", records)

    def save_progress(self, run_id: str, progress: dict[str, Any]) -> None:
        self._write_json(self.run_dir(run_id) / "progress.json", progress)

    def save_metric_series(self, run_id: str, series: dict[str, Any]) -> None:
        self._write_json(self.run_dir(run_id) / "metric_series.json", series)

    def save_conclusion(self, run_id: str, conclusion: dict[str, Any]) -> None:
        self._write_json(self.run_dir(run_id) / "conclusion.json", conclusion)

    def save_warnings(self, run_id: str, warnings: list[str]) -> None:
        self._write_json(self.run_dir(run_id) / "warnings.json", warnings)

    def load_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        if not run_dir.exists():
            raise KeyError(f"simulation run not found: {run_id}")
        return {
            "spec": self._read_json(run_dir / "spec.json", {}),
            "progress": self.load_progress(run_id),
            "data_validation": self._read_json(run_dir / "data_validation.json", {}),
            "warnings": self._read_json(run_dir / "warnings.json", []),
        }

    def load_progress(self, run_id: str) -> dict[str, Any]:
        return self._read_json(self.run_dir(run_id) / "progress.json", {})

    def load_partial_results(self, run_id: str, limit: int = 80) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        step_results = self._read_frame(run_dir / "step_results.parquet")
        scores = self._read_frame(run_dir / "scores.parquet")
        return {
            "step_results": _frame_records(step_results, limit),
            "scores": _frame_records(scores, limit),
            "latest_score": _latest_record(scores),
            "metric_series": self._read_json(run_dir / "metric_series.json", {}),
        }

    def load_final_results(self, run_id: str) -> dict[str, Any]:
        run = self.load_run(run_id)
        partial = self.load_partial_results(run_id, limit=10000)
        return {
            **run,
            **partial,
            "conclusion": self._read_json(self.run_dir(run_id) / "conclusion.json", {}),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("*/spec.json"), reverse=True):
            run_id = path.parent.name
            run = self.load_run(run_id)
            runs.append({"run_id": run_id, **run["spec"], "progress": run["progress"]})
        return runs

    def mark_interrupted_runs(self) -> None:
        active = {"pending", "preparing_data", "validating_data", "running"}
        for path in sorted(self.root.glob("*/progress.json")):
            run_id = path.parent.name
            progress = self.load_progress(run_id)
            if progress.get("status") in active:
                self.mark_failed(run_id, "서버가 재시작되어 이전 시뮬레이션 실행이 중단되었습니다. 새 시뮬레이션을 실행하세요.")

    def mark_failed(self, run_id: str, error: str) -> None:
        self.update_status(run_id, "failed", current_step="실패", error=error)

    def mark_completed(self, run_id: str) -> None:
        self.update_status(run_id, "completed", progress_pct=1.0, current_step="완료")

    def _append_frame(self, path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        incoming = pd.DataFrame(records)
        if path.exists():
            existing = pd.read_parquet(path)
            incoming = pd.concat([existing, incoming], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            incoming.to_parquet(path, index=False)
        except Exception:
            incoming.to_csv(path.with_suffix(".csv"), index=False)

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            return pd.read_csv(fallback)
        return pd.DataFrame()

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))


def _new_run_id() -> str:
    now = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
    return f"sim-{now}-{uuid.uuid4().hex[:8]}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "isoformat") and value.__class__.__name__ in {"datetime", "Timestamp"}:
        return value.isoformat()
    return value


def _frame_records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_json_ready(record) for record in frame.tail(limit).to_dict(orient="records")]


def _latest_record(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return _json_ready(frame.tail(1).iloc[0].to_dict())
