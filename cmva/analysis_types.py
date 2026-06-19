"""Shared analysis result types for methodology and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StatTestResult:
    name: str
    null_hypothesis: str
    formula: str
    statistic: float | None
    p_value: float | None
    decision: str
    sample_size: int
    window: str | None = None
    timestamp: pd.Timestamp | None = None
    interpretation: str = ""
    limitations: str = ""

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "null_hypothesis": self.null_hypothesis,
            "formula": self.formula,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "decision": self.decision,
            "sample_size": self.sample_size,
            "window": self.window,
            "timestamp": self.timestamp,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
        }


@dataclass
class MethodStep:
    timestamp: pd.Timestamp | None
    stage: str
    formula_id: str
    inputs: dict[str, object] = field(default_factory=dict)
    output: object | None = None
    data_cutoff: pd.Timestamp | None = None
    lookahead_status: str = "passed"

    def to_record(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "formula_id": self.formula_id,
            "inputs": self.inputs,
            "output": self.output,
            "data_cutoff": self.data_cutoff,
            "lookahead_status": self.lookahead_status,
        }


@dataclass
class DiagnosticSnapshot:
    model_tests: list[StatTestResult] = field(default_factory=list)
    forecast_tests: list[StatTestResult] = field(default_factory=list)
    risk_tests: list[StatTestResult] = field(default_factory=list)
    backtest_tests: list[StatTestResult] = field(default_factory=list)
    regime_tests: list[StatTestResult] = field(default_factory=list)
    method_steps: list[MethodStep] = field(default_factory=list)
    generated_at: pd.Timestamp | None = None

    @property
    def all_tests(self) -> list[StatTestResult]:
        return [
            *self.model_tests,
            *self.forecast_tests,
            *self.risk_tests,
            *self.backtest_tests,
            *self.regime_tests,
        ]
