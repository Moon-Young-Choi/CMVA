"""Candle data validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cmva.data.candle import closed_only, normalize_candle_frame


@dataclass
class ValidationIssue:
    severity: str
    check: str
    message: str
    symbol: str | None = None
    count: int = 0


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    symbol_rows: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def count(self, check: str) -> int:
        return sum(issue.count for issue in self.issues if issue.check == check)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.__dict__ for issue in self.issues],
            "symbol_rows": self.symbol_rows,
        }

    def to_markdown(self) -> str:
        lines = ["# CMVA Data Validation", "", f"Valid: `{self.is_valid}`", ""]
        if not self.issues:
            lines.append("No validation issues detected.")
            return "\n".join(lines)
        lines.extend(["| severity | check | symbol | count | message |", "| --- | --- | --- | ---: | --- |"])
        for issue in self.issues:
            lines.append(
                f"| {issue.severity} | {issue.check} | {issue.symbol or ''} | {issue.count} | {issue.message} |"
            )
        return "\n".join(lines)


def validate_candles(
    frame: pd.DataFrame,
    interval: str = "1h",
    symbols: list[str] | None = None,
    outlier_z: float = 8.0,
) -> ValidationReport:
    report = ValidationReport()
    if frame.empty:
        report.issues.append(ValidationIssue("error", "empty", "no candle rows available"))
        return report

    data = normalize_candle_frame(frame)
    report.symbol_rows = data.groupby("symbol").size().astype(int).to_dict()
    expected_symbols = [symbol.upper() for symbol in symbols] if symbols else sorted(data["symbol"].unique())

    unclosed = data.loc[~data["is_closed"]]
    if not unclosed.empty:
        report.issues.append(
            ValidationIssue("error", "unclosed", "unclosed candles cannot enter research data", count=len(unclosed))
        )
    data = closed_only(data)

    missing_symbols = sorted(set(expected_symbols) - set(data["symbol"].unique()))
    for symbol in missing_symbols:
        report.issues.append(ValidationIssue("error", "coverage", "missing symbol data", symbol=symbol, count=1))

    duplicates = data.duplicated(["symbol", "interval", "open_time"], keep=False)
    if duplicates.any():
        report.issues.append(
            ValidationIssue("error", "duplicates", "duplicate candle timestamps detected", count=int(duplicates.sum()))
        )

    bad_ohlc = data.loc[
        (data["high"] < data[["open", "close", "low"]].max(axis=1))
        | (data["low"] > data[["open", "close", "high"]].min(axis=1))
        | (data[["open", "high", "low", "close"]] <= 0).any(axis=1)
    ]
    if not bad_ohlc.empty:
        report.issues.append(ValidationIssue("error", "ohlc", "invalid OHLC candle logic", count=len(bad_ohlc)))

    bad_volume = data.loc[data["volume"] < 0]
    if not bad_volume.empty:
        report.issues.append(ValidationIssue("error", "volume", "negative volume detected", count=len(bad_volume)))

    expected_delta = pd.Timedelta(interval)
    for symbol, group in data.groupby("symbol"):
        ordered = group.sort_values("open_time")
        gaps = ordered["open_time"].diff().dropna()
        missing_count = int((gaps > expected_delta).sum())
        if missing_count:
            report.issues.append(
                ValidationIssue("warning", "missing", "missing candle gap detected", symbol=symbol, count=missing_count)
            )

        returns = np.log(ordered["close"]).diff()
        std = returns.std(skipna=True)
        if pd.notna(std) and std > 0:
            outliers = (returns.sub(returns.mean(skipna=True)).abs() / std) > outlier_z
            if outliers.fillna(False).any():
                report.issues.append(
                    ValidationIssue("warning", "outlier", "large return outlier detected", symbol=symbol, count=int(outliers.sum()))
                )

    return report


def reject_unclosed(frame: pd.DataFrame) -> pd.DataFrame:
    data = normalize_candle_frame(frame)
    if (~data["is_closed"]).any():
        raise ValueError("unclosed candles cannot enter the research pipeline")
    return data
