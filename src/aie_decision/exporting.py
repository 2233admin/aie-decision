"""Deterministic machine and human projections for analysis packages."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    """Convert domain values to stable JSON-compatible values."""
    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    if isinstance(value, set):
        return sorted((to_primitive(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    return value


def package_json(package: Any) -> str:
    """Render a reproducible machine projection without mutating the package."""
    return json.dumps(to_primitive(package), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_package_json(package: Any, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(package_json(package), encoding="utf-8", newline="\n")
    return path


def _section(title: str, value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    lines = [f"## {title}", ""]
    if isinstance(value, str):
        lines.extend([value, ""])
    else:
        lines.extend(["```json", json.dumps(to_primitive(value), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    return lines


def human_report(package: Any) -> str:
    """Render a report that keeps facts, judgments, gaps, and conclusions apart."""
    data = to_primitive(package)
    if not isinstance(data, Mapping):
        raise TypeError("analysis package must project to an object")
    schema_version = data.get("schema_version", "unknown")
    run_id = data.get("run_id") or data.get("id", "unknown")
    lines = ["# AIE Decision Analysis", "", f"- Run: `{run_id}`", f"- Schema: `{schema_version}`", ""]
    for title, keys in (
        ("Answer contract", ("answer_contract",)),
        ("Supported evidence and reconstructed facts", ("evidence_propositions", "evidence", "reconstructed_scene")),
        ("Assumptions and estimates", ("missing_conditions", "condition_estimates", "assumptions")),
        ("Contradictions and unresolved gaps", ("contradictions", "omissions", "blockers")),
        ("Derived-factor hypotheses", ("derived_factor_candidates", "derived_factors")),
        ("Forecast interval audit", ("interval_audit", "forecast_interval_evaluation")),
        ("Conclusion", ("conclusion",)),
    ):
        selected = {key: data[key] for key in keys if key in data and data[key] not in (None, "", [], {})}
        if selected:
            lines.extend(_section(title, selected if len(selected) > 1 else next(iter(selected.values()))))
    if len(lines) == 5:
        lines.extend(["## Partial result", "", "No answer was produced. Inspect the machine package for the failure stage and retry guidance.", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_human_report(package: Any, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(human_report(package), encoding="utf-8", newline="\n")
    return path
