"""JSON-friendly joint-schema validation for the wave-loop entry point.

This module validates the ``outcome_space``, ``variable_specs``,
``mapping_specs``, ``decision_policy``, and ``budget`` sections of the
wave-loop input payload.  It depends only on :mod:`wave_loop_contract`
for its error and config types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .wave_loop_contract import (
    JOINT_SCHEMA_VERSION,
    WaveLoopConfig,
    WaveLoopError,
)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WaveLoopError(f"{field} must be a non-empty string")
    return value.strip()


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WaveLoopError(f"{field} must be a finite number")
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        raise WaveLoopError(f"{field} must be finite")
    return result


def _validate_run_id(payload: Mapping[str, Any]) -> str:
    return _require_string(payload.get("run_id"), "run_id")


def _validate_outcome_space(
    payload: Mapping[str, Any], issues: list[str]
) -> tuple[dict[str, dict[str, Any]], ...]:
    raw = payload.get("outcome_space")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise WaveLoopError("outcome_space must be a non-empty array")
    axes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"$.outcome_space[{index}]"
        if not isinstance(item, Mapping):
            raise WaveLoopError(f"{path} must be an object")
        axis_id = _require_string(item.get("axis_id"), f"{path}.axis_id")
        if axis_id in seen:
            raise WaveLoopError(f"{path}.axis_id must be unique: {axis_id}")
        seen.add(axis_id)
        _require_string(item.get("name"), f"{path}.name")
        _require_string(item.get("unit"), f"{path}.unit")
        absolute = item.get("absolute_tolerance")
        if absolute is not None:
            _require_number(absolute, f"{path}.absolute_tolerance")
        if not isinstance(item.get("decision_useful"), (bool, type(None))):
            raise WaveLoopError(f"{path}.decision_useful must be a boolean when present")
        axes.append(
            {
                "axis_id": axis_id,
                "name": str(item["name"]).strip(),
                "unit": str(item["unit"]).strip(),
                "absolute_tolerance": float(absolute) if absolute is not None else None,
                "reference_value": item.get("reference_value"),
                "decision_useful": bool(item.get("decision_useful", True)),
                "loss_function": item.get("loss_function"),
            }
        )
    return tuple(axes)


def _validate_variable_specs(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("variable_specs")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise WaveLoopError("variable_specs must be a non-empty array")
    seen: set[str] = set()
    specs: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        path = f"$.variable_specs[{index}]"
        if not isinstance(item, Mapping):
            raise WaveLoopError(f"{path} must be an object")
        name = _require_string(item.get("name"), f"{path}.name")
        if name in seen:
            raise WaveLoopError(f"{path}.name must be unique: {name}")
        seen.add(name)
        unit = item.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise WaveLoopError(f"{path}.unit must be a string when present")
        status = _require_string(item.get("status", "bounded"), f"{path}.status")
        if status not in {"observed", "derived", "estimated", "bounded", "missing"}:
            raise WaveLoopError(f"{path}.status must be a known variable status")
        lower = item.get("lower")
        upper = item.get("upper")
        if lower is None or upper is None:
            if status != "missing":
                raise WaveLoopError(f"{path}: missing variables must declare status=missing")
            bounds = (float("-inf"), float("inf"))
        else:
            bounds = (
                _require_number(lower, f"{path}.lower"),
                _require_number(upper, f"{path}.upper"),
            )
            if bounds[0] > bounds[1]:
                raise WaveLoopError(f"{path}.lower must not exceed upper")
        evidence = item.get("evidence_atom_id")
        if evidence is not None and not isinstance(evidence, str):
            raise WaveLoopError(f"{path}.evidence_atom_id must be a string when present")
        specs[name] = {
            "name": name,
            "unit": unit,
            "status": status,
            "lower": bounds[0],
            "upper": bounds[1],
            "method": str(item.get("method", "user_supplied_90_percent_interval")),
            "evidence_atom_id": evidence,
        }
    return specs


def _validate_mapping_specs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("mapping_specs")
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise WaveLoopError("mapping_specs must be an array")
    mappings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"$.mapping_specs[{index}]"
        if not isinstance(item, Mapping):
            raise WaveLoopError(f"{path} must be an object")
        mapping_id = _require_string(item.get("mapping_id"), f"{path}.mapping_id")
        if mapping_id in seen:
            raise WaveLoopError(f"{path}.mapping_id must be unique: {mapping_id}")
        seen.add(mapping_id)
        variables = item.get("variable_names")
        if (
            not isinstance(variables, Sequence)
            or isinstance(variables, (str, bytes))
            or not variables
        ):
            raise WaveLoopError(f"{path}.variable_names must be a non-empty array")
        for var_index, var_name in enumerate(variables):
            if not isinstance(var_name, str) or not var_name.strip():
                raise WaveLoopError(
                    f"{path}.variable_names[{var_index}] must be a string"
                )
        formula = _require_string(item.get("formula"), f"{path}.formula")
        applicability = item.get("applicability")
        if applicability is not None and not isinstance(applicability, str):
            raise WaveLoopError(f"{path}.applicability must be a string when present")
        evidence = item.get("evidence_atom_ids")
        if evidence is not None:
            if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
                raise WaveLoopError(f"{path}.evidence_atom_ids must be an array")
            for atom_index, atom_id in enumerate(evidence):
                if not isinstance(atom_id, str) or not atom_id.strip():
                    raise WaveLoopError(
                        f"{path}.evidence_atom_ids[{atom_index}] must be a string"
                    )
        mappings.append(
            {
                "mapping_id": mapping_id,
                "variable_names": tuple(str(name).strip() for name in variables),
                "formula": formula,
                "applicability": applicability,
                "evidence_atom_ids": tuple(evidence or ()),
                "direction": str(item.get("direction", "support")),
            }
        )
    return mappings


def _validate_decision_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("decision_policy")
    if raw is None:
        return {
            "relative_tolerance": 0.25,
            "min_effective_sample_size": 0.1,
            "min_action_benefit": 0.0,
            "residual_interaction_threshold": 0.05,
        }
    if not isinstance(raw, Mapping):
        raise WaveLoopError("decision_policy must be an object")
    policy: dict[str, Any] = {}
    relative = raw.get("relative_tolerance", 0.25)
    policy["relative_tolerance"] = _require_number(
        relative, "decision_policy.relative_tolerance"
    )
    if not 0 <= policy["relative_tolerance"] <= 1:
        raise WaveLoopError(
            "decision_policy.relative_tolerance must be between 0 and 1"
        )
    ess = raw.get("min_effective_sample_size", 0.1)
    policy["min_effective_sample_size"] = _require_number(
        ess, "decision_policy.min_effective_sample_size"
    )
    if not 0 <= policy["min_effective_sample_size"] <= 1:
        raise WaveLoopError(
            "decision_policy.min_effective_sample_size must be between 0 and 1"
        )
    benefit = raw.get("min_action_benefit", 0.0)
    policy["min_action_benefit"] = _require_number(
        benefit, "decision_policy.min_action_benefit"
    )
    if policy["min_action_benefit"] < 0:
        raise WaveLoopError(
            "decision_policy.min_action_benefit must be non-negative"
        )
    residual = raw.get("residual_interaction_threshold", 0.05)
    policy["residual_interaction_threshold"] = _require_number(
        residual, "decision_policy.residual_interaction_threshold"
    )
    return policy


def _validate_budget(payload: Mapping[str, Any]) -> WaveLoopConfig:
    raw = payload.get("budget", {})
    if not isinstance(raw, Mapping):
        raise WaveLoopError("budget must be an object when present")
    return WaveLoopConfig(
        max_rounds=max(1, int(raw.get("max_rounds", 1))),
        max_actions=max(1, int(raw.get("max_actions", 5))),
        particle_count=max(2, int(raw.get("particle_count", 128))),
        seed=int(raw.get("seed", 0)),
        started_at=str(raw.get("started_at", "1970-01-01T00:00:00Z")),
    )


def validate_joint_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the JSON-friendly wave input against the narrow joint schema."""

    if not isinstance(payload, Mapping):
        raise WaveLoopError("wave input must be an object")
    run_id = _validate_run_id(payload)
    if payload.get("schema_version") not in (None, JOINT_SCHEMA_VERSION):
        raise WaveLoopError(
            f"unsupported schema_version: expected {JOINT_SCHEMA_VERSION} or absent"
        )
    issues: list[str] = []
    axes = _validate_outcome_space(payload, issues)
    variables = _validate_variable_specs(payload)
    mappings = _validate_mapping_specs(payload)
    decision_policy = _validate_decision_policy(payload)
    budget = _validate_budget(payload)
    for mapping in mappings:
        for var_name in mapping["variable_names"]:
            if var_name not in variables:
                raise WaveLoopError(
                    f"mapping {mapping['mapping_id']} references unknown variable {var_name}"
                )
    return {
        "run_id": run_id,
        "axes": axes,
        "variables": variables,
        "mappings": mappings,
        "decision_policy": decision_policy,
        "budget": budget,
    }


__all__ = [
    "validate_joint_schema",
]
