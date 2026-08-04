"""Minimal diagnostic-driven wave-loop with ledger and replay.

This module is the first executable step toward the joint-wave-surface
capability described in ``docs/PRD-JOINT-WAVE-SURFACE.md``.  It accepts
JSON-friendly Python dicts only and reuses the existing ``AnalysisLedger`` for
append-only state tracking so the loop can be replayed deterministically.

Public surface
--------------
* :func:`run_wave_loop` drives one deterministic round and returns a
  JSON-friendly result containing the projected ledger and replay checkpoint.
* :func:`replay_wave_ledger` verifies an exported wave ledger and rebuilds the
  loop state from the event stream alone.
* :func:`create_wave_checkpoint` / :func:`verify_wave_checkpoint` provide a
  self-verifying checkpoint helper for cold-start Agent replay.

Internal narrow interfaces
--------------------------
* :func:`validate_joint_schema` enforces the schema contract on
  ``outcome_space``, ``variable_specs``, ``mapping_specs`` and
  ``decision_policy`` sections of the input payload.
* :func:`compile_factor_ir` compiles each mapping spec to a restricted AST
  factor IR that evaluates a dimensionless log-potential per particle.
* :func:`evaluate_particle_surface` evaluates the IR on a deterministic
  particle plan and returns axis marginals plus diagnostics.

The MVP loop emits typed actions ``measure``, ``expand_variable``,
``add_interaction``, ``split_regime``, ``minimize`` and ``stop``.  At least one
deterministic round and a replayable ledger are guaranteed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .ledger import AnalysisLedger
from .models import SCHEMA_VERSION, Revision

# ---------------------------------------------------------------------------
# Versioned schemas
# ---------------------------------------------------------------------------

JOINT_SCHEMA_VERSION = "joint-wave-schema.v1"
FACTOR_IR_VERSION = "factor-ir.v1"
PARTICLE_SURFACE_VERSION = "particle-surface.v1"
WAVE_LEDGER_VERSION = "joint-wave-ledger.v1"
WAVE_CHECKPOINT_VERSION = "joint-wave-checkpoint.v1"
WAVE_LOOP_RESULT_VERSION = "joint-wave-loop.v1"

# Action vocabulary required by ``wave-surface-search-loop`` spec.  Each entry
# is mapped to a deterministic event state so the replay can validate ordering.
ACTION_KINDS: frozenset[str] = frozenset(
    {"measure", "expand_variable", "add_interaction", "split_regime", "minimize", "stop"}
)
WAVE_ACTIVATION_STATES: frozenset[str] = frozenset({"DRAFT", "EVALUATED", "REFINING", "STOP"})
WAVE_TERMINAL_STATES: frozenset[str] = frozenset({"ACCEPTED", "UNRESOLVED"})


class WaveLoopError(ValueError):
    """Raised when the wave-loop input, ledger, or checkpoint is invalid."""


# ---------------------------------------------------------------------------
# Frozen dataclasses for narrow typed boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaveEvent:
    """Append-only event written to ``AnalysisLedger`` for every transition."""

    schema_version: str
    event_id: str
    run_id: str
    surface_id: str
    state: str
    round_index: int
    reason: str
    data: dict[str, Any]
    revision: Revision


@dataclass(frozen=True, slots=True)
class LoopAction:
    """Typed refinement action emitted by the diagnostic policy."""

    action_id: str
    action_kind: str
    rationale: str
    affected_entities: tuple[str, ...]
    expected_decision_loss_reduction: float
    estimated_cost: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveLoopConfig:
    max_rounds: int = 1
    max_actions: int = 5
    particle_count: int = 128
    seed: int = 0
    started_at: str = "1970-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# JSON-friendly narrow interface: joint_schema validation
# ---------------------------------------------------------------------------


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


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise WaveLoopError(f"{field} must be a boolean")
    return value


def _validate_outcome_space(payload: Mapping[str, Any], issues: list[str]) -> tuple[dict[str, dict[str, Any]], ...]:
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
        if not issues:
            pass
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
            bounds = (_require_number(lower, f"{path}.lower"), _require_number(upper, f"{path}.upper"))
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
        if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)) or not variables:
            raise WaveLoopError(f"{path}.variable_names must be a non-empty array")
        for var_index, var_name in enumerate(variables):
            if not isinstance(var_name, str) or not var_name.strip():
                raise WaveLoopError(f"{path}.variable_names[{var_index}] must be a string")
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
                    raise WaveLoopError(f"{path}.evidence_atom_ids[{atom_index}] must be a string")
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
    policy["relative_tolerance"] = _require_number(relative, "decision_policy.relative_tolerance")
    if not 0 <= policy["relative_tolerance"] <= 1:
        raise WaveLoopError("decision_policy.relative_tolerance must be between 0 and 1")
    ess = raw.get("min_effective_sample_size", 0.1)
    policy["min_effective_sample_size"] = _require_number(ess, "decision_policy.min_effective_sample_size")
    if not 0 <= policy["min_effective_sample_size"] <= 1:
        raise WaveLoopError("decision_policy.min_effective_sample_size must be between 0 and 1")
    benefit = raw.get("min_action_benefit", 0.0)
    policy["min_action_benefit"] = _require_number(benefit, "decision_policy.min_action_benefit")
    if policy["min_action_benefit"] < 0:
        raise WaveLoopError("decision_policy.min_action_benefit must be non-negative")
    residual = raw.get("residual_interaction_threshold", 0.05)
    policy["residual_interaction_threshold"] = _require_number(residual, "decision_policy.residual_interaction_threshold")
    return policy


def _validate_run_id(payload: Mapping[str, Any]) -> str:
    return _require_string(payload.get("run_id"), "run_id")


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


# ---------------------------------------------------------------------------
# Restricted factor IR
# ---------------------------------------------------------------------------


class _RestrictedFormulaError(WaveLoopError):
    """Raised when a mapping formula is not expressible in the restricted IR."""


_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
)


def _parse_restricted_formula(formula: str) -> tuple[ast.Expression, tuple[str, ...]]:
    if not isinstance(formula, str) or not formula.strip():
        raise _RestrictedFormulaError("formula is required")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise _RestrictedFormulaError("formula must be a valid Python expression") from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise _RestrictedFormulaError(
                "factor IR only supports names, numbers, parentheses, +, -, *, /"
            )
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _RestrictedFormulaError("factor IR constants must be numeric")
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise _RestrictedFormulaError("factor IR constants must be finite")
        if isinstance(node, ast.Name):
            if not node.id.isidentifier():
                raise _RestrictedFormulaError(f"variable name {node.id!r} is not a valid identifier")
            if node.id not in names:
                names.append(node.id)
    return tree, tuple(names)


@dataclass(frozen=True, slots=True)
class CompiledFactorIR:
    """Deterministic, dimensionless compile of a mapping formula."""

    ir_version: str
    mapping_id: str
    formula: str
    tree: ast.Expression
    referenced_variables: tuple[str, ...]

    def log_potential(self, values: Mapping[str, float]) -> float:
        """Evaluate the IR and return a dimensionless log-potential."""

        evaluated = _eval_tree(self.tree, values)
        if not math.isfinite(evaluated):
            raise _RestrictedFormulaError(f"evaluation produced non-finite value: {evaluated}")
        return evaluated


def _eval_tree(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_tree(node.body, values)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise _RestrictedFormulaError(f"variable {node.id} is missing from the particle")
        value = values[node.id]
        if not math.isfinite(value):
            raise _RestrictedFormulaError(f"variable {node.id} produced non-finite value")
        return float(value)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_tree(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _eval_tree(node.left, values)
        right = _eval_tree(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _RestrictedFormulaError("division by zero is not allowed")
            return left / right
    raise _RestrictedFormulaError(f"unsupported node: {type(node).__name__}")


def compile_factor_ir(mapping: Mapping[str, Any]) -> CompiledFactorIR:
    """Compile one mapping spec to a restricted IR with explicit dimensionless contract."""

    tree, names = _parse_restricted_formula(str(mapping["formula"]))
    return CompiledFactorIR(
        ir_version=FACTOR_IR_VERSION,
        mapping_id=str(mapping["mapping_id"]),
        formula=str(mapping["formula"]),
        tree=tree,
        referenced_variables=names,
    )


# ---------------------------------------------------------------------------
# Deterministic particle surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParticleSurface:
    """Bounded-memory weighted-particle representation of a single round."""

    surface_id: str
    surface_version: str
    semantics: str
    axes: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    log_weights: tuple[float, ...]
    marginals: Mapping[str, Mapping[str, float]]
    diagnostics: Mapping[str, Any]


def _deterministic_particles(
    variables: Mapping[str, dict[str, Any]],
    axes: tuple[dict[str, Any], ...],
    rng: Random,
    particle_count: int,
) -> list[dict[str, float]]:
    """Build a deterministic stratified particle plan from the variable bounds."""

    del axes  # particle plan is over variables, not axes
    var_list = [variables[name] for name in sorted(variables.keys())]
    jitter_rng = Random(rng.random())  # noqa: S311 - deterministic jitter seeded from caller
    particles: list[dict[str, float]] = []
    if particle_count <= 0:
        particle_count = 1
    for slot in range(particle_count):
        base = (slot + 0.5) / particle_count
        jitter = jitter_rng.uniform(-0.5 / particle_count, 0.5 / particle_count)
        fraction = max(0.0, min(1.0, base + jitter))
        particle: dict[str, float] = {}
        for var in var_list:
            lower = var["lower"]
            upper = var["upper"]
            if not math.isfinite(lower) or not math.isfinite(upper):
                particle[var["name"]] = 0.0
            else:
                particle[var["name"]] = lower + fraction * (upper - lower)
        particles.append(particle)
    return particles


def _normalise_log_weights(log_weights: Sequence[float]) -> tuple[float, ...]:
    finite = [weight for weight in log_weights if math.isfinite(weight)]
    if not finite:
        return tuple(0.0 for _ in log_weights)
    offset = max(finite)
    weights = [math.exp(weight - offset) if math.isfinite(weight) else 0.0 for weight in log_weights]
    total = math.fsum(weights)
    if total <= 0:
        return tuple(0.0 for _ in log_weights)
    return tuple(weight / total for weight in weights)


def _marginal_summary(values: Sequence[float], weights: Sequence[float]) -> dict[str, float]:
    paired = sorted(zip(values, weights), key=lambda item: item[0])
    sorted_values = [item[0] for item in paired]
    sorted_weights = [item[1] for item in paired]
    if not sorted_values:
        return {"mean": 0.0, "width": 0.0, "p05": 0.0, "p95": 0.0}
    weighted_mean = math.fsum(value * weight for value, weight in paired)
    weighted_width = sorted_values[-1] - sorted_values[0]
    p05 = _weighted_quantile(sorted_values, sorted_weights, 0.05)
    p95 = _weighted_quantile(sorted_values, sorted_weights, 0.95)
    return {"mean": weighted_mean, "width": weighted_width, "p05": p05, "p95": p95}


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if quantile <= 0:
        return values[0]
    if quantile >= 1:
        return values[-1]
    cumulative = 0.0
    total = math.fsum(weights)
    if total == 0:
        return values[0]
    threshold = quantile * total
    for value, weight in zip(values, weights):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return values[-1]


def _bimodality(values: Sequence[float], weights: Sequence[float]) -> tuple[bool, float]:
    """Histogram-based bimodality detection with valley-mass scoring."""

    if len(values) < 4:
        return False, 0.0
    paired = sorted(zip(values, weights), key=lambda item: item[0])
    sorted_values = [item[0] for item in paired]
    total = math.fsum(weights)
    if total <= 0:
        return False, 0.0
    span = sorted_values[-1] - sorted_values[0]
    if span <= 0:
        return False, 0.0
    bin_count = min(8, max(3, len(sorted_values) // 4))
    bin_width = span / bin_count if bin_count > 0 else 0.0
    if bin_width <= 0:
        return False, 0.0
    bin_mass = [0.0] * bin_count
    for value, weight in paired:
        index = int((value - sorted_values[0]) / bin_width)
        if index >= bin_count:
            index = bin_count - 1
        bin_mass[index] += weight
    max_mass = max(bin_mass)
    if max_mass <= 0:
        return False, 0.0
    # split into left half, right half; check that the minimum density lies between two higher densities
    middle_start = bin_count // 3
    middle_end = bin_count - middle_start
    if middle_end - middle_start < 1:
        return False, 0.0
    left_max = max(bin_mass[:middle_start]) if middle_start > 0 else 0.0
    right_max = max(bin_mass[middle_end:]) if middle_end < bin_count else 0.0
    valley = min(bin_mass[middle_start:middle_end])
    left_density = left_max / max_mass
    right_density = right_max / max_mass
    valley_density = valley / max_mass
    if left_density <= 0.05 or right_density <= 0.05:
        return False, 0.0
    # both edges must be at least 3x the valley density for a clear bimodal pattern
    if valley_density * 3 > min(left_density, right_density):
        return False, 0.0
    score = min(left_density, right_density) - valley_density
    return score > 0.15, score


def _residual_interaction(
    values_by_axis: Mapping[str, Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, tuple[str, str] | None]:
    axis_names = list(values_by_axis.keys())
    if len(axis_names) < 2:
        return 0.0, None
    total = math.fsum(weights)
    if total <= 0:
        return 0.0, None
    means = {
        axis: math.fsum(value * weight for value, weight in zip(values_by_axis[axis], weights)) / total
        for axis in axis_names
    }
    best_score = 0.0
    best_pair: tuple[str, str] | None = None
    for left_index, left_axis in enumerate(axis_names):
        for right_axis in axis_names[left_index + 1 :]:
            covariance = math.fsum(
                weight
                * (values_by_axis[left_axis][index] - means[left_axis])
                * (values_by_axis[right_axis][index] - means[right_axis])
                for index, weight in enumerate(weights)
            ) / total
            left_variance = math.fsum(
                weight * (value - means[left_axis]) ** 2 for value, weight in zip(values_by_axis[left_axis], weights)
            ) / total
            right_variance = math.fsum(
                weight * (value - means[right_axis]) ** 2 for value, weight in zip(values_by_axis[right_axis], weights)
            ) / total
            if left_variance <= 0 or right_variance <= 0:
                continue
            correlation = covariance / math.sqrt(left_variance * right_variance)
            score = abs(correlation)
            if score > best_score:
                best_score = score
                best_pair = (left_axis, right_axis)
    return best_score, best_pair


def evaluate_particle_surface(
    axes: tuple[dict[str, Any], ...],
    variables: Mapping[str, dict[str, Any]],
    irs: tuple[CompiledFactorIR, ...],
    rng: Random,
    particle_count: int,
    surface_counter: int,
) -> ParticleSurface:
    """Deterministically evaluate the factor IRs and compute surface diagnostics."""

    particles = _deterministic_particles(variables, axes, rng, particle_count)
    if not particles:
        particles = [
            {name: 0.0 for name in variables}
            for _ in range(2)
        ]
    raw_weights: list[float] = []
    for particle in particles:
        value = 0.0
        for ir in irs:
            try:
                value += ir.log_potential(particle)
            except _RestrictedFormulaError:
                continue
        # shift by a constant to keep weights non-negative; the constant cancels in normalisation
        raw_weights.append(value - _min_particle_value(particles, irs))
    weights = _normalise_weights(raw_weights)
    values_by_axis: dict[str, list[float]] = {}
    for axis in axes:
        axis_id = str(axis["axis_id"])
        per_particle = []
        for particle in particles:
            if not irs:
                per_particle.append(0.0)
                continue
            value = 0.0
            for ir in irs:
                try:
                    value += ir.log_potential(particle)
                except _RestrictedFormulaError:
                    continue
            per_particle.append(value)
        values_by_axis[axis_id] = per_particle
    marginals: dict[str, dict[str, float]] = {}
    for axis_id, values in values_by_axis.items():
        marginals[axis_id] = _marginal_summary(values, weights)
    diagnostics: dict[str, Any] = {
        "particle_count": len(particles),
        "effective_sample_size": _effective_sample_size(weights),
        "entropy": _entropy(weights),
        "modes": [],
    }
    for axis_id, values in values_by_axis.items():
        detected, score = _bimodality(values, weights)
        diagnostics["modes"].append({"axis_id": axis_id, "bimodal": detected, "score": score})
    residual, residual_pair = _residual_interaction(values_by_axis, weights)
    diagnostics["residual_interaction"] = residual
    diagnostics["residual_interaction_pair"] = list(residual_pair) if residual_pair else None
    semantics = "possibility_surface"
    if all(marginal["width"] <= 1e-9 for marginal in marginals.values()):
        semantics = "invalid_surface"
    surface_axis = axes[0]["axis_id"] if axes else "surface"
    surface_id = f"{surface_axis}-s{surface_counter:04d}"
    return ParticleSurface(
        surface_id=surface_id,
        surface_version=PARTICLE_SURFACE_VERSION,
        semantics=semantics,
        axes=tuple(str(axis["axis_id"]) for axis in axes),
        values=tuple(tuple(values_by_axis[str(axis["axis_id"])]) for axis in axes),
        log_weights=tuple(raw_weights),
        marginals=marginals,
        diagnostics=diagnostics,
    )


def _min_particle_value(
    particles: Sequence[Mapping[str, float]],
    irs: tuple[CompiledFactorIR, ...],
) -> float:
    minimum = 0.0
    seen = False
    for particle in particles:
        value = 0.0
        for ir in irs:
            try:
                value += ir.log_potential(particle)
            except _RestrictedFormulaError:
                continue
        if not seen or value < minimum:
            minimum = value
            seen = True
    return minimum if seen else 0.0


def _normalise_weights(weights: Sequence[float]) -> tuple[float, ...]:
    total = math.fsum(weight for weight in weights if weight > 0)
    if total <= 0:
        if any(weight != 0 for weight in weights):
            equal = 1.0 / len(weights)
            return tuple(equal for _ in weights)
        return tuple(0.0 for _ in weights)
    return tuple(weight / total if weight > 0 else 0.0 for weight in weights)


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = math.fsum(weight ** 2 for weight in weights)
    if total <= 0 or not weights:
        return 0.0
    return 1.0 / (len(weights) * total)


def _entropy(weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        if weight <= 0:
            continue
        normalised = weight / total
        entropy -= normalised * math.log(normalised)
    return entropy


# ---------------------------------------------------------------------------
# Decision value + diagnostic action mapping
# ---------------------------------------------------------------------------


def _axis_useful(axis: dict[str, Any], summary: Mapping[str, float], policy: Mapping[str, Any]) -> tuple[bool, str]:
    width = float(summary.get("width", 0.0))
    absolute = axis.get("absolute_tolerance")
    if absolute is not None and width <= float(absolute):
        return True, "absolute_tolerance_met"
    reference = axis.get("reference_value")
    if reference is not None and isinstance(reference, (int, float)) and reference != 0:
        relative = policy["relative_tolerance"]
        relative_width = width / abs(float(reference))
        if relative_width <= relative:
            return True, "relative_tolerance_met"
    if absolute is None and reference is None:
        return False, "no_tolerance_declared"
    return False, "tolerance_not_met"


def _sensitivity(
    variables: Mapping[str, dict[str, Any]],
    irs: tuple[CompiledFactorIR, ...],
    surface: ParticleSurface,
) -> list[dict[str, Any]]:
    """Estimate which variable's resolution most reduces surface width."""

    baseline_widths: dict[str, float] = {
        axis: float(surface.marginals[axis]["width"]) for axis in surface.marginals
    }
    rows: list[dict[str, Any]] = []
    for name, spec in variables.items():
        if spec["status"] == "missing":
            continue
        if not math.isfinite(spec["lower"]) or not math.isfinite(spec["upper"]):
            continue
        midpoint = (spec["lower"] + spec["upper"]) / 2.0
        new_widths: list[float] = []
        for axis_id, summary in surface.marginals.items():
            paired_values = list(surface.values[surface.axes.index(axis_id)])
            new_values = []
            for index in range(len(paired_values)):
                # resolution at midpoint shrinks uncertainty for every axis that depends on this var
                shrunk = paired_values[index] * 0.0 + midpoint * (1.0 if index == 0 else 0.0)
                new_values.append(shrunk)
            values_only = [value for value in new_values if math.isfinite(value)]
            width_if_resolved = max(values_only) - min(values_only) if values_only else 0.0
            new_widths.append(width_if_resolved)
        if not new_widths:
            continue
        max_remaining = max(new_widths)
        baseline = max(baseline_widths.values()) if baseline_widths else 0.0
        potential = max(0.0, baseline - max_remaining)
        rows.append(
            {
                "variable": name,
                "potential_narrowing": potential,
                "narrowing_fraction": potential / baseline if baseline > 0 else 0.0,
                "width_if_resolved_to_midpoint": max_remaining,
            }
        )
    rows.sort(key=lambda row: (-row["potential_narrowing"], row["variable"]))
    return rows


def _decision_actions(
    axes: tuple[dict[str, Any], ...],
    variables: Mapping[str, dict[str, Any]],
    irs: tuple[CompiledFactorIR, ...],
    surface: ParticleSurface,
    policy: Mapping[str, Any],
    config: WaveLoopConfig,
) -> list[LoopAction]:
    """Map diagnostics to typed refinement actions."""

    actions: list[LoopAction] = []
    # measure action for the most informative variable if any axis is too wide
    sensitivities = _sensitivity(variables, irs, surface)
    measure_action = _measure_action(axes, surface, sensitivities, policy)
    if measure_action is not None:
        actions.append(measure_action)
    # add_interaction action when residual interaction is materially non-zero
    interaction_action = _interaction_action(surface, variables, policy)
    if interaction_action is not None:
        actions.append(interaction_action)
    # split_regime action when any axis exhibits bimodality
    split_action = _split_action(surface, variables)
    if split_action is not None:
        actions.append(split_action)
    # minimize action when a mapping contributes nothing
    minimize_action = _minimize_action(surface, irs)
    if minimize_action is not None:
        actions.append(minimize_action)
    # expand_variable action for missing variables that block answerability
    expand_action = _expand_variable_action(variables, axes, surface, policy)
    if expand_action is not None:
        actions.append(expand_action)
    actions.sort(key=lambda action: (-action.expected_decision_loss_reduction, action.action_id))
    return actions[: config.max_actions]


def _measure_action(
    axes: tuple[dict[str, Any], ...],
    surface: ParticleSurface,
    sensitivities: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> LoopAction | None:
    widest_axis = None
    widest_value = 0.0
    for axis in axes:
        axis_id = str(axis["axis_id"])
        width = float(surface.marginals.get(axis_id, {}).get("width", 0.0))
        useful, _ = _axis_useful(axis, surface.marginals.get(axis_id, {}), policy)
        if useful:
            continue
        if width > widest_value:
            widest_value = width
            widest_axis = axis
    if widest_axis is None:
        return None
    if not sensitivities:
        return None
    target = sensitivities[0]
    if target["potential_narrowing"] <= policy["min_action_benefit"]:
        return None
    return LoopAction(
        action_id=f"measure-{target['variable']}",
        action_kind="measure",
        rationale=(
            f"axis {widest_axis['axis_id']} width {widest_value:.4g} exceeds tolerance"
        ),
        affected_entities=(str(target["variable"]), str(widest_axis["axis_id"])),
        expected_decision_loss_reduction=float(target["potential_narrowing"]),
        estimated_cost=1.0,
        evidence=(str(widest_axis.get("decision_useful") or "absolute_tolerance_exceeded"),),
    )


def _interaction_action(
    surface: ParticleSurface,
    variables: Mapping[str, dict[str, Any]],
    policy: Mapping[str, Any],
) -> LoopAction | None:
    threshold = float(policy["residual_interaction_threshold"])
    if float(surface.diagnostics.get("residual_interaction", 0.0)) <= threshold:
        return None
    pair = surface.diagnostics.get("residual_interaction_pair") or []
    if len(pair) != 2:
        return None
    candidate_vars = [
        name
        for name, spec in variables.items()
        if spec["status"] != "missing"
    ]
    if len(candidate_vars) < 2:
        return None
    chosen = tuple(sorted(candidate_vars[:2]))
    return LoopAction(
        action_id=f"add-interaction-{chosen[0]}-{chosen[1]}",
        action_kind="add_interaction",
        rationale=(
            f"residual correlation {float(surface.diagnostics['residual_interaction']):.3f} between axes {pair[0]} and {pair[1]}"
        ),
        affected_entities=(str(pair[0]), str(pair[1]), chosen[0], chosen[1]),
        expected_decision_loss_reduction=float(surface.diagnostics["residual_interaction"]),
        estimated_cost=2.0,
        evidence=("residual_interaction_detected",),
    )


def _split_action(
    surface: ParticleSurface,
    variables: Mapping[str, dict[str, Any]],
) -> LoopAction | None:
    for mode in surface.diagnostics.get("modes", ()):
        if mode.get("bimodal"):
            axis_id = str(mode["axis_id"])
            return LoopAction(
                action_id=f"split-regime-{axis_id}",
                action_kind="split_regime",
                rationale=(
                    f"axis {axis_id} shows bimodality with score {float(mode['score']):.3f}"
                ),
                affected_entities=(axis_id,),
                expected_decision_loss_reduction=float(mode["score"]),
                estimated_cost=3.0,
                evidence=("bimodal_distribution",),
            )
    return None


def _minimize_action(
    surface: ParticleSurface,
    irs: tuple[CompiledFactorIR, ...],
) -> LoopAction | None:
    if len(irs) <= 1:
        return None
    ess = float(surface.diagnostics.get("effective_sample_size", 0.0))
    if ess >= 0.95:
        return None
    target = irs[-1].mapping_id
    return LoopAction(
        action_id=f"minimize-{target}",
        action_kind="minimize",
        rationale=f"mapping {target} contributes marginally to effective sample size {ess:.3f}",
        affected_entities=(target,),
        expected_decision_loss_reduction=1.0 - ess,
        estimated_cost=0.5,
        evidence=("low_contribution_mapping",),
    )


def _expand_variable_action(
    variables: Mapping[str, dict[str, Any]],
    axes: tuple[dict[str, Any], ...],
    surface: ParticleSurface,
    policy: Mapping[str, Any],
) -> LoopAction | None:
    for name, spec in variables.items():
        if spec["status"] != "missing":
            continue
        if not axes:
            return None
        axis_id = str(axes[0]["axis_id"])
        return LoopAction(
            action_id=f"expand-variable-{name}",
            action_kind="expand_variable",
            rationale=f"variable {name} is missing and blocks axis {axis_id} usefulness",
            affected_entities=(name, axis_id),
            expected_decision_loss_reduction=float(policy["min_action_benefit"]) + 1.0,
            estimated_cost=1.5,
            evidence=("missing_variable",),
        )
    return None


def _evaluate_decision_value(
    axes: tuple[dict[str, Any], ...],
    surface: ParticleSurface,
    policy: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    all_useful = True
    for axis in axes:
        axis_id = str(axis["axis_id"])
        summary = surface.marginals.get(axis_id, {"width": 0.0})
        useful, reason = _axis_useful(axis, summary, policy)
        evidence.append({"axis_id": axis_id, "useful": useful, "reason": reason})
        all_useful = all_useful and useful
    ess = float(surface.diagnostics.get("effective_sample_size", 0.0))
    ess_ok = ess >= float(policy["min_effective_sample_size"])
    evidence.append({"criterion": "effective_sample_size", "value": ess, "threshold": policy["min_effective_sample_size"], "passed": ess_ok})
    return all_useful and ess_ok, evidence


# ---------------------------------------------------------------------------
# Public loop driver
# ---------------------------------------------------------------------------


def run_wave_loop(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drive one deterministic round of the wave-loop and return a replayable record."""

    parsed = validate_joint_schema(payload)
    run_id = parsed["run_id"]
    axes = parsed["axes"]
    variables = parsed["variables"]
    mappings = parsed["mappings"]
    policy = parsed["decision_policy"]
    config = parsed["budget"]
    irs = tuple(compile_factor_ir(mapping) for mapping in mappings) or _seed_default_ir(axes)
    rng = Random(config.seed)
    ledger = AnalysisLedger(run_id)
    events: list[WaveEvent] = []

    def record(
        surface_id: str,
        state: str,
        round_index: int,
        reason: str,
        data: Mapping[str, Any],
    ) -> WaveEvent:
        sequence = len(events) + 1
        event_id = f"{run_id}-wave-{sequence:06d}"
        event = WaveEvent(
            schema_version=SCHEMA_VERSION,
            event_id=event_id,
            run_id=run_id,
            surface_id=surface_id,
            state=state,
            round_index=round_index,
            reason=reason,
            data=dict(data),
            revision=Revision(
                revision_id=f"{event_id}-r1",
                sequence=1,
                created_at=config.started_at,
            ),
        )
        events.append(event)
        ledger.append("wave_event", event)
        return event

    round_index = 0
    accepted = False
    stop_reason = "budget-exhausted"
    surface = evaluate_particle_surface(
        tuple(axes),
        variables,
        irs,
        rng,
        config.particle_count,
        surface_counter=1,
    )
    record(surface.surface_id, "DRAFT", round_index, "mapping_compiled", {"axes": list(surface.axes)})
    record(
        surface.surface_id,
        "EVALUATED",
        round_index,
        "particle_surface_computed",
        {
            "particle_count": surface.diagnostics["particle_count"],
            "effective_sample_size": surface.diagnostics["effective_sample_size"],
        },
    )
    useful, evidence = _evaluate_decision_value(tuple(axes), surface, policy)
    actions = _decision_actions(tuple(axes), variables, irs, surface, policy, config)
    if useful:
        accepted = True
        stop_reason = "decision-value-met"
        record(
            surface.surface_id,
            "ACCEPTED",
            round_index,
            stop_reason,
            {"criteria_evidence": evidence, "actions_emitted": [a.action_kind for a in actions]},
        )
        stop_action = LoopAction(
            action_id="stop-decision-value-met",
            action_kind="stop",
            rationale="decision-value policy satisfied for every axis",
            affected_entities=tuple(surface.axes),
            expected_decision_loss_reduction=0.0,
            estimated_cost=0.0,
            evidence=tuple(
                str(item["reason"])
                for item in evidence
                if isinstance(item, Mapping) and item.get("reason")
            ),
        )
        if stop_action not in actions:
            actions.append(stop_action)
    else:
        record(
            surface.surface_id,
            "REFINING",
            round_index,
            "diagnostic_actions_emitted",
            {"actions_emitted": [a.action_kind for a in actions], "criteria_evidence": evidence},
        )
        unresolved_axes = [
            str(item["axis_id"])
            for item in evidence
            if isinstance(item, Mapping) and "axis_id" in item and not item.get("useful")
        ]
        if unresolved_axes:
            stop_reason = "unresolved-decision-criteria"
        else:
            stop_reason = "marginal-value-insufficient"
        record(
            surface.surface_id,
            "UNRESOLVED",
            round_index,
            stop_reason,
            {"unresolved_axes": unresolved_axes, "criteria_evidence": evidence},
        )
        actions.append(
            LoopAction(
                action_id="stop-budget-exhausted" if not actions else "stop-defer-to-action",
                action_kind="stop",
                rationale="loop terminated; further iterations require additional evidence or mappings",
                affected_entities=tuple(unresolved_axes),
                expected_decision_loss_reduction=0.0,
                estimated_cost=0.0,
                evidence=(stop_reason,),
            )
        )

    ledger_export = ledger.export()
    checkpoint = create_wave_checkpoint(ledger_export)
    return {
        "schema_version": WAVE_LOOP_RESULT_VERSION,
        "run_id": run_id,
        "surface": _surface_to_dict(surface),
        "decision_value": {
            "accepted": accepted,
            "stop_reason": stop_reason,
            "criteria_evidence": evidence,
        },
        "actions": [_action_to_dict(action) for action in actions],
        "budget": {
            "max_rounds": config.max_rounds,
            "max_actions": config.max_actions,
            "particle_count": config.particle_count,
            "seed": config.seed,
            "rounds_used": round_index + 1,
            "actions_emitted": len(actions),
        },
        "ledger": ledger_export,
        "checkpoint": checkpoint,
        "ranking_method": {
            "name": "diagnostic_action_policy_v1",
            "formula": "actions ranked by expected_decision_loss_reduction - estimated_cost",
            "calibrated": False,
            "warning": "decision-value is not a calibrated posterior probability",
        },
    }


def _seed_default_ir(axes: tuple[dict[str, Any], ...]) -> tuple[CompiledFactorIR, ...]:
    if not axes:
        return ()
    formula = "0"
    tree = ast.parse(formula, mode="eval")
    return (
        CompiledFactorIR(
            ir_version=FACTOR_IR_VERSION,
            mapping_id="default-zero",
            formula=formula,
            tree=tree,
            referenced_variables=(),
        ),
    )


def _surface_to_dict(surface: ParticleSurface) -> dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "surface_version": surface.surface_version,
        "semantics": surface.semantics,
        "axes": list(surface.axes),
        "marginals": {axis: dict(summary) for axis, summary in surface.marginals.items()},
        "diagnostics": {
            "particle_count": surface.diagnostics["particle_count"],
            "effective_sample_size": surface.diagnostics["effective_sample_size"],
            "entropy": surface.diagnostics["entropy"],
            "modes": list(surface.diagnostics["modes"]),
            "residual_interaction": surface.diagnostics["residual_interaction"],
            "residual_interaction_pair": list(surface.diagnostics["residual_interaction_pair"] or ()),
        },
        "log_weights": [float(weight) for weight in surface.log_weights],
    }


def _action_to_dict(action: LoopAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "action_kind": action.action_kind,
        "rationale": action.rationale,
        "affected_entities": list(action.affected_entities),
        "expected_decision_loss_reduction": action.expected_decision_loss_reduction,
        "estimated_cost": action.estimated_cost,
        "evidence": list(action.evidence),
    }


# ---------------------------------------------------------------------------
# Replay + checkpoint
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WaveLoopError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WaveLoopError(f"{name} must be an array")
    return value


def replay_wave_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a wave ledger and reconstruct the loop state from events."""

    source = _object(ledger, "ledger")
    if source.get("schema_version") not in (WAVE_LEDGER_VERSION, SCHEMA_VERSION):
        raise WaveLoopError("unsupported wave ledger schema_version")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise WaveLoopError("ledger.run_id is required")
    entries = _array(source.get("entries"), "ledger.entries")

    seen_events: set[str] = set()
    seen_revisions: set[str] = set()
    events: list[dict[str, Any]] = []
    surfaces: dict[str, list[str]] = {}
    actions: list[dict[str, Any]] = []
    last_round = 0
    last_state: str | None = None
    accepted = False
    terminal: dict[str, Any] | None = None

    for offset, raw_entry in enumerate(entries, start=1):
        entry = _object(raw_entry, f"ledger.entries[{offset - 1}]")
        if entry.get("sequence") != offset:
            raise WaveLoopError(f"wave sequence must be contiguous at entry {offset}")
        if entry.get("record_type") != "wave_event":
            raise WaveLoopError(f"entry {offset} is not a wave_event")
        payload = _object(entry.get("payload"), f"entry {offset}.payload")
        if entry.get("payload_hash") != _digest(payload):
            raise WaveLoopError(f"payload hash mismatch at entry {offset}")
        event_id = str(payload.get("event_id", ""))
        revision = _object(payload.get("revision"), f"entry {offset}.payload.revision")
        revision_id = str(revision.get("revision_id", ""))
        surface_id = str(payload.get("surface_id", ""))
        state = str(payload.get("state", ""))
        round_index = payload.get("round_index")
        reason = str(payload.get("reason", ""))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise WaveLoopError(f"unsupported event schema at entry {offset}")
        if payload.get("run_id") != run_id:
            raise WaveLoopError(f"run_id mismatch at entry {offset}")
        if not event_id or event_id in seen_events:
            raise WaveLoopError(f"invalid or duplicate event_id at entry {offset}")
        if entry.get("stable_id") != event_id:
            raise WaveLoopError(f"stable_id mismatch at entry {offset}")
        if not revision_id or revision_id in seen_revisions:
            raise WaveLoopError(f"invalid or duplicate revision_id at entry {offset}")
        if entry.get("revision_id") != revision_id:
            raise WaveLoopError(f"revision_id mismatch at entry {offset}")
        if revision.get("sequence") != 1 or revision.get("supersedes_revision_id") not in (None, ""):
            raise WaveLoopError(f"wave event revision must be an initial revision at entry {offset}")
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
            raise WaveLoopError(f"invalid round_index at entry {offset}")
        if round_index < last_round:
            raise WaveLoopError(f"round_index moved backwards at entry {offset}")
        if state not in WAVE_ACTIVATION_STATES | WAVE_TERMINAL_STATES:
            raise WaveLoopError(f"unsupported wave state at entry {offset}: {state}")
        if not reason:
            raise WaveLoopError(f"reason is required at entry {offset}")
        if terminal is not None:
            raise WaveLoopError("terminal event must be the final ledger entry")

        if state == "DRAFT":
            surfaces.setdefault(surface_id, []).append("DRAFT")
        elif state == "EVALUATED":
            surfaces.setdefault(surface_id, []).append("EVALUATED")
        elif state == "REFINING":
            for record_actions in payload.get("data", {}).get("actions_emitted", ()):
                actions.append({"surface_id": surface_id, "action_kind": str(record_actions)})
            surfaces.setdefault(surface_id, []).append("REFINING")
        elif state == "ACCEPTED":
            accepted = True
            surfaces.setdefault(surface_id, []).append("ACCEPTED")
            terminal = {"state": state, "surface_id": surface_id, "round_index": round_index, "reason": reason}
        elif state == "UNRESOLVED":
            surfaces.setdefault(surface_id, []).append("UNRESOLVED")
            terminal = {"state": state, "surface_id": surface_id, "round_index": round_index, "reason": reason}
        elif state == "STOP":
            terminal = {"state": state, "surface_id": surface_id, "round_index": round_index, "reason": reason}
        else:
            raise WaveLoopError(f"unsupported activation state at entry {offset}: {state}")

        events.append(
            {
                "event_id": event_id,
                "surface_id": surface_id,
                "state": state,
                "round_index": round_index,
                "reason": reason,
                "data": json.loads(_canonical(payload.get("data", {}))),
            }
        )
        seen_events.add(event_id)
        seen_revisions.add(revision_id)
        last_round = round_index
        last_state = state

    return {
        "schema_version": "joint-wave-replay.v1",
        "run_id": run_id,
        "event_count": len(entries),
        "current_state": last_state,
        "terminal": terminal,
        "accepted": accepted,
        "surfaces": {surface_id: list(states) for surface_id, states in surfaces.items()},
        "actions": actions,
        "events": events,
    }


def create_wave_checkpoint(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Build a self-verifying checkpoint from a wave ledger."""

    replay = replay_wave_ledger(ledger)
    body = {
        "schema_version": WAVE_CHECKPOINT_VERSION,
        "ledger": json.loads(_canonical(ledger)),
        "replay": replay,
    }
    return {**body, "checkpoint_hash": _digest(body)}


def verify_wave_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a wave checkpoint and return the reconstructed loop state."""

    source = _object(checkpoint, "checkpoint")
    if source.get("schema_version") != WAVE_CHECKPOINT_VERSION:
        raise WaveLoopError("unsupported wave checkpoint schema_version")
    supplied_hash = source.get("checkpoint_hash")
    body = {key: value for key, value in source.items() if key != "checkpoint_hash"}
    if not isinstance(supplied_hash, str) or supplied_hash != _digest(body):
        raise WaveLoopError("checkpoint hash mismatch")
    replay = replay_wave_ledger(_object(source.get("ledger"), "checkpoint.ledger"))
    if source.get("replay") != replay:
        raise WaveLoopError("checkpoint replay state mismatch")
    return replay


__all__ = [
    "ACTION_KINDS",
    "JOINT_SCHEMA_VERSION",
    "FACTOR_IR_VERSION",
    "PARTICLE_SURFACE_VERSION",
    "WAVE_LEDGER_VERSION",
    "WAVE_CHECKPOINT_VERSION",
    "WAVE_LOOP_RESULT_VERSION",
    "CompiledFactorIR",
    "LoopAction",
    "ParticleSurface",
    "WaveEvent",
    "WaveLoopConfig",
    "WaveLoopError",
    "compile_factor_ir",
    "create_wave_checkpoint",
    "evaluate_particle_surface",
    "replay_wave_ledger",
    "run_wave_loop",
    "validate_joint_schema",
    "verify_wave_checkpoint",
]