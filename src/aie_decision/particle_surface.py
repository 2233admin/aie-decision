"""Deterministic CPU particle joint wave surface.

This module represents a joint outcome surface as a weighted particle cloud rather
than a dense high-dimensional tensor.  Each particle carries one value per outcome
axis and a non-negative unnormalised weight derived from the declared mappings
and evidence.  Diagnostics in :mod:`aie_decision.wave_diagnostics` consume the
particle representation directly.

The module enforces two semantic rules from the joint-wave PRD:

* It refuses to call the surface a ``probability_surface`` unless a calibration
  basis has been declared and validated.  Otherwise it is a
  ``possibility_surface``.
* It never materialises ``candidates × variables × samples`` dense tensors; all
  sampling is per-particle.

The implementation depends only on the standard library and :mod:`numpy`.  The
sampler uses the deterministic ``numpy.random.SeedSequence`` /
``numpy.random.default_rng`` chain so equal seeds yield equal particles.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


class SurfaceKind(StrEnum):
    POSSIBILITY = "possibility_surface"
    PROBABILITY = "probability_surface"


class CoverageSemantics(StrEnum):
    UNCALIBRATED_RANGE = "uncalibrated_range"
    DECLARED_CREDIBLE_INTERVAL = "declared_credible_interval"
    EMPIRICAL_PREDICTION_INTERVAL = "empirical_prediction_interval"
    SCENARIO_BOUND = "scenario_bound"


class MappingKind(StrEnum):
    """Mapping families accepted by the particle surface.

    ``FORMULA`` is a deterministic mapping compiled from a Fermi-style arithmetic
    expression.  ``LIKELIHOOD`` is an evidence-style soft constraint that
    down-weights particles which disagree with the declared observation
    interval.
    """

    FORMULA = "formula"
    LIKELIHOOD = "likelihood"


class CalibrationBasis(StrEnum):
    UNMEASURED = "unmeasured"
    USER_DECLARED = "user_declared"
    HISTORICAL_RESIDUAL = "historical_residual"


@dataclass(frozen=True, slots=True)
class OutcomeAxis:
    """A single axis of the joint outcome space.

    The axis records its unit and an optional admissible domain.  Mixing
    incommensurable units into a single axis is rejected by the surface
    compiler.
    """

    name: str
    unit: str
    domain: tuple[float, float] | None = None
    time_semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("OutcomeAxis.name is required")
        if not self.unit.strip():
            raise ValueError("OutcomeAxis.unit is required")
        if self.domain is not None:
            lower, upper = self.domain
            if not isfinite(lower) or not isfinite(upper) or lower > upper:
                raise ValueError("OutcomeAxis.domain must be finite and ordered")


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """A declared input variable with bounded support.

    Variables may carry ``evidence_method`` metadata (``observed``,
    ``measured``, ``assumed``).  The metadata feeds the likelihood mapping and
    the diagnostics summary but never replaces a bounded support with a point.
    """

    name: str
    unit: str
    lower: float
    upper: float
    evidence_method: str = "declared"
    domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("VariableSpec.name is required")
        if not self.unit.strip():
            raise ValueError("VariableSpec.unit is required")
        if not isfinite(self.lower) or not isfinite(self.upper) or self.lower > self.upper:
            raise ValueError("VariableSpec bounds must be finite and ordered")
        if self.domain is not None:
            d_lower, d_upper = self.domain
            if not isfinite(d_lower) or not isfinite(d_upper) or d_lower > d_upper:
                raise ValueError("VariableSpec.domain must be finite and ordered")
            if self.lower < d_lower or self.upper > d_upper:
                raise ValueError("VariableSpec bounds must lie inside its domain")
        if not self.evidence_method.strip():
            raise ValueError("VariableSpec.evidence_method is required")


@dataclass(frozen=True, slots=True)
class MappingSpec:
    """A deterministic or likelihood-style connection between variables and axes.

    For ``FORMULA`` mappings the ``expression`` is parsed with the same
    arithmetic grammar used by :mod:`aie_decision.fermi`; the expression must
    be dimensionless over the chosen ``result_axis``.  ``LIKELIHOOD`` mappings
    apply a Gaussian penalty between a single variable and a single axis using
    the declared ``observation`` and ``observation_scale``.
    """

    mapping_id: str
    kind: MappingKind
    variables: tuple[str, ...]
    result_axis: str
    expression: str | None = None
    observation: tuple[float, float] | None = None
    observation_scale: float | None = None
    unit_signature: str = "dimensionless"

    def __post_init__(self) -> None:
        if not self.mapping_id.strip():
            raise ValueError("MappingSpec.mapping_id is required")
        if not self.result_axis.strip():
            raise ValueError("MappingSpec.result_axis is required")
        if not self.variables:
            raise ValueError("MappingSpec.variables must be non-empty")
        if self.kind is MappingKind.FORMULA:
            if not self.expression or not self.expression.strip():
                raise ValueError("FORMULA mapping requires expression")
            if self.observation is not None or self.observation_scale is not None:
                raise ValueError("FORMULA mapping cannot declare observation")
        elif self.kind is MappingKind.LIKELIHOOD:
            if len(self.variables) != 1:
                raise ValueError("LIKELIHOOD mapping requires exactly one variable")
            if self.expression:
                raise ValueError("LIKELIHOOD mapping cannot declare expression")
            if self.observation is None or self.observation_scale is None:
                raise ValueError(
                    "LIKELIHOOD mapping requires observation and observation_scale"
                )
            obs_lower, obs_upper = self.observation
            if (
                not isfinite(obs_lower)
                or not isfinite(obs_upper)
                or obs_lower > obs_upper
            ):
                raise ValueError("LIKELIHOOD observation must be finite and ordered")
            if not isfinite(self.observation_scale) or self.observation_scale <= 0:
                raise ValueError("LIKELIHOOD observation_scale must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """Proof that a surface can be relabelled as a probability surface.

    The record is intentionally narrow: it only asserts that a calibration basis
    has been declared and that the required metadata is present.  It does not
    itself establish empirical calibration.
    """

    basis: CalibrationBasis
    declared_at: str
    sample_size: int | None = None
    reference_set: str | None = None

    def __post_init__(self) -> None:
        if not self.declared_at.strip():
            raise ValueError("CalibrationRecord.declared_at is required")
        if self.sample_size is not None and self.sample_size < 0:
            raise ValueError("CalibrationRecord.sample_size must be non-negative")


@dataclass(frozen=True, slots=True)
class SurfaceRequest:
    """Declarative description of the wave surface to compile."""

    question_id: str
    seed: int
    particle_count: int
    axes: tuple[OutcomeAxis, ...]
    variables: tuple[VariableSpec, ...]
    mappings: tuple[MappingSpec, ...]
    coverage_semantics: CoverageSemantics = CoverageSemantics.UNCALIBRATED_RANGE
    calibration: CalibrationRecord | None = None
    schema_version: str = "particle_surface/1"

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("SurfaceRequest.question_id is required")
        if not isinstance(self.seed, int):
            raise ValueError("SurfaceRequest.seed must be an integer")
        if self.particle_count <= 0:
            raise ValueError("SurfaceRequest.particle_count must be positive")
        if not self.axes:
            raise ValueError("SurfaceRequest.axes must be non-empty")
        if not self.variables:
            raise ValueError("SurfaceRequest.variables must be non-empty")
        axis_names = [axis.name for axis in self.axes]
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("OutcomeAxis.name values must be unique")
        variable_names = [variable.name for variable in self.variables]
        if len(set(variable_names)) != len(variable_names):
            raise ValueError("VariableSpec.name values must be unique")
        mapping_ids = [mapping.mapping_id for mapping in self.mappings]
        if len(set(mapping_ids)) != len(mapping_ids):
            raise ValueError("MappingSpec.mapping_id values must be unique")
        axis_set = set(axis_names)
        variable_set = set(variable_names)
        for mapping in self.mappings:
            if mapping.result_axis not in axis_set:
                raise ValueError(
                    f"MappingSpec.result_axis unknown: {mapping.result_axis}"
                )
            missing = [name for name in mapping.variables if name not in variable_set]
            if missing:
                raise ValueError(
                    f"MappingSpec {mapping.mapping_id} references unknown variables: "
                    + ", ".join(missing)
                )


@dataclass(frozen=True, slots=True)
class ParticleSurface:
    """Compiled weighted particle representation of a joint wave surface.

    The surface stores raw outcome coordinates and unnormalised log-weights
    alongside the metadata required for a fair replay.  Weight normalisation
    and diagnostics live in :mod:`aie_decision.wave_diagnostics`.
    """

    surface_id: str
    kind: SurfaceKind
    coverage_semantics: CoverageSemantics
    calibration_basis: CalibrationBasis
    schema_version: str
    seed: int
    question_id: str
    particle_count: int
    axis_names: tuple[str, ...]
    axis_units: tuple[str, ...]
    variable_names: tuple[str, ...]
    mapping_ids: tuple[str, ...]
    particles: np.ndarray  # shape (n_particles, n_axes)
    log_weights: np.ndarray  # shape (n_particles,)
    mapping_breakdown: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.particles.ndim != 2:
            raise ValueError("particles must be a 2-D array")
        if self.particles.shape[0] != self.particle_count:
            raise ValueError("particle_count must match particles.shape[0]")
        if self.particles.shape[1] != len(self.axis_names):
            raise ValueError("particles must have one column per axis")
        if self.log_weights.shape != (self.particle_count,):
            raise ValueError("log_weights must be a 1-D array of length particle_count")
        if not bool(np.all(np.isfinite(self.particles))):
            raise ValueError("particles must be finite")
        if not bool(np.all(np.isfinite(self.log_weights))):
            raise ValueError("log_weights must be finite")
        for mapping_id, contribution in self.mapping_breakdown.items():
            if not isinstance(mapping_id, str):
                raise ValueError("mapping_breakdown keys must be strings")
            if contribution.shape != (self.particle_count,):
                raise ValueError(
                    f"mapping_breakdown[{mapping_id}] must have length particle_count"
                )
            if not bool(np.all(np.isfinite(contribution))):
                raise ValueError(
                    f"mapping_breakdown[{mapping_id}] must be finite"
                )


_ALLOWED_FORMULA_NODES: tuple[type[ast.AST], ...] = (
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


@dataclass(frozen=True, slots=True)
class _ParsedFormula:
    tree: ast.Expression
    variable_names: tuple[str, ...]


def _parse_formula(expression: str, allowed_variables: set[str]) -> _ParsedFormula:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("mapping expression is required")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError("mapping expression must be valid arithmetic") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_FORMULA_NODES):
            raise ValueError(
                "mapping expression supports only names, numbers, +, -, *, / and parentheses"
            )
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("mapping expression constants must be numeric")
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in allowed_variables:
                raise ValueError(f"mapping references unknown variable: {node.id}")
            if node.id not in names:
                names.append(node.id)
    if not names:
        raise ValueError("mapping expression must reference at least one variable")
    return _ParsedFormula(tree, tuple(names))


def _evaluate_formula_vector(
    parsed: _ParsedFormula,
    samples: np.ndarray,
    name_to_index: Mapping[str, int],
) -> np.ndarray:
    """Vectorised evaluation across every particle in the cloud."""

    name_to_index = dict(name_to_index)

    def _walk(node: ast.AST) -> np.ndarray:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Name):
            return samples[:, name_to_index[node.id]]
        if isinstance(node, ast.Constant):
            return np.full(samples.shape[0], float(node.value))
        if isinstance(node, ast.UnaryOp):
            operand = _walk(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if np.any(right == 0):
                    raise ValueError("mapping division by zero")
                return left / right
        raise ValueError("unsupported mapping node")

    return _walk(parsed.tree)


def _stable_surface_id(request: SurfaceRequest) -> str:
    payload = {
        "question_id": request.question_id,
        "axes": [axis.name for axis in request.axes],
        "variables": [
            {
                "name": variable.name,
                "lower": variable.lower,
                "upper": variable.upper,
                "unit": variable.unit,
            }
            for variable in request.variables
        ],
        "mappings": [
            {
                "mapping_id": mapping.mapping_id,
                "kind": mapping.kind.value,
                "variables": list(mapping.variables),
                "result_axis": mapping.result_axis,
                "expression": mapping.expression,
                "observation": mapping.observation,
                "observation_scale": mapping.observation_scale,
            }
            for mapping in request.mappings
        ],
        "seed": request.seed,
        "particle_count": request.particle_count,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "surface-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _classify_surface(request: SurfaceRequest) -> tuple[SurfaceKind, CalibrationBasis]:
    """Apply the possibility/probability semantic gate.

    A surface is only relabelled ``probability_surface`` when the caller has
    supplied a :class:`CalibrationRecord` with a non-``UNMEASURED`` basis.
    Otherwise the surface is always a ``possibility_surface`` regardless of
    how often the weights happen to sum to one after normalisation.
    """

    if request.calibration is None:
        return SurfaceKind.POSSIBILITY, CalibrationBasis.UNMEASURED
    if request.calibration.basis is CalibrationBasis.UNMEASURED:
        return SurfaceKind.POSSIBILITY, CalibrationBasis.UNMEASURED
    return SurfaceKind.PROBABILITY, request.calibration.basis


def compile_particle_surface(request: SurfaceRequest) -> ParticleSurface:
    """Compile a deterministic weighted particle surface.

    The compile step:

    * samples uniformly inside each variable's bounded support,
    * evaluates every declared mapping on each particle,
    * writes the unnormalised log potential as the sum of per-mapping log
      contributions,
    * enforces the possibility/probability semantic gate.

    The result is reproducible for the same ``seed`` and ``particle_count``
    because it relies only on :func:`numpy.random.default_rng`.  Callers can
    therefore replay a search replay or audit trail from the stored metadata.
    """

    axis_names = list(axis.name for axis in request.axes)
    axis_units = tuple(axis.unit for axis in request.axes)
    variable_names = list(variable.name for variable in request.variables)
    variable_bounds = np.array(
        [(variable.lower, variable.upper) for variable in request.variables],
        dtype=float,
    )
    name_to_index = {name: index for index, name in enumerate(variable_names)}

    seed_sequence = np.random.SeedSequence(request.seed)
    child_seeds = seed_sequence.spawn(2)
    sampler = np.random.default_rng(child_seeds[0])

    # Sample in u ∈ [0, 1] and project into each variable's bounded support.  No
    # grid is materialised: samples has shape (n_particles, n_variables).
    unit_samples = sampler.random((request.particle_count, len(variable_names)))
    samples = variable_bounds[:, 0] + unit_samples * (
        variable_bounds[:, 1] - variable_bounds[:, 0]
    )

    mapping_breakdown: dict[str, np.ndarray] = {}
    log_potential = np.zeros(request.particle_count, dtype=float)
    particles = np.zeros((request.particle_count, len(axis_names)), dtype=float)
    for mapping in request.mappings:
        axis_index = axis_names.index(mapping.result_axis)
        if mapping.kind is MappingKind.FORMULA:
            parsed = _parse_formula(mapping.expression or "", set(variable_names))
            contribution = _evaluate_formula_vector(parsed, samples, name_to_index)
            particles[:, axis_index] = contribution
            # Formula mappings do not change the weight; they define the axis
            # value.  We record a zero contribution so downstream diagnostics
            # can still see the per-mapping ledger.
            mapping_contribution = np.zeros(request.particle_count, dtype=float)
        else:  # LIKELIHOOD
            variable_index = name_to_index[mapping.variables[0]]
            obs_lower, obs_upper = mapping.observation  # type: ignore[misc]
            centre = 0.5 * (obs_lower + obs_upper)
            half_width = 0.5 * (obs_upper - obs_lower)
            scale = float(mapping.observation_scale)  # type: ignore[arg-type]
            deviation = np.abs(samples[:, variable_index] - centre)
            outside = np.maximum(deviation - half_width, 0.0)
            # Bounded flat penalty outside the interval with a light Gaussian
            # tail.  The flat step preserves bimodal shapes when competing
            # likelihoods protect disjoint intervals; the Gaussian tail only
            # distinguishes particles that fall far from any observation.
            flat_penalty = (outside > 0).astype(float)
            tail_penalty = flat_penalty + 0.001 * (outside / max(scale, 1e-12)) ** 2
            mapping_contribution = -tail_penalty
            # Likelihood mappings expose the variable value on the axis so
            # diagnostics still observe the original input distribution.
            particles[:, axis_index] = samples[:, variable_index]
        log_potential = log_potential + mapping_contribution
        mapping_breakdown[mapping.mapping_id] = mapping_contribution

    kind, basis = _classify_surface(request)
    surface_id = _stable_surface_id(request)
    return ParticleSurface(
        surface_id=surface_id,
        kind=kind,
        coverage_semantics=request.coverage_semantics,
        calibration_basis=basis,
        schema_version=request.schema_version,
        seed=request.seed,
        question_id=request.question_id,
        particle_count=request.particle_count,
        axis_names=tuple(axis_names),
        axis_units=axis_units,
        variable_names=tuple(variable_names),
        mapping_ids=tuple(mapping.mapping_id for mapping in request.mappings),
        particles=particles,
        log_weights=log_potential,
        mapping_breakdown=mapping_breakdown,
    )


def normalise_weights(surface: ParticleSurface) -> np.ndarray:
    """Return the normalised particle weights using log-sum-exp.

    Returns zero weights if all log-weights collapse so the caller can detect
    collapse separately.  The result is a fresh numpy array and may be used as a
    discrete measure on top of a possibility surface: it never upgrades the
    surface kind on its own.
    """

    log_weights = surface.log_weights
    if not np.isfinite(log_weights).any():
        return np.zeros(surface.particle_count, dtype=float)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(weights)
    return weights / total


def surface_as_mapping(surface: ParticleSurface) -> dict[str, Any]:
    """Render the surface into a JSON-compatible mapping for ledger export."""

    weights = normalise_weights(surface)
    return {
        "surface_id": surface.surface_id,
        "schema_version": surface.schema_version,
        "question_id": surface.question_id,
        "kind": surface.kind.value,
        "coverage_semantics": surface.coverage_semantics.value,
        "calibration_basis": surface.calibration_basis.value,
        "seed": surface.seed,
        "particle_count": surface.particle_count,
        "axis_names": list(surface.axis_names),
        "axis_units": list(surface.axis_units),
        "variable_names": list(surface.variable_names),
        "mapping_ids": list(surface.mapping_ids),
        "weight_summary": {
            "max": float(weights.max()) if weights.size else 0.0,
            "min": float(weights.min()) if weights.size else 0.0,
            "sum": float(weights.sum()),
        },
    }


# Lightweight runtime cache: identical request objects should yield identical
# surfaces without rerunning the sampler.
_CACHE: dict[tuple[int, int, str], ParticleSurface] = {}


def compile_particle_surface_cached(request: SurfaceRequest) -> ParticleSurface:
    """Compile and cache the particle surface for an idempotent request."""

    key = (request.seed, request.particle_count, request.question_id)
    cached = _CACHE.get(key)
    if cached is not None and cached.axis_names == tuple(axis.name for axis in request.axes):
        return cached
    compiled = compile_particle_surface(request)
    _CACHE[key] = compiled
    return compiled


__all__ = [
    "CalibrationBasis",
    "CalibrationRecord",
    "CoverageSemantics",
    "MappingKind",
    "MappingSpec",
    "OutcomeAxis",
    "ParticleSurface",
    "SurfaceKind",
    "SurfaceRequest",
    "compile_particle_surface",
    "compile_particle_surface_cached",
    "normalise_weights",
    "surface_as_mapping",
    "VariableSpec",
]