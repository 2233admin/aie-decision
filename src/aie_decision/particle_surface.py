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


# ---------------------------------------------------------------------------
# Compiled IR discriminator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledIR:
    """Minimal typed container: the compiled AST plus its IR kind discriminator.

    ``is_factor=True``  → FactorIR — dimensionless log-potential weight only.
    ``is_factor=False`` → DeterministicTransform — axis-value computation only.
    """

    mapping_id: str
    tree: ast.Expression
    is_factor: bool


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
    """Declarative description of the wave surface to compile.

    *compiled_ir_trees* is required: every FORMULA mapping must have exactly
    one compiled entry with a matching *mapping_id*.  The compiled IR is the
    single evaluation authority — there is no raw-formula parser fallback.
    Each entry discriminates FactorIR (*is_factor=True*) from
    DeterministicTransform (*is_factor=False*), which determines whether the
    mapping contributes to log-weights or axis values.
    """

    question_id: str
    seed: int
    particle_count: int
    axes: tuple[OutcomeAxis, ...]
    variables: tuple[VariableSpec, ...]
    mappings: tuple[MappingSpec, ...]
    compiled_ir_trees: Mapping[str, CompiledIR] = field(default_factory=dict)
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
        formula_ids = set()
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
            if mapping.kind is MappingKind.FORMULA:
                formula_ids.add(mapping.mapping_id)
        # Validate compiled_ir_trees: every FORMULA mapping must have exactly
        # one compiled entry with matching mapping_id and a valid tree.
        compiled_ids = set(self.compiled_ir_trees.keys())
        missing_compiled = formula_ids - compiled_ids
        if missing_compiled:
            raise ValueError(
                "SurfaceRequest.compiled_ir_trees missing entries for FORMULA mappings: "
                + ", ".join(sorted(missing_compiled))
            )
        extra_compiled = compiled_ids - {m.mapping_id for m in self.mappings}
        if extra_compiled:
            raise ValueError(
                "SurfaceRequest.compiled_ir_trees has entries for unknown mapping_ids: "
                + ", ".join(sorted(extra_compiled))
            )
        for mid in formula_ids:
            compiled = self.compiled_ir_trees[mid]
            if compiled.mapping_id != mid:
                raise ValueError(
                    f"compiled_ir_trees[{mid!r}].mapping_id mismatch: "
                    f"{compiled.mapping_id!r}"
                )
            if not isinstance(compiled.tree, ast.Expression):
                raise ValueError(
                    f"compiled_ir_trees[{mid!r}].tree must be ast.Expression"
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


def _evaluate_formula_vector(
    tree: ast.Expression,
    samples: np.ndarray,
    name_to_index: Mapping[str, int],
) -> np.ndarray:
    """Vectorised evaluation across every particle in the cloud.

    The *tree* must be a pre-validated compiled IR AST — this function does
    NOT parse any raw expression and has no fallback parser path.
    """

    _name_to_index = dict(name_to_index)

    def _walk(node: ast.AST) -> np.ndarray:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Name):
            return samples[:, _name_to_index[node.id]]
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

    return _walk(tree)


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
                "observation": mapping.observation,
                "observation_scale": mapping.observation_scale,
                # Surface identity derives from the compiled IR (single
                # authority), not the raw expression string.  A tampered
                # expression with unchanged compiled IR produces the same
                # identity because the compiled AST is what executes.
                "compiled_ir_is_factor": (
                    request.compiled_ir_trees[mapping.mapping_id].is_factor
                    if mapping.kind is MappingKind.FORMULA and mapping.mapping_id in request.compiled_ir_trees
                    else None
                ),
                "compiled_ir_ast_hash": (
                    hashlib.sha256(
                        ast.dump(request.compiled_ir_trees[mapping.mapping_id].tree, annotate_fields=False)
                        .encode("utf-8")
                    ).hexdigest()[:16]
                    if mapping.kind is MappingKind.FORMULA and mapping.mapping_id in request.compiled_ir_trees
                    else None
                ),
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
    * evaluates every declared mapping on each particle using its pre-compiled
      IR tree (no raw formula parsing — the compiled IR is the single authority),
    * splits behaviour by compiled IR kind:
      - **DeterministicTransform** (*is_factor=False*): writes the evaluated
        result to the target axis column; zero weight contribution.
      - **FactorIR** (*is_factor=True*): adds the evaluated result to the
        log-potential weight; does **not** overwrite axis values.
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
            # Mandatory: every FORMULA mapping must have a compiled IR entry.
            # Missing, wrong-id, or wrong-type entries are rejected here before
            # any particle is evaluated — no raw formula fallback exists.
            compiled = request.compiled_ir_trees.get(mapping.mapping_id)
            if compiled is None:
                raise ValueError(
                    f"FORMULA mapping {mapping.mapping_id!r} has no compiled IR entry"
                )
            contribution = _evaluate_formula_vector(
                compiled.tree, samples, name_to_index
            )
            if compiled.is_factor:
                # FactorIR — dimensionless log-potential weight only.
                # Add the contribution to log_weights; do NOT overwrite axis
                # values (the axis column retains its prior value, which may
                # be zero or computed by a DeterministicTransform).
                mapping_contribution = contribution
                log_potential = log_potential + mapping_contribution
            else:
                # DeterministicTransform — axis-value computation only.
                # Write the axis particles; zero weight contribution.
                particles[:, axis_index] = contribution
                mapping_contribution = np.zeros(request.particle_count, dtype=float)
        else:  # LIKELIHOOD
            variable_index = name_to_index[mapping.variables[0]]
            obs_lower, obs_upper = mapping.observation  # type: ignore[misc]
            centre = 0.5 * (obs_lower + obs_upper)
            half_width = 0.5 * (obs_upper - obs_lower)
            scale = float(mapping.observation_scale)  # type: ignore[arg-type]
            deviation = np.abs(samples[:, variable_index] - centre)
            outside = np.maximum(deviation - half_width, 0.0)
            flat_penalty = (outside > 0).astype(float)
            tail_penalty = flat_penalty + 0.001 * (outside / max(scale, 1e-12)) ** 2
            mapping_contribution = -tail_penalty
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
    "CompiledIR",
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