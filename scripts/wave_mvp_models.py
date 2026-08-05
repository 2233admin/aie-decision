"""MVP domain models and result types."""

from __future__ import annotations
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from wave_mvp_unit import Dimension

SCHEMA_VERSION = "joint-wave-surface-mvp.v1"

@dataclass(frozen=True, slots=True)
class OutcomeAxis:
    name: str
    unit: str
    domain: tuple[float, float]
    time_semantics: str = "static"
    tolerance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutcomeSpace:
    axes: tuple[OutcomeAxis, ...]

    def axis(self, name: str) -> OutcomeAxis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class VariableSpec:
    name: str
    unit: str
    lower: float
    upper: float
    method: str = "user_supplied"
    ablatable: bool = False
    bimodal: bool = False


@dataclass(frozen=True, slots=True)
class MappingSpec:
    mapping_id: str
    formula: str
    output_axes: tuple[str, ...]
    expected_unit: str
    expect_failure: str | None = None

    def is_legal(self) -> bool:
        return self.expect_failure is None


@dataclass(frozen=True, slots=True)
class MappingFailure:
    """Structured failure returned before any particle evaluation."""

    mapping_id: str
    code: str
    message: str
    operand: str
    operand_unit: str
    expected_unit: str


@dataclass(frozen=True, slots=True)
class CompiledMapping:
    mapping: MappingSpec
    compiled: Any  # ast.Expression
    variables: tuple[str, ...]
    expected_dimension: Dimension
    failure: MappingFailure | None = None

    @property
    def is_legal(self) -> bool:
        return self.failure is None


@dataclass(frozen=True, slots=True)
class ParticleSurface:
    axis_order: tuple[str, ...]
    values: np.ndarray  # shape (n_particles, n_axes)
    log_weight: np.ndarray  # shape (n_particles,)
    surface_kind: str
    calibration: str
    coverage_semantics: str
    evaluator_version: str
    seed: int

    def weight(self) -> np.ndarray:
        shifted = self.log_weight - np.max(self.log_weight)
        weights = np.exp(shifted)
        total = float(np.sum(weights))
        if total <= 0 or not np.isfinite(total):
            return np.zeros_like(weights)
        return weights / total

    def marginal(self, axis_index: int, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
        column = self.values[:, axis_index]
        weights = self.weight()
        histogram, edges = np.histogram(column, bins=bins, weights=weights)
        density = histogram / max(float(np.sum(histogram)), 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, density


@dataclass(frozen=True, slots=True)
class AxisDiagnostic:
    name: str
    unit: str
    absolute_width: float
    relative_width: float
    sharpness_absolute: float
    sharpness_relative: float
    effective_sample_size: float
    entropy_nats: float
    mode_count: int
    mode_locations: tuple[float, ...]
    residual_proxy: float


@dataclass(frozen=True, slots=True)
class SurfaceDiagnostics:
    particle_count: int
    axes: tuple[AxisDiagnostic, ...]
    bimodal_axes: tuple[str, ...]
    constraint_failures: tuple[str, ...]
    effective_sample_size: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "particle_count": self.particle_count,
            "axes": [dataclasses.asdict(axis) for axis in self.axes],
            "bimodal_axes": list(self.bimodal_axes),
            "constraint_failures": list(self.constraint_failures),
            "effective_sample_size": self.effective_sample_size,
        }


@dataclass(frozen=True, slots=True)
class TypedAction:
    """A diagnostic-driven loop action."""

    action_id: str
    kind: str  # measure | add_interaction | split_regime | minimize | stop
    target: str
    rationale: str
    expected_loss_improvement: float
    estimated_cost: float
    affected_entities: tuple[str, ...]
    basis: Mapping[str, Any] = field(default_factory=dict)



@dataclass(frozen=True, slots=True)
class LoopIteration:
    round_index: int
    surface: dict[str, Any]
    diagnostics: dict[str, Any]
    decision: dict[str, Any]
    action: dict[str, Any] | None
    event_id: str


@dataclass(frozen=True, slots=True)
class LoopResult:
    run_id: str
    status: str  # result-found | budget-exhausted | insufficient-information
    iterations: tuple[LoopIteration, ...]
    accepted_surface: dict[str, Any] | None
    illegal_mappings: tuple[dict[str, Any], ...]
    ledger: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "iterations": [dataclasses.asdict(iteration) for iteration in self.iterations],
            "accepted_surface": self.accepted_surface,
            "illegal_mappings": list(self.illegal_mappings),
            "ledger": self.ledger,
            "actions": list(self.actions),
            "summary": self.summary,
        }

    def to_json_safe(self) -> dict[str, Any]:
        """Return a JSON-encodable copy with non-finite floats stringified."""
        return _sanitize_for_json(self.to_mapping())


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON-safe strings."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    return value


def _new_event_id(run_id: str, sequence: int) -> str:
    return f"{run_id}-wave-{sequence:06d}"


def _payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

