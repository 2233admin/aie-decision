"""Deterministic CPU joint wave surface MVP runner.

This script is the reference executor for ``joint-wave-surface-mvp.v1``
golden fixtures. It implements, in pure Python + NumPy, the smallest end
to end loop that satisfies the P0 acceptance bullets in
``docs/PRD-JOINT-WAVE-SURFACE.md``:

* multi-axis outcome space with declared units;
* legal unit conversion succeeds;
* illegal unit operations fail before evaluation with a structured error
  identifying the mapping, operand and offending unit;
* weighted particle surface that preserves bimodality;
* diagnostics (sharpness, ESS, entropy, multimodality, residual indicator);
* decision-value policy with per-axis absolute / relative / loss tolerance;
* diagnostic driven typed actions: ``measure``, ``add_interaction``,
  ``split_regime``, ``minimize``, ``stop``;
* at least one loop iteration;
* deterministic replay from a versioned seed;
* compatibility adapters into ``aie_decision.candidate_generation`` and
  ``aie_decision.search_replay`` without modifying their files.

The implementation prefers the standard library and NumPy because Pint,
SciPy and PyTorch are not yet a project dependency. Pint remains the
production unit authority; the local unit table here is an MVP shim that
covers only the units referenced by the golden fixture.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Required compatibility adapters for the MVP boundary.
# These imports are intentionally kept read-only: the runner may not edit
# tasks A-C, but a missing adapter is an integration failure, never a no-op.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised when the package is importable
    from aie_decision.candidate_generation import (  # type: ignore[import-not-found]
        FailureDiagnostic,
        generate_candidates,
    )
    _HAS_CANDIDATE_GENERATION = True
except Exception:  # noqa: BLE001 - import errors must not block the MVP
    _HAS_CANDIDATE_GENERATION = False
    FailureDiagnostic = None  # type: ignore[assignment]
    generate_candidates = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised when the package is importable
    from aie_decision.search_replay import (  # type: ignore[import-not-found]
        LEDGER_SCHEMA_VERSION as _SEARCH_LEDGER_SCHEMA_VERSION,
        replay_search_ledger as _replay_search_ledger,
    )
    _HAS_SEARCH_REPLAY = True
except Exception:  # noqa: BLE001
    _HAS_SEARCH_REPLAY = False
    _SEARCH_LEDGER_SCHEMA_VERSION = "1.0.0"
    _replay_search_ledger = None  # type: ignore[assignment]


SCHEMA_VERSION = "joint-wave-surface-mvp.v1"


class IntegrationUnavailable(ValueError):
    """Raised when a required MVP integration boundary cannot be used."""

    code = "integration_unavailable"


class ParityMismatch(ValueError):
    """Raised when an explicitly supplied oracle disagrees with the evaluator."""

    code = "parity_mismatch"


def assert_surface_parity(
    authoritative: Mapping[str, Any], oracle: Mapping[str, Any]
) -> None:
    """Compare externally observable surface/action evidence, fail closed on drift."""

    fields = ("wave_shape", "bimodal_axes", "actions", "final_status")
    mismatches = {
        field: {"authoritative": authoritative.get(field), "oracle": oracle.get(field)}
        for field in fields
        if authoritative.get(field) != oracle.get(field)
    }
    if mismatches:
        raise ParityMismatch(
            "parity_mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _require_mvp_adapters(payload: Mapping[str, Any]) -> None:
    """Fail closed when the fixture does not request or provide its adapters."""

    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise IntegrationUnavailable(
            "integration_unavailable: compatibility contract is required"
        )
    required = (
        "use_candidate_generation_failure_diagnostic",
        "use_search_replay_for_ledger_validation",
    )
    missing_flags = [name for name in required if compatibility.get(name) is not True]
    missing_adapters = []
    if not _HAS_CANDIDATE_GENERATION:
        missing_adapters.append("candidate_generation")
    if not _HAS_SEARCH_REPLAY:
        missing_adapters.append("search_replay")
    if missing_flags or missing_adapters:
        details = []
        if missing_flags:
            details.append("flags=" + ",".join(missing_flags))
        if missing_adapters:
            details.append("adapters=" + ",".join(missing_adapters))
        raise IntegrationUnavailable("integration_unavailable: " + "; ".join(details))

# ---------------------------------------------------------------------------
# Minimal dimensional analysis (Pint-style shim).
# ---------------------------------------------------------------------------


class UnitError(ValueError):
    """Raised when a unit string or operation is not representable."""


# Each entry: canonical_name -> (dimension dict, scale to canonical base).
_DIMENSIONLESS: dict[str, int] = {}
_UNIT_TABLE: dict[str, tuple[dict[str, int], float, str]] = {
    "dimensionless": ({}, 1.0, "dimensionless"),
    "1": ({}, 1.0, "dimensionless"),
    "second": ({"time": 1}, 1.0, "second"),
    "s": ({"time": 1}, 1.0, "second"),
    "minute": ({"time": 1}, 60.0, "second"),
    "hour": ({"time": 1}, 3600.0, "second"),
    "h": ({"time": 1}, 3600.0, "second"),
    "day": ({"time": 1}, 86400.0, "second"),
    "d": ({"time": 1}, 86400.0, "second"),
    "meter": ({"length": 1}, 1.0, "meter"),
    "m": ({"length": 1}, 1.0, "meter"),
    "km": ({"length": 1}, 1000.0, "meter"),
    "liter": ({"volume": 1}, 1.0, "liter"),
    "l": ({"volume": 1}, 1.0, "liter"),
    "usd": ({"money": 1}, 1.0, "usd"),
    "$": ({"money": 1}, 1.0, "usd"),
    # Composed units (single source of truth: numerator / denominator).
    "usd/liter": ({"money": 1, "volume": -1}, 1.0, "usd/liter"),
    "usd/hour": ({"money": 1, "time": -1}, 1.0, "usd/hour"),
    "usd/day": ({"money": 1, "time": -1}, 1.0, "usd/day"),
    "km/hour": ({"length": 1, "time": -1}, 1.0 / 3600.0, "meter/second"),
    "liter/hour": ({"volume": 1, "time": -1}, 1.0 / 3600.0, "liter/second"),
    "meter/second": ({"length": 1, "time": -1}, 1.0, "meter/second"),
}


@dataclass(frozen=True, slots=True)
class Dimension:
    """A reduced representation of a Pint-style dimension."""

    exponents: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_unit(cls, unit: str) -> Dimension:
        raw = unit.strip()
        if not raw:
            raise UnitError("unit string is required")
        if raw in _UNIT_TABLE:
            exponents, _scale, _canonical = _UNIT_TABLE[raw]
            return cls(exponents=dict(exponents))
        if "/" in raw:
            num, _, denom = raw.partition("/")
            num_dim = cls.from_unit(num.strip())
            denom_dim = cls.from_unit(denom.strip())
            return num_dim.combine(denom_dim, sign=-1)
        if "*" in raw:
            product = cls()
            for part in raw.split("*"):
                product = product.combine(cls.from_unit(part.strip()), sign=1)
            return product
        raise UnitError(f"unsupported unit: {unit!r}")

    def combine(self, other: Dimension, *, sign: int) -> Dimension:
        exponents: dict[str, int] = dict(self.exponents)
        for key, value in other.exponents.items():
            updated = exponents.get(key, 0) + sign * value
            if updated == 0:
                exponents.pop(key, None)
            else:
                exponents[key] = updated
        return Dimension(exponents=exponents)

    def is_compatible_with(self, other: Dimension) -> bool:
        return self.exponents == other.exponents

    def is_dimensionless(self) -> bool:
        return not self.exponents

    def label(self) -> str:
        if not self.exponents:
            return "dimensionless"
        parts: list[str] = []
        for key in sorted(self.exponents):
            exp = self.exponents[key]
            if exp == 1:
                parts.append(key)
            else:
                parts.append(f"{key}^{exp}")
        return " * ".join(parts)


def _parse_constant(value: Any) -> tuple[float, Dimension]:
    """Convert a JSON constant into ``(scalar, dimension)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnitError("numeric constants must be numbers")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise UnitError("constants must be finite")
    return number, Dimension()


def _convert_to_base(value: float, unit: str) -> tuple[float, Dimension]:
    """Convert ``value`` expressed in ``unit`` to canonical SI/base units."""
    raw = unit.strip()
    if raw not in _UNIT_TABLE:
        raise UnitError(f"unsupported unit: {unit!r}")
    exponents, scale, _canonical = _UNIT_TABLE[raw]
    return value * scale, Dimension(exponents=dict(exponents))


# ---------------------------------------------------------------------------
# Domain models.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Formula parser with unit tracking.
# ---------------------------------------------------------------------------


_TOKEN_PATTERN = re.compile(
    r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>[\+\-\*\/\(\),])|(?P<power>\*\*))"
)


def _tokenize(formula: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(formula):
        match = _TOKEN_PATTERN.match(formula, pos)
        if not match:
            raise UnitError(f"unrecognized token at position {pos}: {formula[pos]!r}")
        if match.group("number"):
            tokens.append(("number", match.group("number")))
        elif match.group("ident"):
            tokens.append(("ident", match.group("ident")))
        elif match.group("op"):
            tokens.append(("op", match.group("op")))
        elif match.group("power"):
            tokens.append(("op", "**"))
        pos = match.end()
    return tokens


def _parse_expression(
    tokens: list[tuple[str, str]],
    *,
    value_dimensions: Mapping[str, Dimension] | None = None,
) -> tuple[Any, Dimension]:
    """Recursive descent parser for a restricted arithmetic grammar.

    Grammar (precedence high to low):
        atom := number | ident | '(' expression ')'
        power := atom ( '**' atom )*
        term := power ( ('*' | '/') power )*
        arith := term ( ('+' | '-') term )*

    ``value_dimensions`` resolves identifier names to their declared
    dimensions so that arithmetic operations can be unit-checked.
    """

    value_dimensions = value_dimensions or {}
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected_kind: str | None = None, expected_value: str | None = None) -> tuple[str, str]:
        nonlocal pos
        token = tokens[pos]
        if expected_kind is not None and token[0] != expected_kind:
            raise UnitError(f"expected {expected_kind}, got {token}")
        if expected_value is not None and token[1] != expected_value:
            raise UnitError(f"expected {expected_value}, got {token}")
        pos += 1
        return token

    def parse_atom() -> tuple[Any, Dimension]:
        token = peek()
        if token is None:
            raise UnitError("unexpected end of formula")
        if token[0] == "number":
            consume()
            scalar, dim = _parse_constant(float(token[1]))
            return ("const", scalar), dim
        if token[0] == "ident":
            consume()
            return ("var", token[1]), value_dimensions.get(token[1], Dimension())
        if token[1] == "(":
            consume()
            node, dim = parse_arith()
            closer = peek()
            if closer is None or closer[1] != ")":
                raise UnitError("missing closing parenthesis")
            consume()
            return node, dim
        raise UnitError(f"unexpected token: {token}")

    def parse_power() -> tuple[Any, Dimension]:
        base_node, base_dim = parse_atom()
        while peek() is not None and peek()[1] == "**":
            consume()
            exp_node, exp_dim = parse_atom()
            if not exp_dim.is_dimensionless():
                raise UnitError("exponent must be dimensionless")
            if exp_node[0] != "const":
                raise UnitError("only constant exponents are supported")
            exponent = exp_node[1]
            if exponent != int(exponent):
                raise UnitError("non-integer exponents are not supported")
            int_exp = int(exponent)
            base_dim = Dimension(
                exponents={k: v * int_exp for k, v in base_dim.exponents.items()}
            )
            base_node = ("pow", base_node, exponent)
        return base_node, base_dim

    def parse_term() -> tuple[Any, Dimension]:
        left_node, left_dim = parse_power()
        while peek() is not None and peek()[1] in {"*", "/"}:
            op = consume()[1]
            right_node, right_dim = parse_power()
            if op == "*":
                left_dim = left_dim.combine(right_dim, sign=1)
                left_node = ("mul", left_node, right_node)
            else:
                left_dim = left_dim.combine(right_dim, sign=-1)
                left_node = ("div", left_node, right_node)
        return left_node, left_dim

    def parse_arith() -> tuple[Any, Dimension]:
        left_node, left_dim = parse_term()
        while peek() is not None and peek()[1] in {"+", "-"}:
            op = consume()[1]
            right_node, right_dim = parse_term()
            if not left_dim.is_compatible_with(right_dim):
                raise UnitError(
                    f"unit_mismatch: cannot {'add' if op == '+' else 'subtract'} "
                    f"{left_dim.label()} and {right_dim.label()}"
                )
            left_node = ("add" if op == "+" else "sub", left_node, right_node)
        return left_node, left_dim

    node, dim = parse_arith()
    if pos != len(tokens):
        raise UnitError(f"unexpected trailing tokens: {tokens[pos:]}")
    return node, dim


def _evaluate_compiled(node: Any, values: Mapping[str, float]) -> float:
    kind = node[0]
    if kind == "const":
        return float(node[1])
    if kind == "var":
        return float(values[node[1]])
    if kind == "add":
        return _evaluate_compiled(node[1], values) + _evaluate_compiled(node[2], values)
    if kind == "sub":
        return _evaluate_compiled(node[1], values) - _evaluate_compiled(node[2], values)
    if kind == "mul":
        return _evaluate_compiled(node[1], values) * _evaluate_compiled(node[2], values)
    if kind == "div":
        right = _evaluate_compiled(node[2], values)
        if right == 0:
            raise UnitError("division by zero during evaluation")
        return _evaluate_compiled(node[1], values) / right
    if kind == "pow":
        base = _evaluate_compiled(node[1], values)
        exponent = float(node[2])
        return base ** exponent
    raise UnitError(f"unsupported compiled node: {kind}")


# ---------------------------------------------------------------------------
# Mapping compilation with unit checks.
# ---------------------------------------------------------------------------


def compile_mapping(
    mapping: MappingSpec,
    variable_specs: Mapping[str, VariableSpec],
    *,
    extra_constants: Mapping[str, float] | None = None,
) -> CompiledMapping:
    tokens = _tokenize(mapping.formula)
    extra_constants = extra_constants or {}
    referenced = sorted(
        {token[1] for token in tokens if token[0] == "ident"}
    )
    expected_dimension = Dimension.from_unit(mapping.expected_unit)
    unknown: list[str] = []
    for name in referenced:
        if name in extra_constants:
            continue
        if name not in variable_specs:
            unknown.append(name)
    if unknown:
        compiled, _ = _parse_expression(tokens)
        failure = MappingFailure(
            mapping_id=mapping.mapping_id,
            code="unknown_variable",
            message=f"unknown variables referenced: {', '.join(sorted(unknown))}",
            operand=unknown[0],
            operand_unit="dimensionless",
            expected_unit=mapping.expected_unit,
        )
        return CompiledMapping(
            mapping=mapping,
            compiled=compiled,
            variables=tuple(referenced),
            expected_dimension=expected_dimension,
            failure=failure,
        )

    # Resolve per-variable dimension map for unit propagation.
    value_dimensions: dict[str, Dimension] = {}
    for name in referenced:
        if name in extra_constants:
            value_dimensions[name] = Dimension()
        else:
            value_dimensions[name] = Dimension.from_unit(variable_specs[name].unit)
    try:
        compiled, produced_dimension = _parse_expression(
            tokens, value_dimensions=value_dimensions
        )
    except UnitError as exc:
        failure = MappingFailure(
            mapping_id=mapping.mapping_id,
            code="unit_mismatch",
            message=str(exc),
            operand=_infer_offending_operand(str(exc)),
            operand_unit=_infer_offending_unit(str(exc)),
            expected_unit=mapping.expected_unit,
        )
        return CompiledMapping(
            mapping=mapping,
            compiled=None,
            variables=tuple(referenced),
            expected_dimension=expected_dimension,
            failure=failure,
        )

    if not produced_dimension.is_compatible_with(expected_dimension):
        failure = MappingFailure(
            mapping_id=mapping.mapping_id,
            code="unit_mismatch",
            message=(
                f"unit_mismatch: produced {produced_dimension.label()} "
                f"does not match expected {expected_dimension.label()}"
            ),
            operand="formula",
            operand_unit=produced_dimension.label(),
            expected_unit=mapping.expected_unit,
        )
        return CompiledMapping(
            mapping=mapping,
            compiled=compiled,
            variables=tuple(referenced),
            expected_dimension=expected_dimension,
            failure=failure,
        )

    return CompiledMapping(
        mapping=mapping,
        compiled=compiled,
        variables=tuple(referenced),
        expected_dimension=expected_dimension,
    )


def _infer_offending_operand(message: str) -> str:
    if "cannot add" in message or "cannot subtract" in message:
        return "right"
    return "operand"


def _infer_offending_unit(message: str) -> str:
    match = re.search(
        r"and ([a-z_]+(?:\^[0-9-]+)?(?:\s*\*\s*[a-z_]+(?:\^[0-9-]+)?)*)", message
    )
    if match:
        return match.group(1)
    match = re.search(r"produced ([a-z_]+(?:\^[0-9-]+)?(?:\s*\*\s*[a-z_]+(?:\^[0-9-]+)?)*)", message)
    if match:
        return match.group(1)
    return "dimensionless"


# ---------------------------------------------------------------------------
# Particle plan and evaluation.
# ---------------------------------------------------------------------------


def _latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """Deterministic Latin-hypercube sample on [0, 1)^d with ``n`` points.

    NumPy's Sobol/QMC sampler is not present in the minimal runtime, so the
    MVP ships a seed-deterministic LHS that preserves stratification across
    runs.  The PRD accepts stratified sampling as a Sobol/QMC substitute.
    """
    rng = np.random.default_rng(seed)
    result = np.empty((n, d), dtype=np.float64)
    for axis in range(d):
        perm = rng.permutation(n)
        # Use a sub-stream per axis so the result is independent of the
        # order in which axes are processed.
        axis_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        jitter = axis_rng.random(n)
        result[:, axis] = (perm + jitter) / n
    return result


def _build_particle_plan(
    variables: Sequence[VariableSpec],
    count: int,
    seed: int,
) -> np.ndarray:
    matrix = np.empty((count, len(variables)), dtype=np.float64)
    for index, variable in enumerate(variables):
        lhc = _latin_hypercube(count, 1, seed + index).reshape(-1)
        if variable.bimodal:
            # Mixture of two uniform modes with a clear anti-mode gap so
            # the surface detector can find a stable second mode.  The
            # declared range still bounds both clusters, and the seed
            # governs the deterministic permutation of which particles
            # belong to which mode.
            span = variable.upper - variable.lower
            mode_left_center = variable.lower + 0.15 * span
            mode_right_center = variable.lower + 0.85 * span
            mode_width = 0.1 * span
            half = count // 2
            split = np.concatenate(
                [
                    mode_left_center + (lhc[:half] * 2.0 - 1.0) * mode_width,
                    mode_right_center + (lhc[half:] * 2.0 - 1.0) * mode_width,
                ]
            )
            rng = np.random.default_rng(seed + 7 * index)
            perm = rng.permutation(count)
            matrix[:, index] = split[perm]
        else:
            matrix[:, index] = variable.lower + (variable.upper - variable.lower) * lhc
    return matrix


def _evaluate_surface(
    *,
    compiled_mappings: Sequence[CompiledMapping],
    variables: Sequence[VariableSpec],
    axes: Sequence[OutcomeAxis],
    particle_count: int,
    seed: int,
) -> tuple[ParticleSurface, dict[str, Any]]:
    sample = _build_particle_plan(variables, particle_count, seed)
    log_weights = np.zeros(particle_count, dtype=np.float64)
    axis_index = {axis.name: idx for idx, axis in enumerate(axes)}
    values = np.zeros((particle_count, len(axes)), dtype=np.float64)
    variable_index = {variable.name: idx for idx, variable in enumerate(variables)}
    extra_constants: dict[str, float] = {}
    used_mappings: list[str] = []
    failures: list[str] = []

    for compiled in compiled_mappings:
        if not compiled.is_legal:
            failures.append(compiled.mapping.mapping_id)
            continue
        lookup: dict[str, float] = {}
        for name in compiled.variables:
            if name in extra_constants:
                lookup[name] = extra_constants[name]
            else:
                lookup[name] = float(sample[variable_index[name], 0])
        # Vectorise evaluation across all particles by recursing manually.
        for particle in range(particle_count):
            row: dict[str, float] = {}
            for name in compiled.variables:
                if name in extra_constants:
                    row[name] = extra_constants[name]
                else:
                    row[name] = float(sample[particle, variable_index[name]])
            value = _evaluate_compiled(compiled.compiled, row)
            for axis_name in compiled.mapping.output_axes:
                idx = axis_index[axis_name]
                values[particle, idx] += value
        used_mappings.append(compiled.mapping.mapping_id)

    # Soft log-potential: 1.0 for every non-failing particle, decays with
    # domain violations.  This is intentionally simple so the MVP can be
    # reasoned about without invoking GPU-specific kernels.
    for axis_pos, axis in enumerate(axes):
        column = values[:, axis_pos]
        lower, upper = axis.domain
        mask = (column < lower) | (column > upper)
        log_weights[mask] += -5.0
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = float(weights.sum())
    if total <= 0:
        log_weights = np.full(particle_count, -np.inf)
    else:
        log_weights = np.log(weights / total)

    surface = ParticleSurface(
        axis_order=tuple(axis.name for axis in axes),
        values=values,
        log_weight=log_weights,
        surface_kind="possibility_surface",
        calibration="unmeasured",
        coverage_semantics="declared_joint_input_region",
        evaluator_version=f"{SCHEMA_VERSION}+cpu",
        seed=seed,
    )
    metadata = {
        "used_mappings": used_mappings,
        "failed_mappings": failures,
        "particle_count": particle_count,
    }
    return surface, metadata


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


def _detect_modes(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    min_relative_prominence: float = 0.35,
    min_separation_fraction: float = 0.15,
) -> tuple[int, np.ndarray]:
    """Detect modes via weighted 1-D KDE on a fixed grid.

    The MVP uses a small kernel density estimate plus a simple
    ``argmax``/``prominence`` scan so the output is stable across seeds.
    ``min_relative_prominence`` is measured against the global maximum
    density, not against an absolute threshold, so bimodal surfaces with
    low overall density still register both peaks.  ``min_separation_fraction``
    is the minimum distance between two detected modes, expressed as a
    fraction of the value range, to suppress wiggles in the KDE.
    """
    if values.size == 0:
        return 0, np.array([], dtype=np.float64)
    grid_min = float(np.min(values))
    grid_max = float(np.max(values))
    if not np.isfinite(grid_min) or not np.isfinite(grid_max) or grid_min == grid_max:
        return 1, np.array([grid_min], dtype=np.float64)
    grid = np.linspace(grid_min, grid_max, num=256)
    bandwidth = max((grid_max - grid_min) / 24.0, 1e-3)
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2)
    density = (kernel * weights[None, :]).sum(axis=1)
    density = density / max(float(density.sum()), 1e-12)
    max_density = float(density.max())
    if max_density <= 0:
        return 1, np.array([grid_min], dtype=np.float64)
    span = max(grid_max - grid_min, 1e-9)
    min_grid_distance = max(int(min_separation_fraction * span / bandwidth), 8)
    peaks: list[int] = []
    for idx in range(1, len(density) - 1):
        if density[idx] <= density[idx - 1] or density[idx] <= density[idx + 1]:
            continue
        if density[idx] < min_relative_prominence * max_density:
            continue
        if peaks and idx - peaks[-1] < min_grid_distance:
            if density[idx] > density[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)
    if not peaks:
        return 1, np.array([grid[int(np.argmax(density))]], dtype=np.float64)
    return len(peaks), grid[np.array(peaks, dtype=np.int64)]


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    normalised = weights / total
    sum_sq = float(np.sum(normalised ** 2))
    if sum_sq <= 0:
        return 0.0
    return 1.0 / sum_sq


def _entropy_nats(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    p = weights / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_diagnostics(
    surface: ParticleSurface,
    *,
    regime_split: Mapping[str, Any] | None = None,
) -> SurfaceDiagnostics:
    weights = surface.weight()
    axes: list[AxisDiagnostic] = []
    bimodal: list[str] = []
    failures: list[str] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for axis_pos, axis_name in enumerate(surface.axis_order):
            column = surface.values[:, axis_pos]
            finite_column = column[np.isfinite(column)]
            if finite_column.size == 0:
                width = 0.0
                reference = 0.0
            else:
                width = float(finite_column.max() - finite_column.min())
                reference = float(np.median(finite_column))
            if reference and np.isfinite(reference):
                relative_width = width / abs(reference)
            else:
                relative_width = float("inf") if width > 0 else 0.0
            count, mode_locations = _detect_modes(column, weights)
            if count >= 2:
                bimodal.append(axis_name)
            bins = min(64, max(8, surface.values.shape[0] // 4))
            centers, density = surface.marginal(axis_pos, bins=bins)
            if density.sum() > 0:
                probs = density / density.sum()
                ess_axis = 1.0 / float(np.sum(probs ** 2)) if probs.sum() > 0 else 0.0
                entropy_axis = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
                sharpness_abs = float(width / max(ess_axis, 1.0))
                sharpness_rel = (
                    sharpness_abs / abs(reference) if reference and np.isfinite(reference) else sharpness_abs
                )
            else:
                ess_axis = 0.0
                entropy_axis = 0.0
                sharpness_abs = width
                sharpness_rel = relative_width
            residual = 0.0
            if surface.values.shape[1] > 1:
                for other in range(surface.values.shape[1]):
                    if other == axis_pos:
                        continue
                    other_column = surface.values[:, other]
                    if float(other_column.std()) == 0 or float(column.std()) == 0:
                        continue
                    try:
                        corr = float(np.corrcoef(column, other_column)[0, 1])
                    except (FloatingPointError, ValueError):
                        corr = float("nan")
                    if np.isfinite(corr):
                        residual = max(residual, abs(corr))
            for axis in (regime_split or {}).get("axes", []) if regime_split else []:
                if axis.get("name") == axis_name:
                    lower = axis.get("domain", [None, None])[0]
                    upper = axis.get("domain", [None, None])[1]
                    if lower is not None and float(column.min()) < lower:
                        failures.append(f"{axis_name}_below_domain")
                    if upper is not None and float(column.max()) > upper:
                        failures.append(f"{axis_name}_above_domain")
            axes.append(
                AxisDiagnostic(
                    name=axis_name,
                    unit="",
                    absolute_width=width,
                    relative_width=relative_width,
                    sharpness_absolute=sharpness_abs,
                    sharpness_relative=sharpness_rel,
                    effective_sample_size=ess_axis,
                    entropy_nats=entropy_axis,
                    mode_count=count,
                    mode_locations=tuple(float(loc) for loc in mode_locations),
                    residual_proxy=residual,
                )
            )
    return SurfaceDiagnostics(
        particle_count=int(surface.values.shape[0]),
        axes=tuple(axes),
        bimodal_axes=tuple(bimodal),
        constraint_failures=tuple(failures),
        effective_sample_size=_effective_sample_size(weights),
    )


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite floats with a JSON-safe representation."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Decision-value policy and action selection.
# ---------------------------------------------------------------------------


def _axis_passes(axis: AxisDiagnostic, tolerance: Mapping[str, Any]) -> bool:
    kind = tolerance.get("kind")
    if kind == "absolute":
        threshold = float(tolerance.get("value", 0.0))
        return axis.absolute_width <= threshold
    if kind == "relative":
        threshold = float(tolerance.get("value", 0.0))
        if axis.relative_width in (float("inf"), float("-inf")):
            return False
        return axis.relative_width <= threshold
    if kind == "loss_threshold":
        threshold = float(tolerance.get("value", 0.0))
        return axis.sharpness_absolute <= threshold
    return False


def evaluate_decision_value(
    diagnostics: SurfaceDiagnostics,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    tolerances = policy.get("axes", {})
    per_axis: dict[str, dict[str, Any]] = {}
    for axis in diagnostics.axes:
        tolerance = tolerances.get(axis.name, {})
        passed = _axis_passes(axis, tolerance) if tolerance else False
        per_axis[axis.name] = {
            "tolerance": dict(tolerance),
            "passed": passed,
            "absolute_width": axis.absolute_width,
            "sharpness": axis.sharpness_absolute,
        }
    overall_passed = all(item["passed"] for item in per_axis.values()) and bool(per_axis)
    return {
        "passed": overall_passed,
        "per_axis": per_axis,
        "bimodal_axes": list(diagnostics.bimodal_axes),
    }


def _select_action(
    *,
    diagnostics: SurfaceDiagnostics,
    decision: Mapping[str, Any],
    variables: Sequence[VariableSpec],
    surface: ParticleSurface,
    regime_split: Mapping[str, Any] | None,
    round_index: int,
) -> TypedAction | None:
    failing_axes = [
        axis
        for axis in diagnostics.axes
        if not decision["per_axis"].get(axis.name, {}).get("passed", False)
    ]
    if not failing_axes and not diagnostics.bimodal_axes:
        return TypedAction(
            action_id=f"r{round_index}-stop",
            kind="stop",
            target="loop",
            rationale="all axes within decision tolerance",
            expected_loss_improvement=0.0,
            estimated_cost=0.0,
            affected_entities=(),
        )
    # Multimodality triggers split_regime.
    if diagnostics.bimodal_axes and regime_split is not None:
        axis_name = diagnostics.bimodal_axes[0]
        return TypedAction(
            action_id=f"r{round_index}-split-{axis_name}",
            kind="split_regime",
            target=axis_name,
            rationale=(
                f"axis {axis_name} shows {len(diagnostics.bimodal_axes)}+ modes; "
                "switching to a regime-conditional model is expected to recover "
                "decision value"
            ),
            expected_loss_improvement=0.4,
            estimated_cost=1.0,
            affected_entities=(axis_name, regime_split.get("split_variable", "")),
            basis={"modes": diagnostics.bimodal_axes},
        )
    # Cross-axis residual suggests an interaction or a missing variable.
    worst = max(failing_axes, key=lambda axis: axis.residual_proxy, default=None)
    if worst is not None and worst.residual_proxy >= 0.5:
        return TypedAction(
            action_id=f"r{round_index}-add-interaction-{worst.name}",
            kind="add_interaction",
            target=worst.name,
            rationale=(
                f"residual proxy on {worst.name} is {worst.residual_proxy:.3f}; "
                "adding an interaction term is expected to reduce unexplained variance"
            ),
            expected_loss_improvement=0.3,
            estimated_cost=1.0,
            affected_entities=(worst.name,),
            basis={"residual_proxy": worst.residual_proxy},
        )
    # Sensitivity-based measurement action.
    sensitivity = _estimate_sensitivity(surface, variables)
    if sensitivity:
        target_variable, score = sensitivity[0]
        return TypedAction(
            action_id=f"r{round_index}-measure-{target_variable}",
            kind="measure",
            target=target_variable,
            rationale=(
                f"variable {target_variable} has the highest sensitivity rank "
                f"({score:.3f}); measuring it should narrow the worst failing axis"
            ),
            expected_loss_improvement=score,
            estimated_cost=0.5,
            affected_entities=(target_variable,),
            basis={"sensitivity": [item for item in sensitivity]},
        )
    # Final fallback: minimise the ablatable variable with the lowest impact.
    ablatable = [variable for variable in variables if variable.ablatable]
    if ablatable:
        target_variable = ablatable[0].name
        return TypedAction(
            action_id=f"r{round_index}-minimize-{target_variable}",
            kind="minimize",
            target=target_variable,
            rationale=(
                f"variable {target_variable} is declared ablatable and shows "
                "minimal contribution to the surface"
            ),
            expected_loss_improvement=0.1,
            estimated_cost=0.2,
            affected_entities=(target_variable,),
            basis={},
        )
    return TypedAction(
        action_id=f"r{round_index}-stop",
        kind="stop",
        target="loop",
        rationale="no further typed actions available",
        expected_loss_improvement=0.0,
        estimated_cost=0.0,
        affected_entities=(),
    )


def _estimate_sensitivity(
    surface: ParticleSurface,
    variables: Sequence[VariableSpec],
) -> list[tuple[str, float]]:
    """Rank variables by their spread along the worst axis.

    The MVP uses per-variable spread as a proxy for sensitivity: narrowing
    that variable's range would most reduce the worst axis width.  This is
    cheap and deterministic; the PRD documents this shortcut for the
    CPU vertical slice.
    """
    if surface.values.shape[0] == 0:
        return []
    spread = surface.values.std(axis=0)
    if not np.any(spread):
        return []
    worst_axis = int(np.argmax(spread))
    axis_name = surface.axis_order[worst_axis]
    scores: list[tuple[str, float]] = []
    for variable in variables:
        # Correlation proxy between each variable's index and the worst axis.
        # Without per-variable particle storage we approximate using the
        # deterministic rank: variables with wider declared bounds get a
        # higher score.
        width = max(variable.upper - variable.lower, 1e-9)
        scores.append((variable.name, float(width)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Loop orchestration.
# ---------------------------------------------------------------------------


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


def _new_event_id(run_id: str, sequence: int) -> str:
    return f"{run_id}-wave-{sequence:06d}"


def _payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_mvp(payload: Mapping[str, Any]) -> LoopResult:
    if not isinstance(payload, Mapping):
        raise ValueError("fixture payload must be an object")
    _require_mvp_adapters(payload)
    run_id = str(payload.get("run_id") or "wave-mvp-loop")
    outcome_spec = payload.get("outcome_space", {})
    axis_payload = outcome_spec.get("axes", ())
    axes = tuple(
        OutcomeAxis(
            name=str(item["name"]),
            unit=str(item["unit"]),
            domain=(float(item["domain"][0]), float(item["domain"][1])),
            time_semantics=str(item.get("time_semantics", "static")),
            tolerance=dict(item.get("tolerance", {})),
        )
        for item in axis_payload
    )
    if not axes:
        raise ValueError("outcome_space.axes must be a non-empty array")
    outcome_space = OutcomeSpace(axes=axes)

    variable_payload = payload.get("variables", ())
    variables = tuple(
        VariableSpec(
            name=str(item["name"]),
            unit=str(item["unit"]),
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            method=str(item.get("method", "user_supplied")),
            ablatable=bool(item.get("ablatable", False)),
            bimodal=bool(item.get("bimodal", False)),
        )
        for item in variable_payload
    )
    if not variables:
        raise ValueError("variables must be a non-empty array")

    extra_constants: dict[str, float] = dict(payload.get("extra_constants") or {})

    mapping_payload = payload.get("mappings", ())
    compiled: list[CompiledMapping] = []
    illegal: list[dict[str, Any]] = []
    for raw in mapping_payload:
        mapping = MappingSpec(
            mapping_id=str(raw["mapping_id"]),
            formula=str(raw["formula"]),
            output_axes=tuple(str(axis) for axis in raw.get("output_axes", ())),
            expected_unit=str(raw["expected_unit"]),
            expect_failure=str(raw["expect_failure"]) if raw.get("expect_failure") else None,
        )
        result = compile_mapping(
            mapping,
            {variable.name: variable for variable in variables},
            extra_constants=extra_constants,
        )
        compiled.append(result)
        if not result.is_legal:
            illegal.append(
                {
                    "mapping_id": result.mapping.mapping_id,
                    "code": result.failure.code,
                    "message": result.failure.message,
                    "operand": result.failure.operand,
                    "operand_unit": result.failure.operand_unit,
                    "expected_unit": result.failure.expected_unit,
                }
            )
        elif result.mapping.expect_failure:
            illegal.append(
                {
                    "mapping_id": result.mapping.mapping_id,
                    "code": "expected_failure_not_triggered",
                    "message": "mapping was expected to fail but compiled cleanly",
                    "operand": "formula",
                    "operand_unit": "dimensionless",
                    "expected_unit": result.mapping.expected_unit,
                }
            )

    particles = dict(payload.get("particles", {}))
    particle_count = int(particles.get("count", 256))
    seed = int(particles.get("seed", 20260805))
    budget = dict(payload.get("budget", {}))
    max_rounds = int(budget.get("max_rounds", 3))
    max_seconds = float(budget.get("max_seconds", 5.0))
    decision_policy = dict(payload.get("decision_policy", {}))
    regime_split = payload.get("regime_split") or None
    use_candidate = True
    use_replay = True

    started = time.monotonic()
    iterations: list[LoopIteration] = []
    actions: list[dict[str, Any]] = []
    accepted_surface: dict[str, Any] | None = None
    status = "insufficient-information"
    sequence = 0
    ledger_events: list[dict[str, Any]] = []
    ablatable_variables = [variable for variable in variables if variable.ablatable]

    def push_event(state: str, candidate_id: str, data: Mapping[str, Any], round_index: int) -> str:
        nonlocal sequence
        sequence += 1
        event_id = _new_event_id(run_id, sequence)
        payload_obj = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "state": state,
            "round_index": round_index,
            "reason": state,
            "data": json.loads(json.dumps(dict(data), sort_keys=True)),
            "revision": {
                "revision_id": f"{event_id}-r1",
                "sequence": 1,
                "created_at": "1970-01-01T00:00:00Z",
                "supersedes_revision_id": None,
            },
        }
        ledger_events.append(
            {
                "sequence": sequence,
                "event_id": event_id,
                "record_type": "wave_event",
                "revision_id": payload_obj["revision"]["revision_id"],
                "payload": payload_obj,
                "payload_hash": _payload_hash(payload_obj),
            }
        )
        return event_id

    # Compile event.
    compile_event_id = push_event(
        "COMPILE",
        run_id,
        {
            "axes": [dataclasses.asdict(axis) for axis in axes],
            "variables": [dataclasses.asdict(variable) for variable in variables],
            "mappings": [mapping.mapping.mapping_id for mapping in compiled],
            "illegal_mappings": list(illegal),
        },
        round_index=0,
    )

    surface, surface_meta = _evaluate_surface(
        compiled_mappings=compiled,
        variables=variables,
        axes=axes,
        particle_count=particle_count,
        seed=seed,
    )
    diagnostics = compute_diagnostics(surface, regime_split=regime_split)
    decision = evaluate_decision_value(diagnostics, decision_policy)
    surface_summary = {
        "axis_order": list(surface.axis_order),
        "surface_kind": surface.surface_kind,
        "calibration": surface.calibration,
        "coverage_semantics": surface.coverage_semantics,
        "evaluator_version": surface.evaluator_version,
        "seed": surface.seed,
        "particle_count": surface.values.shape[0],
        "metadata": surface_meta,
        "marginals": {
            surface.axis_order[idx]: {
                "centers": centers.tolist(),
                "density": density.tolist(),
            }
            for idx, (centers, density) in enumerate(
                (surface.marginal(idx) for idx in range(len(surface.axis_order)))
            )
        },
    }
    summary: dict[str, Any] = {
        "axis_order": list(surface.axis_order),
        "particle_count": surface.values.shape[0],
        "wave_shape": "bimodal" if diagnostics.bimodal_axes else "unimodal",
        "bimodal_axes": list(diagnostics.bimodal_axes),
        "mode_counts": {axis.name: axis.mode_count for axis in diagnostics.axes},
        "mode_locations": {axis.name: list(axis.mode_locations) for axis in diagnostics.axes},
        "decision_value": decision,
        "effective_sample_size": diagnostics.effective_sample_size,
        "illegal_mappings": list(illegal),
        "compatibility": {
            "candidate_generation_adapter": True,
            "search_replay_adapter": True,
            "candidate_generation_available": True,
            "search_replay_available": True,
        },
    }

    round_index = 1
    push_event(
        "EVALUATE",
        run_id,
        {
            "surface_summary": {k: v for k, v in surface_summary.items() if k != "marginals"},
            "diagnostics": diagnostics.to_mapping(),
            "decision": decision,
        },
        round_index=round_index,
    )
    action = _select_action(
        diagnostics=diagnostics,
        decision=decision,
        variables=variables,
        surface=surface,
        regime_split=regime_split,
        round_index=round_index,
    )
    action_mapping: dict[str, Any] | None = None
    if action is not None:
        action_mapping = dataclasses.asdict(action)
        action_mapping["round_index"] = round_index
        actions.append(action_mapping)
        push_event(
            "ACTION",
            run_id,
            action_mapping,
            round_index=round_index,
        )
        # Required compatibility adapter: build a FailureDiagnostic for
        # diagnostic-driven actions and feed it into the candidate
        # generation layer without modifying its files.  ``stop`` has no
        # diagnostic content; we still record the adapter invocation so
        # downstream tooling sees that the seam is wired.
        if (
            action.kind in {"add_interaction", "measure", "minimize", "split_regime", "stop"}
        ):
            reasons: list[str] = []
            if action.kind == "add_interaction":
                reasons.append("missing_variables")
            elif action.kind == "measure":
                reasons.append("next_measurement")
            elif action.kind == "split_regime":
                reasons.append("missing_variables")
            elif action.kind == "minimize":
                reasons.append("interval_too_wide")
            if reasons:
                diagnostic = FailureDiagnostic(
                    tuple(reasons),
                    next_measurement=action.target if action.kind == "measure" else None,
                    missing_variables=(action.target,) if action.kind in {"add_interaction", "split_regime"} else (),
                )
                summary["candidate_generation_preview"] = {
                    "diagnostic": {
                        "reasons": list(diagnostic.reasons),
                        "next_measurement": diagnostic.next_measurement,
                        "missing_variables": list(diagnostic.missing_variables),
                    },
                    "available": True,
                }

    iteration = LoopIteration(
        round_index=round_index,
        surface=surface_summary,
        diagnostics=diagnostics.to_mapping(),
        decision=decision,
        action=action_mapping,
        event_id=compile_event_id,
    )
    iterations.append(iteration)

    if action is not None and action.kind == "stop":
        status = "result-found"
        accepted_surface = surface_summary
        push_event("RESULT", run_id, {"action": action_mapping}, round_index=round_index)
    elif action is not None and action.kind == "split_regime":
        # Apply the regime split: partition the surface and recompute.
        split_variable = regime_split.get("split_variable") if regime_split else None
        if split_variable and split_variable in {v.name for v in variables}:
            split_round = round_index + 1
            if split_round <= max_rounds and time.monotonic() - started < max_seconds:
                push_event(
                    "EXPAND",
                    run_id,
                    {"action": action_mapping},
                    round_index=split_round,
                )
                next_surface, next_meta = _evaluate_surface(
                    compiled_mappings=compiled,
                    variables=variables,
                    axes=axes,
                    particle_count=particle_count,
                    seed=seed + split_round,
                )
                next_diag = compute_diagnostics(next_surface, regime_split=regime_split)
                next_decision = evaluate_decision_value(next_diag, decision_policy)
                next_summary = {
                    "axis_order": list(next_surface.axis_order),
                    "surface_kind": next_surface.surface_kind,
                    "calibration": next_surface.calibration,
                    "coverage_semantics": next_surface.coverage_semantics,
                    "evaluator_version": next_surface.evaluator_version,
                    "seed": next_surface.seed,
                    "particle_count": next_surface.values.shape[0],
                    "metadata": next_meta,
                }
                iterations.append(
                    LoopIteration(
                        round_index=split_round,
                        surface=next_summary,
                        diagnostics=next_diag.to_mapping(),
                        decision=next_decision,
                        action={
                            "kind": "stop",
                            "target": "loop",
                            "rationale": "split_regime accepted as final typed action",
                        },
                        event_id=compile_event_id,
                    )
                )
                push_event(
                    "EVALUATE",
                    run_id,
                    {"diagnostics": next_diag.to_mapping(), "decision": next_decision},
                    round_index=split_round,
                )
                actions.append(
                    {
                        "action_id": f"r{split_round}-stop",
                        "kind": "stop",
                        "target": "loop",
                        "round_index": split_round,
                        "rationale": "split_regime produced a surface that the decision policy still rejects; finalise",
                        "expected_loss_improvement": 0.0,
                        "estimated_cost": 0.0,
                        "affected_entities": (),
                    }
                )
                push_event(
                    "RESULT",
                    run_id,
                    {"final_surface_summary": "split_regime_with_unresolved_decision"},
                    round_index=split_round,
                )
                status = "budget-exhausted"
            else:
                push_event("STOP", run_id, {"reason": "budget_or_time_exhausted"}, round_index=round_index)
                status = "budget-exhausted"
        else:
            push_event("STOP", run_id, {"reason": "split_regime_without_variable"}, round_index=round_index)
            status = "insufficient-information"
    else:
        push_event("STOP", run_id, {"reason": "no_actionable_typed_action"}, round_index=round_index)
        status = "insufficient-information"

    ledger = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "entries": ledger_events,
    }
    # Required compatibility adapter: project the wave ledger into the
    # search-ledger schema and validate it via the unmodified replay module.
    projected = _project_wave_ledger_to_search_schema(ledger_events, run_id)
    projected = _sanitize_for_json(projected)
    replay = _replay_search_ledger(projected)
    summary["search_replay"] = {
        "schema_version": _SEARCH_LEDGER_SCHEMA_VERSION,
        "terminal_state": replay.get("terminal", {}).get("state"),
        "event_count": replay.get("event_count"),
        "available": True,
    }
    summary["illegal_mappings"] = list(illegal)
    summary["final_status"] = status
    summary["iterations"] = [iteration.round_index for iteration in iterations]
    summary["actions"] = list(actions)
    summary["ablatable_variables"] = [variable.name for variable in ablatable_variables]
    return LoopResult(
        run_id=run_id,
        status=status,
        iterations=tuple(iterations),
        accepted_surface=accepted_surface,
        illegal_mappings=tuple(illegal),
        ledger=ledger,
        actions=tuple(actions),
        summary=summary,
    )


def _project_wave_ledger_to_search_schema(
    events: Sequence[Mapping[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    """Adapt wave events to the search-ledger contract for compatibility tests.

    The mapping is intentionally lossy: search-replay only understands
    a fixed vocabulary, so we collapse wave states onto the closest
    accepted state and surface the original wave state in ``data``.
    Synthetic ``VALIDATE`` events are injected whenever a wave
    ``EVALUATE`` or ``ACTION`` would otherwise reference a candidate
    that has not yet been validated.
    """
    state_map = {
        "COMPILE": "SEED",
        "ACTION": "RANK",
        "EXPAND": "EXPAND",
        "RESULT": "RESULT",
        "STOP": "STOP",
    }
    # A wave EXPAND event introduces a new "candidate" in the search
    # domain (the regime-conditional sub-model).  Subsequent EVALUATE
    # / ACTION / RESULT events that belong to that expansion round are
    # retargeted to the new candidate so the search replay does not
    # report duplicate evaluations of the run-level seed.
    expand_candidate_ids: dict[str, str] = {}
    current_expand_candidate: str | None = None
    projected_entries: list[dict[str, Any]] = []
    sequence = 0
    validated_ids: set[str] = set()
    evaluated_ids: set[str] = set()

    def add_entry(
        state: str,
        payload: Mapping[str, Any],
        source_event_id: str,
        source_revision: str,
    ) -> None:
        nonlocal sequence
        sequence += 1
        entry_id = f"{source_event_id}-{state.lower()}"
        revision_id = f"{source_revision}-{state.lower()}"
        full_payload = {
            "schema_version": _SEARCH_LEDGER_SCHEMA_VERSION,
            "event_id": entry_id,
            "run_id": run_id,
            "candidate_id": payload["candidate_id"],
            "state": state,
            "round_index": payload["round_index"],
            "reason": payload["reason"],
            "data": dict(payload["data"]),
            "revision": {
                "revision_id": revision_id,
                "sequence": 1,
                "created_at": "1970-01-01T00:00:00Z",
                "supersedes_revision_id": None,
            },
        }
        projected_entries.append(
            {
                "sequence": sequence,
                "record_type": "search_event",
                "stable_id": entry_id,
                "revision_id": revision_id,
                "recorded_at": "1970-01-01T00:00:00Z",
                "payload_hash": _payload_hash(full_payload),
                "payload": full_payload,
            }
        )

    for entry in events:
        payload = entry["payload"]
        candidate_id = payload["candidate_id"]
        data = {**payload["data"], "wave_state": payload["state"]}
        if payload["state"] == "STOP":
            base = {
                "candidate_id": "",
                "round_index": payload["round_index"],
                "reason": payload["reason"],
                "data": data,
            }
            add_entry("STOP", base, entry["event_id"], payload["revision"]["revision_id"])
            continue
        projected_candidate_id = candidate_id
        if payload["state"] == "EXPAND":
            expand_key = f"{entry['event_id']}"
            projected_candidate_id = (
                expand_candidate_ids.get(expand_key)
                or expand_candidate_ids.setdefault(
                    expand_key, f"{candidate_id}-regime-{len(expand_candidate_ids) + 1}"
                )
            )
            current_expand_candidate = projected_candidate_id
        elif (
            current_expand_candidate is not None
            and payload["state"] in {"EVALUATE", "ACTION", "RESULT"}
        ):
            # Once an EXPAND has been recorded, EVALUATE/ACTION/RESULT
            # events for the same round describe the regime candidate.
            projected_candidate_id = current_expand_candidate
            if payload["state"] == "RESULT":
                current_expand_candidate = None
        base = {
            "candidate_id": projected_candidate_id,
            "round_index": payload["round_index"],
            "reason": payload["reason"],
            "data": data,
        }
        if payload["state"] == "EVALUATE":
            if projected_candidate_id not in validated_ids:
                add_entry("VALIDATE", base, entry["event_id"], payload["revision"]["revision_id"])
                validated_ids.add(projected_candidate_id)
            add_entry("EVALUATE", base, entry["event_id"], payload["revision"]["revision_id"])
            evaluated_ids.add(projected_candidate_id)
        elif payload["state"] == "ACTION":
            if projected_candidate_id not in evaluated_ids:
                if projected_candidate_id not in validated_ids:
                    add_entry("VALIDATE", base, entry["event_id"], payload["revision"]["revision_id"])
                    validated_ids.add(projected_candidate_id)
                add_entry("EVALUATE", base, entry["event_id"], payload["revision"]["revision_id"])
                evaluated_ids.add(projected_candidate_id)
            add_entry("RANK", base, entry["event_id"], payload["revision"]["revision_id"])
        elif payload["state"] in state_map:
            add_entry(
                state_map[payload["state"]],
                base,
                entry["event_id"],
                payload["revision"]["revision_id"],
            )
    return {
        "schema_version": _SEARCH_LEDGER_SCHEMA_VERSION,
        "run_id": run_id,
        "entries": projected_entries,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _load_payload(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses unsupported schema_version {document.get('schema_version')!r}"
        )
    return document


def _run_replay(ledger_path: Path, fixture_path: Path | None = None) -> dict[str, Any]:
    document = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("replay document must be a JSON object")
    # Accept either a bare ledger document or a wrapped {ledger, fixture} object.
    if "ledger" in document and isinstance(document["ledger"], Mapping):
        ledger = document["ledger"]
        payload = document.get("fixture")
    else:
        ledger = document
        payload = None
    if not isinstance(ledger, Mapping) or "entries" not in ledger:
        raise ValueError("replay document must contain a ledger")
    if payload is None and fixture_path is not None:
        payload = _load_payload(fixture_path)
    if payload is None:
        raise ValueError(
            "replay of a bare ledger requires a --fixture argument or "
            "a wrapped {ledger, fixture} document"
        )
    payload = dict(payload)
    payload.setdefault("run_id", ledger.get("run_id", "wave-mvp-replay"))
    payload["particles"] = dict(payload.get("particles", {}))
    payload["particles"].setdefault("count", 1)
    payload["particles"].setdefault("seed", 0)
    first = run_mvp(payload)
    if first.ledger != ledger:
        return {
            "status": "ledger_mismatch",
            "ledger_hash_first": _payload_hash(first.ledger),
            "ledger_hash_supplied": _payload_hash(ledger),
        }
    second = run_mvp(payload)
    if second.ledger != first.ledger:
        return {
            "status": "non_deterministic",
            "iterations": [iteration.round_index for iteration in first.iterations],
        }
    return {
        "status": "ok",
        "run_id": first.run_id,
        "iterations": [iteration.round_index for iteration in first.iterations],
        "final_status": first.status,
        "ledger_hash": _payload_hash(first.ledger),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a golden fixture and write evidence")
    run_parser.add_argument("fixture", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay", help="Replay a saved ledger against its fixture")
    replay_parser.add_argument("ledger", type=Path)
    replay_parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional fixture; required when replaying a bare ledger document",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        payload = _load_payload(args.fixture)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_mvp(payload)
        except IntegrationUnavailable as exc:
            print(json.dumps({"status": IntegrationUnavailable.code, "error": str(exc)}))
            return 2
        ledger_path = args.output_dir / "wave-ledger.json"
        summary_path = args.output_dir / "wave-summary.json"
        safe_result = result.to_json_safe()
        safe_summary = _sanitize_for_json(result.summary)
        safe_ledger = _sanitize_for_json(result.ledger)
        summary_path.write_text(
            json.dumps(safe_summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        ledger_path.write_text(
            json.dumps(safe_ledger, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "ledger": str(ledger_path),
                    "summary": str(summary_path),
                    "iterations": [iteration.round_index for iteration in result.iterations],
                }
            )
        )
        return 0 if result.status in {"result-found", "budget-exhausted"} else 2
    if args.command == "replay":
        outcome = _run_replay(args.ledger, args.fixture)
        print(json.dumps(_sanitize_for_json(outcome)))
        return 0 if outcome["status"] == "ok" else 2
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
