"""Versioned joint-wave schemas with dimension-safe mappings.

This module is the typed dataclass counterpart of the JSON-friendly
validators in :mod:`aie_decision.wave_loop`.  It exists so that any
caller (CLI, Agent SDK, test, GPU adapter) can build a narrow typed
contract without re-parsing JSON.  All numeric values are real; missing
values remain missing.

The dimension registry lives in :mod:`aie_decision._dimensions`; this
module re-exports the public dimension symbols for backward compatibility.
Cross-dimension arithmetic is rejected at compile time so that illegal
operations surface before any particle is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Mapping, Sequence

from ._dimensions import (  # noqa: F401 — re-exported for callers
    DIMENSIONLESS,
    DimensionMismatchError,
    JointSchemaError,
    UnknownUnitError,
    _UNIT_TABLE,
    canonical_unit,
    compatible_units,
    conversion_factor,
    dimension_of_unit,
    known_dimensions,
    normalize_to_canonical,
    parse_composite_dimension,
)
from .factor_ir import (
    FACTOR_IR_VERSION,
    DeterministicTransform,
    FactorIR,
    FactorIRError,
    compile_axis_transform,
    compile_factor_ir,
)


# ---------------------------------------------------------------------------
# Schema versions
# ---------------------------------------------------------------------------

JOINT_SCHEMA_VERSION = "joint-schema/v1"
EVIDENCE_SCHEMA_VERSION = "evidence/v1"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VariableStatus(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    BOUNDED = "bounded"
    MISSING = "missing"


class MappingKind(StrEnum):
    FORMULA = "formula"
    LIKELIHOOD = "likelihood"


class EpistemicType(StrEnum):
    OBSERVED_EVENT = "observed_event"
    PRIMARY_RECORD = "primary_record"
    ATTRIBUTED_STATEMENT = "attributed_statement"
    INFERENCE = "inference"
    CAUSAL_CLAIM = "causal_claim"
    FORECAST = "forecast"
    EVALUATION = "evaluation"
    RHETORIC = "rhetoric"
    OMISSION = "omission"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeAxis:
    """A single outcome axis of the joint result space."""

    schema_version: str
    axis_id: str
    name: str
    unit: str
    domain: tuple[float, float] | None = None
    time_semantics: str | None = None
    absolute_tolerance: float | None = None
    reference_value: float | None = None
    decision_useful: bool = True

    def __post_init__(self) -> None:
        if not self.axis_id.strip():
            raise JointSchemaError("OutcomeAxis.axis_id is required")
        if not self.name.strip():
            raise JointSchemaError("OutcomeAxis.name is required")
        dimension_of_unit(self.unit)
        if self.domain is not None:
            lower, upper = self.domain
            if not isfinite(lower) or not isfinite(upper) or lower > upper:
                raise JointSchemaError("OutcomeAxis.domain must be finite and ordered")
        if self.absolute_tolerance is not None and not isfinite(self.absolute_tolerance):
            raise JointSchemaError("OutcomeAxis.absolute_tolerance must be finite")
        if self.reference_value is not None and not isfinite(self.reference_value):
            raise JointSchemaError("OutcomeAxis.reference_value must be finite")


@dataclass(frozen=True, slots=True)
class OutcomeSpace:
    """Container of jointly declared outcome axes."""

    schema_version: str
    question_id: str
    axes: tuple[OutcomeAxis, ...]

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise JointSchemaError("OutcomeSpace.question_id is required")
        if not self.axes:
            raise JointSchemaError("OutcomeSpace.axes must be non-empty")
        ids = [axis.axis_id for axis in self.axes]
        if len(set(ids)) != len(ids):
            raise JointSchemaError("OutcomeAxis.axis_id values must be unique")


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """A declared input variable with bounded support or an explicit missing state."""

    schema_version: str
    name: str
    unit: str
    status: VariableStatus
    lower: float | None = None
    upper: float | None = None
    method: str = "user_supplied_90_percent_interval"
    evidence_atom_id: str | None = None
    time_semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise JointSchemaError("VariableSpec.name is required")
        if not self.method.strip():
            raise JointSchemaError("VariableSpec.method is required")
        dimension_of_unit(self.unit)
        if self.status is VariableStatus.MISSING:
            if self.lower is not None or self.upper is not None:
                raise JointSchemaError("missing variables must not declare bounds")
        else:
            if self.lower is None or self.upper is None:
                raise JointSchemaError(
                    "non-missing variables must declare both lower and upper"
                )
            if not isfinite(self.lower) or not isfinite(self.upper):
                raise JointSchemaError("VariableSpec bounds must be finite")
            if self.lower > self.upper:
                raise JointSchemaError("VariableSpec.lower must not exceed upper")


@dataclass(frozen=True, slots=True)
class Evidence:
    """An auditable evidence atom referenced by variables or mappings."""

    schema_version: str
    evidence_atom_id: str
    claim: str
    epistemic_type: EpistemicType
    source_id: str
    independence_group: str | None = None
    truth_confidence: float | None = None
    extraction_confidence: float | None = None
    modality: str | None = None
    transformation: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_atom_id.strip():
            raise JointSchemaError("Evidence.evidence_atom_id is required")
        if not self.claim.strip():
            raise JointSchemaError("Evidence.claim is required")
        if not self.source_id.strip():
            raise JointSchemaError("Evidence.source_id is required")
        for name in ("truth_confidence", "extraction_confidence"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise JointSchemaError(
                    f"Evidence.{name} must be between 0 and 1 when present"
                )


@dataclass(frozen=True, slots=True)
class MappingSpec:
    """A deterministic or likelihood mapping from variables to one axis."""

    schema_version: str
    mapping_id: str
    kind: MappingKind
    variables: tuple[str, ...]
    result_axis: str
    expression: str | None = None
    observation: tuple[float, float] | None = None
    observation_scale: float | None = None
    evidence_atom_ids: tuple[str, ...] = ()
    direction: str = "support"
    applicability: str | None = None

    def __post_init__(self) -> None:
        if not self.mapping_id.strip():
            raise JointSchemaError("MappingSpec.mapping_id is required")
        if not self.result_axis.strip():
            raise JointSchemaError("MappingSpec.result_axis is required")
        if not self.variables:
            raise JointSchemaError("MappingSpec.variables must be non-empty")
        for name in self.variables:
            if not isinstance(name, str) or not name.strip():
                raise JointSchemaError("MappingSpec.variables must be non-empty strings")
        if self.direction not in {"support", "penalty"}:
            raise JointSchemaError("MappingSpec.direction must be 'support' or 'penalty'")
        if self.kind is MappingKind.FORMULA:
            if not self.expression or not self.expression.strip():
                raise JointSchemaError("FORMULA mapping requires expression")
            if self.observation is not None or self.observation_scale is not None:
                raise JointSchemaError("FORMULA mapping cannot declare observation")
        elif self.kind is MappingKind.LIKELIHOOD:
            if len(self.variables) != 1:
                raise JointSchemaError("LIKELIHOOD mapping requires exactly one variable")
            if self.expression:
                raise JointSchemaError("LIKELIHOOD mapping cannot declare expression")
            if self.observation is None or self.observation_scale is None:
                raise JointSchemaError(
                    "LIKELIHOOD mapping requires observation and observation_scale"
                )
            obs_lower, obs_upper = self.observation
            if not isfinite(obs_lower) or not isfinite(obs_upper) or obs_lower > obs_upper:
                raise JointSchemaError("LIKELIHOOD observation must be finite and ordered")
            if not isfinite(self.observation_scale) or self.observation_scale <= 0:
                raise JointSchemaError("LIKELIHOOD observation_scale must be positive")


# ---------------------------------------------------------------------------
# Compile result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledMapping:
    """A ``MappingSpec`` after dimension validation and IR compilation.

    For FORMULA mappings, exactly one of *factor_ir* or *transform* is set:

    * *factor_ir* — the formula is dimensionless and contributes a
      log-potential weight.
    * *transform* — the formula is dimensional and its output matches the
      target axis dimension; it computes the axis value directly.
    """

    schema_version: str
    mapping: MappingSpec
    axis: OutcomeAxis
    variables: tuple[VariableSpec, ...]
    variable_dimensions: dict[str, str]
    factor_ir: FactorIR | None = None
    transform: DeterministicTransform | None = None

    def __post_init__(self) -> None:
        if not self.variables:
            raise JointSchemaError("CompiledMapping.variables must be non-empty")
        declared = {variable.name for variable in self.variables}
        referenced = set(self.mapping.variables)
        if referenced != declared:
            raise JointSchemaError(
                f"CompiledMapping variable set must match mapping.variables: "
                f"{declared} vs {referenced}"
            )
        forms = sum(1 for v in (self.factor_ir, self.transform) if v is not None)
        if self.mapping.kind is MappingKind.FORMULA and forms != 1:
            raise JointSchemaError(
                "CompiledMapping with FORMULA kind must set exactly one of factor_ir or transform"
            )

    def log_potential(self, values: Mapping[str, float]) -> float:
        """Evaluate the compiled IR on one particle.  Only FactorIR contributes to weight.

        Values are first normalised to the canonical unit of each variable's
        declared unit so that legal unit conversion (km ↔ m, hour ↔ s) is
        applied before the IR sees them.
        """

        if self.factor_ir is None:
            return 0.0
        canonical_values: dict[str, float] = {}
        for variable in self.variables:
            raw = values[variable.name]
            if not isfinite(raw):
                raise JointSchemaError(
                    f"variable {variable.name!r} produced non-finite value: {raw}"
                )
            unit_key = variable.unit.strip()
            factor = _UNIT_TABLE[unit_key][1]
            canonical_values[variable.name] = raw * factor
        return self.factor_ir.log_potential(canonical_values)

    def evaluate_axis(self, values: Mapping[str, float]) -> float:
        """Evaluate the axis-transform IR on one particle.

        Only valid when *transform* is set.  Values are normalised to the
        canonical unit before evaluation.
        """

        if self.transform is None:
            raise JointSchemaError(
                "CompiledMapping.evaluate_axis called but transform is None"
            )
        canonical_values: dict[str, float] = {}
        for variable in self.variables:
            raw = values[variable.name]
            if not isfinite(raw):
                raise JointSchemaError(
                    f"variable {variable.name!r} produced non-finite value: {raw}"
                )
            unit_key = variable.unit.strip()
            factor = _UNIT_TABLE[unit_key][1]
            canonical_values[variable.name] = raw * factor
        return self.transform.evaluate(canonical_values)


def compile_joint_schema(
    question_id: str,
    axes: Sequence[OutcomeAxis],
    variables: Sequence[VariableSpec],
    mappings: Sequence[MappingSpec],
    evidence: Sequence[Evidence] = (),
) -> tuple[CompiledMapping, ...]:
    """Validate cross-references and dimension compatibility for the whole schema."""

    if not question_id.strip():
        raise JointSchemaError("question_id is required")
    if not axes:
        raise JointSchemaError("axes must be non-empty")
    if not variables:
        raise JointSchemaError("variables must be non-empty")

    axis_index = {axis.axis_id: axis for axis in axes}
    if len(axis_index) != len(axes):
        raise JointSchemaError("OutcomeAxis.axis_id values must be unique")

    variable_index = {variable.name: variable for variable in variables}
    if len(variable_index) != len(variables):
        raise JointSchemaError("VariableSpec.name values must be unique")

    evidence_index = {atom.evidence_atom_id: atom for atom in evidence}
    if len(evidence_index) != len(evidence):
        raise JointSchemaError("Evidence.evidence_atom_id values must be unique")

    seen_mapping_ids: set[str] = set()
    compiled: list[CompiledMapping] = []
    for mapping in mappings:
        if mapping.mapping_id in seen_mapping_ids:
            raise JointSchemaError(
                f"MappingSpec.mapping_id values must be unique: {mapping.mapping_id}"
            )
        seen_mapping_ids.add(mapping.mapping_id)
        if mapping.result_axis not in axis_index:
            raise JointSchemaError(
                f"mapping {mapping.mapping_id} references unknown axis: {mapping.result_axis}"
            )
        target_axis = axis_index[mapping.result_axis]
        missing = [name for name in mapping.variables if name not in variable_index]
        if missing:
            raise JointSchemaError(
                f"mapping {mapping.mapping_id} references unknown variables: "
                + ", ".join(sorted(missing))
            )
        referenced = tuple(variable_index[name] for name in mapping.variables)
        dimensions = {variable.name: dimension_of_unit(variable.unit) for variable in referenced}
        factor_ir: FactorIR | None = None
        factor_ir: FactorIR | None = None
        transform: DeterministicTransform | None = None
        if mapping.kind is MappingKind.FORMULA:
            try:
                # First try: compile as dimensionless FactorIR.
                factor_ir = compile_factor_ir(
                    mapping.mapping_id,
                    mapping.expression or "",
                    dimensions,
                )
            except FactorIRError as factor_exc:
                msg = str(factor_exc)
                if "dimensionless" in msg.lower():
                    # The formula produced a dimensional output — try
                    # compiling as an axis-transform whose output must
                    # match the target axis dimension.
                    try:
                        dimension_of_unit(target_axis.unit)
                        # Resolve the axis unit to its dimension key.
                        axis_dim = dimension_of_unit(target_axis.unit)
                        # The axis dim may be a composite key — parse it.
                        axis_dim_map = parse_composite_dimension(axis_dim)
                        if not axis_dim_map:
                            raise JointSchemaError(
                                f"mapping {mapping.mapping_id}: target axis "
                                f"{target_axis.axis_id} is dimensionless — "
                                f"cannot compile dimensional axis-transform"
                            ) from factor_exc
                        if len(axis_dim_map) != 1:
                            raise JointSchemaError(
                                f"mapping {mapping.mapping_id}: target axis "
                                f"{target_axis.axis_id} has composite dimension "
                                f"{axis_dim_map}; axis-transform requires a "
                                f"single dimension"
                            ) from factor_exc
                        target_dim = next(iter(axis_dim_map.keys()))
                        transform = compile_axis_transform(
                            mapping.mapping_id,
                            mapping.expression or "",
                            dimensions,
                            target_axis_dimension=target_dim,
                        )
                    except FactorIRError as axis_exc:
                        raise JointSchemaError(
                            f"mapping {mapping.mapping_id}: {axis_exc}"
                        ) from axis_exc
                else:
                    raise JointSchemaError(
                        f"mapping {mapping.mapping_id}: {factor_exc}"
                    ) from factor_exc
        elif mapping.kind is MappingKind.LIKELIHOOD:
            variable = referenced[0]
            if dimension_of_unit(variable.unit) != dimension_of_unit(target_axis.unit):
                raise JointSchemaError(
                    f"mapping {mapping.mapping_id}: likelihood variable "
                    f"{variable.name} ({variable.unit}) does not share dimension "
                    f"with axis {target_axis.axis_id} ({target_axis.unit})"
                )
        for atom_id in mapping.evidence_atom_ids:
            if atom_id not in evidence_index:
                raise JointSchemaError(
                    f"mapping {mapping.mapping_id} references unknown evidence atom: {atom_id}"
                )
        for variable in referenced:
            if variable.evidence_atom_id and variable.evidence_atom_id not in evidence_index:
                raise JointSchemaError(
                    f"variable {variable.name} references unknown evidence atom: "
                    f"{variable.evidence_atom_id}"
                )
        compiled.append(
            CompiledMapping(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping=mapping,
                axis=target_axis,
                variables=referenced,
                variable_dimensions=dimensions,
                factor_ir=factor_ir,
                transform=transform,
            )
        )
    return tuple(compiled)


__all__ = [
    "DIMENSIONLESS",
    "EVIDENCE_SCHEMA_VERSION",
    "JOINT_SCHEMA_VERSION",
    "CompiledMapping",
    "DeterministicTransform",
    "DimensionMismatchError",
    "EpistemicType",
    "Evidence",
    "JointSchemaError",
    "MappingKind",
    "MappingSpec",
    "OutcomeAxis",
    "OutcomeSpace",
    "UnknownUnitError",
    "VariableSpec",
    "VariableStatus",
    "compatible_units",
    "compile_axis_transform",
    "compile_joint_schema",
    "conversion_factor",
    "dimension_of_unit",
    "FACTOR_IR_VERSION",
    "canonical_unit",
    "known_dimensions",
    "normalize_to_canonical",
    "parse_composite_dimension",
]