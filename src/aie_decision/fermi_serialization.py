"""Immutable Fermi records and their fail-closed serialization checks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .fermi_contract_core import (
    ActionKind,
    AtomicClaimError,
    FermiContractError,
    GapKind,
    MeasurementKind,
    NodeRole,
    NodeStatus,
    ObservationKind,
    QuestionStatus,
    RedundancyReason,
    SCHEMA_VERSION,
)
from .fermi_expressions import RestrictedExpression, expressions_are_equivalent, parse_restricted_expression
from .fermi_units import DEFAULT_UNIT_SYMBOLS, parse_compound_unit

@dataclass(frozen=True, slots=True)
class Scope:
    """The universe over which a quantity should be summed, averaged, etc."""

    population: str | None = None
    geography: str | None = None
    time_window: str | None = None
    temporal_basis: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def is_well_defined(self) -> bool:
        """Return True when at least one of population/geography anchors the scope."""

        return bool((self.population and self.population.strip()) or (self.geography and self.geography.strip()))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "population": self.population,
            "geography": self.geography,
            "time_window": self.time_window,
            "temporal_basis": self.temporal_basis,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True, slots=True)
class Question:
    """A raw quantitative question that begins a decomposition."""

    question_id: str
    question: str
    target_subject: str
    target_measure: str
    unit: str
    time_basis: str
    scope: Scope
    status: QuestionStatus = QuestionStatus.OPEN
    decision_use: str | None = None
    acceptable_width: str | None = None
    unresolved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise FermiContractError("question_id is required")
        if not self.question.strip():
            raise FermiContractError("question text is required")
        if not self.target_subject.strip():
            raise FermiContractError("target_subject is required")
        if not self.target_measure.strip():
            raise FermiContractError("target_measure is required")
        if self.unit is None or not str(self.unit).strip():
            raise FermiContractError("unit is required")
        if not self.time_basis.strip():
            raise FermiContractError("time_basis is required")
        if not isinstance(self.scope, Scope):
            raise FermiContractError("scope must be a Scope instance")

    @property
    def target_unit(self) -> CompoundUnit:
        return parse_compound_unit(self.unit)

    def is_minimally_complete(self) -> bool:
        """A raw question is minimally complete when scope anchors reality and no field is unresolved."""

        if self.unresolved_fields:
            return False
        return self.scope.is_well_defined()

    def with_unresolved(self, fields: Iterable[str]) -> "Question":
        new_fields = tuple(sorted(set(self.unresolved_fields) | set(fields)))
        return replace(self, unresolved_fields=new_fields)


@dataclass(frozen=True, slots=True)
class Node:
    """A single node in the decomposition graph."""

    node_id: str
    label: str
    role: NodeRole
    status: NodeStatus = NodeStatus.OPEN
    parent_id: str | None = None
    unit: str | None = None
    scope: Scope | None = None
    description: str = ""
    mechanism: str = ""
    expansion_id: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise FermiContractError("node_id is required")
        if not self.label.strip():
            raise FermiContractError("label is required for node " + self.node_id)
        if self.role is NodeRole.TARGET and self.parent_id is not None:
            raise FermiContractError("the target node cannot have a parent_id")
        if self.unit is not None:
            parse_compound_unit(self.unit)


# ---------------------------------------------------------------------------
# Relationship, expansion, branch, gap, action
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Relationship:
    """How a set of children re-derives a parent node's quantity."""

    relationship_id: str
    parent_node_id: str
    parent_unit: str
    expression: str
    child_node_ids: tuple[str, ...]
    child_units: tuple[str, ...]
    rationale: str
    expression_ast: RestrictedExpression | None = None

    def __post_init__(self) -> None:
        if not self.relationship_id.strip():
            raise FermiContractError("relationship_id is required")
        if not self.parent_node_id.strip():
            raise FermiContractError("parent_node_id is required")
        if not self.parent_unit.strip():
            raise FermiContractError("parent_unit is required")
        if not self.expression.strip():
            raise FermiContractError("expression is required")
        if not self.child_node_ids:
            raise FermiContractError("relationship must declare at least one child node")
        if len(self.child_node_ids) != len(self.child_units):
            raise FermiContractError("child_node_ids and child_units must be parallel tuples")
        if not self.rationale.strip():
            raise FermiContractError("rationale is required to explain the split")


@dataclass(frozen=True, slots=True)
class Expansion:
    """An applied relationship that introduces a set of children to the tree."""

    expansion_id: str
    target_node_id: str
    relationship_id: str
    parent_unit: str
    projected_unit: str
    child_node_ids: tuple[str, ...]
    rationale: str
    is_alternative: bool = False
    alternative_of_expansion_id: str | None = None
    is_redundant: bool = False
    redundancy_reason: str | None = None

    def describe(self) -> str:
        if self.is_alternative:
            base = "alternative"
            if self.is_redundant:
                base = "redundant alternative"
            return (
                f"{base} expansion {self.expansion_id} of node {self.target_node_id} "
                f"via relationship {self.relationship_id}"
            )
        return f"expansion {self.expansion_id} of node {self.target_node_id}"


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    """An operational measurement attached to a leaf node.

    Operational atomicity is described through *structured* fields, not
    through lexical detection against any particular natural language.  The
    required fields are:

    * :attr:`target_object` — the concrete real-world object, event, or
      population being measured.
    * :attr:`unit` — the compound unit of the resulting quantity.
    * :attr:`scope` — the population, geography, and time basis that pin
      the quantity to a universe.
    * :attr:`measurement_kind` — one of the explicit :class:`MeasurementKind`
      operations (count, lookup, observation, etc.).
    * :attr:`source` — the data source, instrument, or registry that
      produces the measurement (e.g. "ACS B08006", "smart-meter log",
      "field tally sheet").
    * :attr:`procedure` — a non-empty structured description of the
      procedural steps that produce the quantity.

    No string in this record is matched against an English or
    domain-specific vocabulary list.  Acceptance depends on field
    presence, enum membership, and the caller's ability to articulate the
    measurement honestly; evidence gaps remain visible through
    :attr:`observation_kind` and :attr:`assumption_notes`.
    """

    node_id: str
    target_object: str
    unit: str
    scope: Scope
    measurement_kind: MeasurementKind
    source: str
    procedure: str
    time_basis: str = ""
    observation_kind: ObservationKind = ObservationKind.UNKNOWN
    assumption_notes: str = ""

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise AtomicClaimError("node_id is required")
        if not isinstance(self.scope, Scope):
            raise AtomicClaimError("scope must be a Scope instance")
        if not isinstance(self.measurement_kind, MeasurementKind):
            raise AtomicClaimError("measurement_kind must be a MeasurementKind value")
        if not isinstance(self.observation_kind, ObservationKind):
            raise AtomicClaimError("observation_kind must be an ObservationKind value")
        parse_compound_unit(self.unit)

    def has_structured_measurement(self) -> bool:
        """Return True when every required structured field is populated.

        The check is purely structural.  It does not attempt to interpret
        the meaning of any field, nor does it look for keywords in any
        human language.
        """

        return bool(
            self.target_object.strip()
            and str(self.unit).strip()
            and self.scope.is_well_defined()
            and self.source.strip()
            and self.procedure.strip()
        )


# Each ``MeasurementKind`` declares the structured fields that must be
# populated for the claim to be admitted.  The check is purely structural:
# it never inspects the textual content of a field for keywords.
_MEASUREMENT_KIND_REQUIRED_FIELDS: dict[MeasurementKind, tuple[str, ...]] = {
    MeasurementKind.DIRECT_OBSERVATION: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.RECORD_LOOKUP: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.COUNT: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.TIMED_MEASUREMENT: ("target_object", "unit", "scope", "source", "procedure", "time_basis"),
    MeasurementKind.INSTRUMENT_MEASUREMENT: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.DERIVED_PROXY: (
        "target_object",
        "unit",
        "scope",
        "source",
        "procedure",
        "assumption_notes",
    ),
}


def validate_atomic_claim(
    claim: AtomicClaim,
    *,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> tuple[str, ...]:
    """Return an empty tuple when ``claim`` is structurally measurable.

    Acceptance is determined by the :class:`MeasurementKind` enum and the
    presence of every required structured field.  The function never
    inspects the textual content of any field for English or
    domain-specific keywords; a Chinese, Arabic, or any other language
    description is admitted on the same structural basis as English.

    Honest evidence and assumption gaps are *not* treated as fatal: a
    ``DERIVED_PROXY`` claim may carry populated ``assumption_notes`` to
    document its dependence on an external mapping, and an
    :attr:`observation_kind` of ``UNKNOWN`` is permitted.  The runtime
    surfaces those facts through other channels (gaps, exports) rather
    than here.
    """

    errors: list[str] = []
    if not isinstance(claim, AtomicClaim):
        errors.append("claim must be an AtomicClaim instance")
        return tuple(errors)

    required = _MEASUREMENT_KIND_REQUIRED_FIELDS.get(claim.measurement_kind, ())
    if not claim.target_object.strip():
        errors.append("target_object is required")
    if not claim.unit or not str(claim.unit).strip():
        errors.append("unit is required")
    else:
        try:
            parse_compound_unit(claim.unit, registry=registry)
        except RestrictedExpressionError as exc:
            errors.append(f"unit is not a valid compound expression: {exc}")
    if not claim.scope.is_well_defined():
        errors.append("scope must declare either a population or a geography anchor")
    if not claim.source.strip():
        errors.append("source is required for measurement_kind=" + str(claim.measurement_kind))
    if not claim.procedure.strip():
        errors.append("procedure is required for measurement_kind=" + str(claim.measurement_kind))
    if "time_basis" in required and not claim.time_basis.strip():
        errors.append(
            f"time_basis is required for measurement_kind={claim.measurement_kind}"
        )
    if (
        claim.measurement_kind is MeasurementKind.DERIVED_PROXY
        and not claim.assumption_notes.strip()
    ):
        errors.append(
            "assumption_notes are required for measurement_kind=derived_proxy"
        )
    if "time_basis" not in required and not claim.time_basis.strip():
        # Optional for non-temporal measurement kinds, but record its absence
        # when the field is later required by downstream code.
        pass
    return tuple(errors)


@dataclass(frozen=True, slots=True)
class Branch:
    """A named lineage of expansion choices inside a decomposition tree."""

    branch_id: str
    root_question_id: str
    expansion_ids: tuple[str, ...]
    divergent_at_expansion_id: str | None = None
    note: str = ""

    def covers(self, expansion_id: str) -> bool:
        return expansion_id in self.expansion_ids


@dataclass(frozen=True, slots=True)
class Gap:
    """An unresolved obstacle that must remain visible to the AI."""

    gap_id: str
    kind: GapKind
    target: str
    explanation: str
    blocking: bool = True
    introduced_by_action_id: str | None = None

    def __post_init__(self) -> None:
        if not self.gap_id.strip():
            raise FermiContractError("gap_id is required")
        if not self.target.strip():
            raise FermiContractError("gap target is required")
        if not self.explanation.strip():
            raise FermiContractError("gap explanation is required")


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Append-only record of a single structural mutation."""

    action_id: str
    kind: ActionKind
    payload: Mapping[str, Any]
    result_summary: str
    accepted: bool
    recorded_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": str(self.kind),
            "payload": _stringify_mapping(self.payload),
            "result_summary": self.result_summary,
            "accepted": self.accepted,
            "recorded_at": self.recorded_at,
            "error": self.error,
        }


def _stringify_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stringify_mapping(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stringify_mapping(item) for item in value]
    if hasattr(value, "to_dict"):
        return _stringify_mapping(value.to_dict())
    if hasattr(value, "to_canonical"):
        return _stringify_mapping(value.to_canonical())
    if isinstance(value, StrEnum):
        return str(value)
    return value


__all__ = [
    "Scope",
    "Question",
    "Node",
    "Relationship",
    "Expansion",
    "AtomicClaim",
    "Branch",
    "Gap",
    "ActionRecord",
    "validate_atomic_claim"
]
