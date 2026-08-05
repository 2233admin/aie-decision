"""Shared enums and errors for Fermi decomposition contracts."""

from __future__ import annotations

from enum import StrEnum


SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QuestionStatus(StrEnum):
    """Lifecycle states of a question root."""

    OPEN = "open"
    EXPANDED = "expanded"
    ATOMIC_LEAF = "atomic_leaf"
    BLOCKED = "blocked"
    ANSWERED = "answered"


class NodeStatus(StrEnum):
    """Lifecycle states of a single node inside a decomposition tree."""

    OPEN = "open"
    EXPANDED = "expanded"
    ATOMIC_LEAF = "atomic_leaf"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"
    PRUNED = "pruned"


class NodeRole(StrEnum):
    """Why a node exists in the tree."""

    TARGET = "target"
    CHILD = "child"
    ALTERNATIVE = "alternative"
    ATOM_CANDIDATE = "atom_candidate"


class ObservationKind(StrEnum):
    """How a quantity in an atomic claim is known to the analyst."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class MeasurementKind(StrEnum):
    """The structured operation used to obtain an atomic quantity."""

    DIRECT_OBSERVATION = "direct_observation"
    RECORD_LOOKUP = "record_lookup"
    COUNT = "count"
    TIMED_MEASUREMENT = "timed_measurement"
    INSTRUMENT_MEASUREMENT = "instrument_measurement"
    DERIVED_PROXY = "derived_proxy"


class GapKind(StrEnum):
    """Why a frontier node or root cannot be promoted yet."""

    INCOMPLETE_ROOT = "incomplete_root"
    MISSING_RELATIONSHIP = "missing_relationship"
    UNRESOLVED_NODE = "unresolved_node"
    ATOM_REJECTED = "atom_rejected"
    UNKNOWN_LEAF = "unknown_leaf"
    UNIT_MISMATCH = "unit_mismatch"
    INSUFFICIENT_DEPENDENCIES = "insufficient_dependencies"
    REDUNDANT_ALTERNATIVE = "redundant_alternative"


class ActionKind(StrEnum):
    """Exhaustive list of structural mutations exposed by the runtime."""

    CREATE_QUESTION = "create_question"
    REGISTER_NODE = "register_node"
    EXPAND = "expand"
    PROPOSE_ATOM = "propose_atom"
    PROPOSE_ALTERNATIVE = "propose_alternative"
    PRUNE = "prune"
    ACTIVATE_BRANCH = "activate_branch"
    REGISTER_GAP = "register_gap"
    EXPORT = "export"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FermiContractError(ValueError):
    """Raised when a structural contract cannot be admitted to the tree."""


class RestrictedExpressionError(FermiContractError):
    """Raised when an expression leaves the allowed arithmetic vocabulary."""


class DimensionalError(FermiContractError):
    """Raised when a relationship cannot produce its declared parent unit."""


class AtomicClaimError(FermiContractError):
    """Raised when an atomic claim remains abstract instead of measurable."""


class RedundancyReason(str):
    """Human-readable reason why an alternative was declared redundant."""

    __slots__ = ()


__all__ = [
    "QuestionStatus",
    "NodeStatus",
    "NodeRole",
    "ObservationKind",
    "MeasurementKind",
    "GapKind",
    "ActionKind",
    "FermiContractError",
    "RestrictedExpressionError",
    "DimensionalError",
    "AtomicClaimError",
    "RedundancyReason",
    "SCHEMA_VERSION"
]
