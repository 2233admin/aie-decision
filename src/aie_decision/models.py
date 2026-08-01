"""Versioned, JSON-compatible records for the standalone AIE ledger.

The models intentionally use only the Python standard library.  They are data
contracts, not inference implementations; deterministic settlement lives in
``validation`` and later pipeline stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any, TypeAlias

SCHEMA_VERSION = "1.0.0"


class AnswerType(StrEnum):
    CURRENT_OBSERVATION = "current_observation"
    HISTORICAL_RECONSTRUCTION = "historical_reconstruction"
    FUTURE_PREDICTION = "future_prediction"
    CAUSAL_EXPLANATION = "causal_explanation"
    DECISION_COMPARISON = "decision_comparison"


class AnswerabilityState(StrEnum):
    """Exhaustive terminal states; no implicit success state is allowed."""

    ANSWERABLE_BOUNDED = "answerable_bounded"
    NOT_ANSWERABLE = "not_answerable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_CONTRACT = "invalid_contract"
    FAILED_VALIDATION = "failed_validation"


class Necessity(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    COMPETING = "competing"
    EXPLORATORY = "exploratory"


class ConditionStatus(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    CONTRADICTED = "contradicted"
    INVALIDATED = "invalidated"


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


class EventStatus(StrEnum):
    CONFIRMED = "confirmed"
    PARTIALLY_CONFIRMED = "partially_confirmed"
    CONTESTED = "contested"
    CLAIMED_ONLY = "claimed_only"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class CoverageSemantics(StrEnum):
    EMPIRICAL_PREDICTION_INTERVAL = "empirical_prediction_interval"
    SUBJECTIVE_CREDIBLE_INTERVAL = "subjective_credible_interval"
    MEASUREMENT_ERROR_BOUND = "measurement_error_bound"
    SCENARIO_BOUND = "scenario_bound"
    UNKNOWN = "unknown"


class IntervalAuditStatus(StrEnum):
    CALIBRATED_INFORMATIVE = "calibrated_informative"
    CALIBRATED_UNINFORMATIVE = "calibrated_uninformative"
    MISCALIBRATED_OVERCONFIDENT = "miscalibrated_overconfident"
    MISCALIBRATED_UNDERCONFIDENT = "miscalibrated_underconfident"
    UNCALIBRATED_INFORMATIVE = "uncalibrated_informative"
    UNCALIBRATED_UNINFORMATIVE = "uncalibrated_uninformative"
    INVALID_TARGET_OR_HORIZON = "invalid_target_or_horizon"


class ValidationStatus(StrEnum):
    UNVALIDATED = "unvalidated"
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    source_id: str
    locator: str
    observed_at: str | None = None
    transformation: str | None = None


@dataclass(frozen=True, slots=True)
class Revision:
    revision_id: str
    sequence: int
    created_at: str
    supersedes_revision_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerTarget:
    entity: str
    measure: str
    unit: str


@dataclass(frozen=True, slots=True)
class AnswerContract:
    question_id: str
    question: str
    answer_type: AnswerType
    target: AnswerTarget
    observation_cutoff: str
    prediction_horizon: str | None = None
    requested_coverage: float | None = None
    uncertainty_semantics: CoverageSemantics | None = None
    decision_thresholds: tuple[float, ...] = ()
    acceptable_width: dict[str, Any] | None = None
    geography: str | None = None
    decision_use: str | None = None
    revision: Revision | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ConditionNode:
    condition_id: str
    name: str
    value_type: str
    necessity: Necessity
    status: ConditionStatus
    answer_impact: str
    unit: str | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None


@dataclass(frozen=True, slots=True)
class ConditionEdge:
    edge_id: str
    from_id: str
    to_id: str
    relation: str
    evidence_status: str
    answer_impact: str
    direction: str | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None


@dataclass(frozen=True, slots=True)
class ConditionGraph:
    graph_id: str
    question_id: str
    answer_contract_revision_id: str
    conditions: tuple[ConditionNode, ...]
    edges: tuple[ConditionEdge, ...]
    minimal_sufficient_sets: tuple[tuple[str, ...], ...] = ()
    revision: Revision | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    schema_version: str = SCHEMA_VERSION


RequiredConditionGraph = ConditionGraph


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    title: str
    retrieved_at: str
    content_hash: str
    uri: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    speaker: str | None = None
    # Admission metadata belongs to the source record rather than to inferred
    # propositions.  An explicitly irrelevant source remains auditable without
    # being promoted into the answer-directed evidence layer.
    target_relevance: tuple[str, ...] = ()
    evidence_disposition: str = "unassessed"
    exclusion_reason: str | None = None
    transformation_lineage: tuple[str, ...] = ()
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class EvidenceProposition:
    evidence_atom_id: str
    source_id: str
    source_locator: str
    claim: str
    epistemic_type: EpistemicType
    independence_group: str
    target_relevance: tuple[str, ...]
    speaker: str | None = None
    event_time: str | None = None
    published_at: str | None = None
    modality: str | None = None
    source_position: str | None = None
    extraction_confidence: float | None = None
    truth_confidence: float | None = None
    # Deprecated compatibility field for early v1 fixtures.  New compilation
    # must keep extraction confidence separate from proposition truth confidence.
    uncertainty: float | None = None
    transformation: str | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


EvidenceAtom = EvidenceProposition


@dataclass(frozen=True, slots=True)
class SceneEvent:
    event_id: str
    status: EventStatus
    actor_ids: tuple[str, ...]
    action: str
    supporting_atom_ids: tuple[str, ...]
    counter_atom_ids: tuple[str, ...] = ()
    time_window: str | None = None
    unknown_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EventScene:
    scene_id: str
    question_id: str
    actors: tuple[dict[str, Any], ...]
    events: tuple[SceneEvent, ...]
    relations: tuple[dict[str, Any], ...] = ()
    known_unknowns: tuple[str, ...] = ()
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


ReconstructedScene = EventScene


@dataclass(frozen=True, slots=True)
class MissingCondition:
    estimate_id: str
    condition_id: str
    estimate_type: str
    lower: float
    upper: float
    unit: str
    coverage_semantics: CoverageSemantics
    coverage: float
    method: str
    input_atom_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    bound_provenance: tuple[ProvenanceRef, ...]
    dependence_case: str
    calibration_profile_id: str | None = None
    valid_until: str | None = None
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


ConditionEstimate = MissingCondition


@dataclass(frozen=True, slots=True)
class BayesianUpdateRecord:
    update_id: str
    condition_id: str
    prior_estimate_id: str
    new_evidence_atom_ids: tuple[str, ...]
    evidence_direction: str
    evidence_strength: str
    posterior: dict[str, Any]
    method: str
    assumptions_changed: tuple[str, ...]
    created_at: str
    provenance: tuple[ProvenanceRef, ...]
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class DerivedFactor:
    factor_id: str
    name: str
    status: str
    input_condition_ids: tuple[str, ...]
    composition: dict[str, Any]
    unit: str
    time_window: str
    target_paths: tuple[str, ...]
    hypothesis: str
    falsification_conditions: tuple[str, ...]
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


DerivedFactorCandidate = DerivedFactor


@dataclass(frozen=True, slots=True)
class IntervalAudit:
    evaluation_id: str
    question_id: str
    forecast: dict[str, Any]
    reference_value: float
    normalized_width: float
    empirical_coverage: float | None
    baseline_interval: dict[str, float]
    information_gain: float
    status: IntervalAuditStatus
    uncertainty_contributions: tuple[dict[str, Any], ...] = ()
    next_information_actions: tuple[str, ...] = ()
    interval_score: float | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    revision: Revision | None = None
    schema_version: str = SCHEMA_VERSION


ForecastIntervalEvaluation = IntervalAudit


@dataclass(frozen=True, slots=True)
class Answerability:
    status: AnswerabilityState
    reasons: tuple[str, ...]
    blocking_condition_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisPackage:
    package_id: str
    run_id: str
    package_state: str
    answer_contract: AnswerContract
    condition_graph: ConditionGraph
    sources: tuple[SourceRecord, ...]
    evidence_propositions: tuple[EvidenceProposition, ...]
    event_scene: EventScene | None
    missing_conditions: tuple[MissingCondition, ...]
    derived_factors: tuple[DerivedFactor, ...]
    calculations: tuple[dict[str, Any], ...]
    interval_audit: IntervalAudit | None
    conclusion: dict[str, Any] | None
    answerability: Answerability
    empty_section_reasons: dict[str, str] = field(default_factory=dict)
    revision: Revision | None = None
    provenance: tuple[ProvenanceRef, ...] = ()
    schema_version: str = SCHEMA_VERSION


CompleteAnalysisPackage: TypeAlias = AnalysisPackage
PartialAnalysisPackage: TypeAlias = AnalysisPackage


def to_dict(value: Any) -> Any:
    """Convert nested records/enums/tuples into a JSON-compatible value."""

    if isinstance(value, StrEnum):
        return str(value)
    if is_dataclass(value):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_dict(item) for item in value]
    return value
