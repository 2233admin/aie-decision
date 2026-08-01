"""Answer-oriented contract and condition-graph operations.

This module is deliberately deterministic.  A model may suggest candidate
conditions, but only these functions may admit them to the settled graph.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    AnswerContract,
    AnswerTarget,
    AnswerType,
    Answerability,
    AnswerabilityState,
    ConditionEdge,
    ConditionGraph,
    ConditionNode,
    ConditionStatus,
    CoverageSemantics,
    Necessity,
    ProvenanceRef,
    Revision,
)


class AnswerContractError(ValueError):
    """Raised when an answer contract is materially ambiguous or invalid."""


class ConditionGraphError(ValueError):
    """Raised when candidate conditions cannot form a traceable graph."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _new_revision(prefix: str, identity: str, previous: Revision | None = None) -> Revision:
    sequence = 1 if previous is None else previous.sequence + 1
    revision_id = _stable_id(prefix, identity, sequence, _now())
    return Revision(
        revision_id=revision_id,
        sequence=sequence,
        created_at=_now(),
        supersedes_revision_id=None if previous is None else previous.revision_id,
    )


def create_answer_contract(
    *,
    question_id: str,
    question: str,
    answer_type: AnswerType,
    subject: str,
    target: str,
    unit: str,
    observation_cutoff: str,
    prediction_horizon: str | None = None,
    geography: str | None = None,
    decision_use: str | None = None,
    decision_thresholds: Sequence[float] = (),
    uncertainty_semantics: CoverageSemantics | None = None,
    requested_coverage: float | None = None,
    acceptable_width: Mapping[str, Any] | None = None,
    provenance: Sequence[ProvenanceRef] = (),
) -> AnswerContract:
    """Create and validate the root contract before evidence collection."""

    contract = AnswerContract(
        question_id=question_id.strip(),
        question=question.strip(),
        answer_type=answer_type,
        target=AnswerTarget(entity=subject.strip(), measure=target.strip(), unit=unit.strip()),
        observation_cutoff=observation_cutoff.strip(),
        prediction_horizon=prediction_horizon.strip() if prediction_horizon else None,
        requested_coverage=requested_coverage,
        decision_thresholds=tuple(float(value) for value in decision_thresholds),
        acceptable_width=dict(acceptable_width) if acceptable_width is not None else None,
        geography=geography.strip() if geography else None,
        decision_use=decision_use.strip() if decision_use else None,
        uncertainty_semantics=uncertainty_semantics,
        revision=_new_revision("contract_rev", question_id),
        provenance=tuple(provenance),
    )
    errors = validate_answer_contract(contract)
    if errors:
        raise AnswerContractError("; ".join(errors))
    return contract


def validate_answer_contract(contract: AnswerContract) -> tuple[str, ...]:
    """Return all contract errors without silently filling unresolved fields."""

    errors: list[str] = []
    required_text = {
        "question_id": contract.question_id,
        "question": contract.question,
        "subject": contract.target.entity,
        "target": contract.target.measure,
        "unit": contract.target.unit,
        "observation_cutoff": contract.observation_cutoff,
    }
    errors.extend(f"{name} is required" for name, value in required_text.items() if not value.strip())
    if contract.answer_type is AnswerType.FUTURE_PREDICTION:
        if not contract.prediction_horizon:
            errors.append("prediction_horizon is required for a future prediction")
        if contract.uncertainty_semantics is None or contract.uncertainty_semantics is CoverageSemantics.UNKNOWN:
            errors.append("uncertainty_semantics must be explicit for a future prediction")
    if contract.requested_coverage is not None and not 0 < contract.requested_coverage < 1:
        errors.append("requested_coverage must be strictly between 0 and 1")
    if any(not isfinite(value) for value in contract.decision_thresholds):
        errors.append("decision_thresholds must contain only finite values")
    if tuple(sorted(set(contract.decision_thresholds))) != contract.decision_thresholds:
        errors.append("decision_thresholds must be unique and increasing")
    if contract.acceptable_width is not None and not contract.acceptable_width:
        errors.append("acceptable_width must be non-empty when supplied")
    if contract.revision is None:
        errors.append("revision is required")
    return tuple(errors)


_MATERIAL_CONTRACT_FIELDS = (
    "answer_type",
    "target",
    "observation_cutoff",
    "prediction_horizon",
    "requested_coverage",
    "decision_thresholds",
    "acceptable_width",
    "geography",
    "decision_use",
    "uncertainty_semantics",
)


def revise_answer_contract(contract: AnswerContract, **changes: Any) -> AnswerContract:
    """Create a lineage-preserving contract revision and validate it."""

    if "question_id" in changes and changes["question_id"] != contract.question_id:
        raise AnswerContractError("question_id is stable across contract revisions")
    revised = replace(
        contract,
        **changes,
        revision=_new_revision("contract_rev", contract.question_id, contract.revision),
    )
    errors = validate_answer_contract(revised)
    if errors:
        raise AnswerContractError("; ".join(errors))
    return revised


def contract_changed(old: AnswerContract, new: AnswerContract) -> bool:
    """Whether a revision changes answer meaning and invalidates descendants."""

    return any(getattr(old, field) != getattr(new, field) for field in _MATERIAL_CONTRACT_FIELDS)


def build_condition_graph(
    contract: AnswerContract,
    conditions: Iterable[ConditionNode],
    edges: Iterable[ConditionEdge] = (),
    minimal_sufficient_sets: Iterable[Iterable[str]] = (),
    *,
    graph_id: str | None = None,
    provenance: Sequence[ProvenanceRef] = (),
) -> ConditionGraph:
    """Settle a directed graph, rejecting nodes without an answer path."""

    contract_errors = validate_answer_contract(contract)
    if contract_errors:
        raise ConditionGraphError("invalid answer contract: " + "; ".join(contract_errors))
    nodes = tuple(conditions)
    graph_edges = tuple(edges)
    if not nodes:
        raise ConditionGraphError("at least one condition is required")
    node_ids = [node.condition_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ConditionGraphError("condition identifiers must be unique")
    missing_paths = [node.condition_id for node in nodes if not node.answer_impact.strip()]
    if missing_paths:
        raise ConditionGraphError(
            "conditions lack a stated path to the target answer: " + ", ".join(missing_paths)
        )
    known = set(node_ids)
    for edge in graph_edges:
        if edge.from_id not in known or edge.to_id not in known:
            raise ConditionGraphError(f"edge {edge.edge_id} references an unknown condition")
        if not edge.answer_impact.strip():
            raise ConditionGraphError(f"edge {edge.edge_id} lacks stated answer impact")
    sets = tuple(tuple(group) for group in minimal_sufficient_sets)
    if not sets:
        required = tuple(node.condition_id for node in nodes if node.necessity is Necessity.REQUIRED)
        sets = (required,) if required else ()
    for group in sets:
        if not group:
            raise ConditionGraphError("minimal sufficient sets cannot be empty")
        unknown = set(group) - known
        if unknown:
            raise ConditionGraphError("minimal sufficient set references unknown conditions: " + ", ".join(sorted(unknown)))
    identity = graph_id or _stable_id("graph", contract.question_id, contract.revision.revision_id)
    return ConditionGraph(
        graph_id=identity,
        question_id=contract.question_id,
        answer_contract_revision_id=contract.revision.revision_id,
        conditions=nodes,
        edges=graph_edges,
        minimal_sufficient_sets=sets,
        revision=_new_revision("graph_rev", identity),
        provenance=tuple(provenance),
    )


def minimal_sufficient_sets(graph: ConditionGraph) -> tuple[tuple[str, ...], ...]:
    """Return the explicit minimal sets after removing strict supersets."""

    unique = {frozenset(group) for group in graph.minimal_sufficient_sets if group}
    minimal = [group for group in unique if not any(other < group for other in unique)]
    return tuple(tuple(sorted(group)) for group in sorted(minimal, key=lambda item: (len(item), sorted(item))))


def evaluate_answerability(contract: AnswerContract, graph: ConditionGraph) -> Answerability:
    """Evaluate a graph using explicit terminal states and blocking nodes."""

    errors = validate_answer_contract(contract)
    if errors:
        return Answerability(AnswerabilityState.INVALID_CONTRACT, errors)
    if contract.revision is None or graph.answer_contract_revision_id != contract.revision.revision_id:
        return Answerability(
            AnswerabilityState.INVALID_CONTRACT,
            ("condition graph was built for a different answer-contract revision",),
        )
    nodes = {node.condition_id: node for node in graph.conditions}
    sets = minimal_sufficient_sets(graph)
    if not sets:
        return Answerability(
            AnswerabilityState.NOT_ANSWERABLE,
            ("no minimal sufficient condition set is declared",),
        )
    resolved = {ConditionStatus.OBSERVED, ConditionStatus.ESTIMATED}
    for group in sets:
        if all(nodes[item].status in resolved for item in group):
            return Answerability(
                AnswerabilityState.ANSWERABLE_BOUNDED,
                ("a minimal sufficient condition set is resolved",),
            )
    blocking = sorted(
        {
            item
            for group in sets
            for item in group
            if nodes[item].status not in resolved
        }
    )
    contradicted = [item for item in blocking if nodes[item].status is ConditionStatus.CONTRADICTED]
    if contradicted:
        return Answerability(
            AnswerabilityState.NOT_ANSWERABLE,
            ("required evidence is contradicted",),
            tuple(contradicted),
        )
    return Answerability(
        AnswerabilityState.INSUFFICIENT_EVIDENCE,
        ("every minimal sufficient set contains unresolved conditions",),
        tuple(blocking),
    )


def invalidate_graph_for_contract_change(
    graph: ConditionGraph,
    old_contract: AnswerContract,
    new_contract: AnswerContract,
) -> ConditionGraph:
    """Invalidate descendants when answer semantics change; preserve lineage."""

    if old_contract.question_id != new_contract.question_id or graph.question_id != old_contract.question_id:
        raise ConditionGraphError("graph and contract question identifiers do not match")
    if not contract_changed(old_contract, new_contract):
        return graph
    invalidated_nodes = tuple(
        replace(
            node,
            status=ConditionStatus.INVALIDATED,
            revision=_new_revision("condition_rev", node.condition_id, node.revision),
        )
        for node in graph.conditions
    )
    return replace(
        graph,
        answer_contract_revision_id=new_contract.revision.revision_id,
        conditions=invalidated_nodes,
        revision=_new_revision("graph_rev", graph.graph_id, graph.revision),
    )
