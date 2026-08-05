"""Track A decomposition-tree runtime.

This module turns the contracts in :mod:`aie_decision.fermi_contracts`
into an immutable, copy-on-write, append-only editing surface that an
external AI can drive from a raw question to a recursively expanded
tree.  It is responsible for:

* Registering questions, target nodes, child nodes, and the relationships
  that bind parents to children.
* Recording alternatives at any node without imposing a fixed candidate
  count, and flagging algebraically redundant alternatives so the search
  does not double-count them.
* Tracking the current branch and producing a projection of just that
  branch's nodes and expansions.
* Returning the active frontier: open nodes that the AI could still
  expand, propose alternatives around, or measure.
* Producing pruning and current-branch projections that excise pruned
  subtrees without rewriting history.
* Surfacing unresolved gaps instead of fabricating values, and exposing
  the entire structural state through a single ``export`` call.

The module never performs probability propagation, scheduling, prompt
construction, or model calls.  Track B and Track C consume the operations
exposed here verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

from .fermi_contracts import (
    ActionKind,
    ActionRecord,
    AtomicClaim,
    Branch,
    CompoundUnit,
    DimensionalError,
    Expansion,
    FermiContractError,
    Gap,
    GapKind,
    Node,
    NodeRole,
    NodeStatus,
    ObservationKind,
    Question,
    QuestionStatus,
    RedundancyReason,
    Relationship,
    RestrictedExpression,
    Scope,
    evaluate_restricted_expression,
    expressions_are_equivalent,
    parse_compound_unit,
    parse_restricted_expression,
    project_dimensional_closure,
    units_close,
    validate_atomic_claim,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DecompositionError(FermiContractError):
    """Raised when a structural mutation cannot be admitted to the tree."""


# ---------------------------------------------------------------------------
# State record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecompositionState:
    """Immutable decomposition-tree state.

    Every mutation returns a new :class:`DecompositionState` with copied
    tuples and updated sequence counters.  The persisted record therefore
    never rewrites history; trajectory recording (Track C) replays from
    :attr:`actions`.
    """

    question: Question
    nodes: tuple[Node, ...]
    relationships: tuple[Relationship, ...]
    expansions: tuple[Expansion, ...]
    branches: tuple[Branch, ...]
    gaps: tuple[Gap, ...]
    actions: tuple[ActionRecord, ...]
    current_branch_id: str | None
    pruned_node_ids: frozenset[str] = field(default_factory=frozenset)
    node_sequence: int = 0
    relationship_sequence: int = 0
    expansion_sequence: int = 0
    branch_sequence: int = 0
    gap_sequence: int = 0
    action_sequence: int = 0

    # -- Counters -----------------------------------------------------------

    def _next_node_id(self) -> str:
        return f"n_{self.node_sequence + 1:04d}"

    def _next_relationship_id(self) -> str:
        return f"rel_{self.relationship_sequence + 1:04d}"

    def _next_expansion_id(self) -> str:
        return f"exp_{self.expansion_sequence + 1:04d}"

    def _next_branch_id(self) -> str:
        return f"br_{self.branch_sequence + 1:04d}"

    def _next_gap_id(self) -> str:
        return f"gap_{self.gap_sequence + 1:04d}"

    def _next_action_id(self) -> str:
        return f"act_{self.action_sequence + 1:04d}"

    # -- Views (read-only helpers) -----------------------------------------

    def node(self, node_id: str) -> Node:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise DecompositionError(f"unknown node_id: {node_id}")

    def relationship(self, relationship_id: str) -> Relationship:
        for item in self.relationships:
            if item.relationship_id == relationship_id:
                return item
        raise DecompositionError(f"unknown relationship_id: {relationship_id}")

    def expansion(self, expansion_id: str) -> Expansion:
        for item in self.expansions:
            if item.expansion_id == expansion_id:
                return item
        raise DecompositionError(f"unknown expansion_id: {expansion_id}")

    def branch(self, branch_id: str) -> Branch:
        for item in self.branches:
            if item.branch_id == branch_id:
                return item
        raise DecompositionError(f"unknown branch_id: {branch_id}")

    def expansions_at(self, node_id: str) -> tuple[Expansion, ...]:
        return tuple(item for item in self.expansions if item.target_node_id == node_id)

    def children_of(self, node_id: str) -> tuple[Node, ...]:
        expansions = self.expansions_at(node_id)
        if not expansions:
            return ()
        child_ids: tuple[str, ...] = expansions[0].child_node_ids
        return tuple(self.node(cid) for cid in child_ids)

    # -- Branch & alternative derivation -----------------------------------

    def current_branch(self) -> Branch:
        if not self.branches:
            raise DecompositionError("no branches exist; call create_decomposition first")
        if self.current_branch_id is None:
            return self.branches[0]
        try:
            return self.branch(self.current_branch_id)
        except DecompositionError:
            return self.branches[0]

    def alternative_branches(self) -> tuple[Branch, ...]:
        current_id = self.current_branch().branch_id
        return tuple(item for item in self.branches if item.branch_id != current_id)

    def expansion_to_branch(self, expansion_id: str) -> Branch:
        for branch in self.branches:
            if branch.covers(expansion_id):
                return branch
        raise DecompositionError(f"no branch covers expansion {expansion_id}")

    def frontier(self) -> tuple[Node, ...]:
        """Return the open frontier nodes on the current branch."""

        branch = self.current_branch()
        branch_expansion_ids = set(branch.expansion_ids)
        reachable = _reachable_node_ids(self, branch_expansion_ids)
        frontier: list[Node] = []
        for node in self.nodes:
            if node.node_id not in reachable:
                continue
            if node.node_id in self.pruned_node_ids:
                continue
            if node.status in (
                NodeStatus.ATOMIC_LEAF,
                NodeStatus.UNRESOLVED,
                NodeStatus.REJECTED,
                NodeStatus.PRUNED,
            ):
                continue
            expansions_here = [
                expansion
                for expansion in self.expansions
                if expansion.target_node_id == node.node_id
                and expansion.expansion_id in branch_expansion_ids
            ]
            if expansions_here and node.status is NodeStatus.EXPANDED:
                continue
            if node.role is NodeRole.TARGET and not node.parent_id:
                # A root with unresolved question fields stays on the frontier.
                frontier.append(node)
                continue
            frontier.append(node)
        return tuple(frontier)

    def is_redundant_alternative(self, expansion_id: str) -> tuple[bool, RedundancyReason | None]:
        expansion = self.expansion(expansion_id)
        if not expansion.is_alternative:
            return False, None
        if expansion.is_redundant:
            return True, RedundancyReason(str(expansion.redundancy_reason))
        return False, None

    def export(self) -> dict[str, Any]:
        """Return a JSON-compatible projection of the structural state."""

        current = self.current_branch()
        branch_payload: list[dict[str, Any]] = []
        for branch in self.branches:
            branch_payload.append(
                {
                    "branch_id": branch.branch_id,
                    "root_question_id": branch.root_question_id,
                    "expansion_ids": list(branch.expansion_ids),
                    "divergent_at_expansion_id": branch.divergent_at_expansion_id,
                    "note": branch.note,
                    "is_current": branch.branch_id == current.branch_id,
                }
            )
        frontier_payload = [
            {
                "node_id": node.node_id,
                "label": node.label,
                "unit": node.unit,
                "scope": node.scope.to_dict() if node.scope else None,
                "status": str(node.status),
                "role": str(node.role),
                "parent_id": node.parent_id,
            }
            for node in self.frontier()
        ]
        return {
            "schema_version": _SCHEMA_VERSION,
            "question": _question_to_dict(self.question),
            "current_branch_id": current.branch_id,
            "branches": branch_payload,
            "nodes": [_node_to_dict(node) for node in self.nodes],
            "relationships": [_relationship_to_dict(rel) for rel in self.relationships],
            "expansions": [_expansion_to_dict(exp) for exp in self.expansions],
            "frontier": frontier_payload,
            "gaps": [_gap_to_dict(gap) for gap in self.gaps],
            "actions": [action.to_dict() for action in self.actions],
            "pruned_node_ids": sorted(self.pruned_node_ids),
        }


# ---------------------------------------------------------------------------
# Versioned output
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0.0"


def _question_to_dict(question: Question) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "question": question.question,
        "target_subject": question.target_subject,
        "target_measure": question.target_measure,
        "unit": question.unit,
        "time_basis": question.time_basis,
        "scope": question.scope.to_dict(),
        "status": str(question.status),
        "decision_use": question.decision_use,
        "acceptable_width": question.acceptable_width,
        "unresolved_fields": list(question.unresolved_fields),
    }


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "label": node.label,
        "role": str(node.role),
        "status": str(node.status),
        "parent_id": node.parent_id,
        "unit": node.unit,
        "scope": node.scope.to_dict() if node.scope else None,
        "description": node.description,
        "mechanism": node.mechanism,
        "expansion_id": node.expansion_id,
    }


def _relationship_to_dict(relationship: Relationship) -> dict[str, Any]:
    return {
        "relationship_id": relationship.relationship_id,
        "parent_node_id": relationship.parent_node_id,
        "parent_unit": relationship.parent_unit,
        "expression": relationship.expression,
        "child_node_ids": list(relationship.child_node_ids),
        "child_units": list(relationship.child_units),
        "rationale": relationship.rationale,
    }


def _expansion_to_dict(expansion: Expansion) -> dict[str, Any]:
    return {
        "expansion_id": expansion.expansion_id,
        "target_node_id": expansion.target_node_id,
        "relationship_id": expansion.relationship_id,
        "parent_unit": expansion.parent_unit,
        "projected_unit": expansion.projected_unit,
        "child_node_ids": list(expansion.child_node_ids),
        "rationale": expansion.rationale,
        "is_alternative": expansion.is_alternative,
        "alternative_of_expansion_id": expansion.alternative_of_expansion_id,
        "is_redundant": expansion.is_redundant,
        "redundancy_reason": expansion.redundancy_reason,
    }


def _gap_to_dict(gap: Gap) -> dict[str, Any]:
    return {
        "gap_id": gap.gap_id,
        "kind": str(gap.kind),
        "target": gap.target,
        "explanation": gap.explanation,
        "blocking": gap.blocking,
        "introduced_by_action_id": gap.introduced_by_action_id,
    }


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = "\x1f".join(f"{key}={value}" for key, value in payload.items())
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _append_action(state: DecompositionState, *, kind: ActionKind, payload: Mapping[str, Any],
                   result_summary: str, accepted: bool, error: str | None = None) -> DecompositionState:
    action = ActionRecord(
        action_id=state._next_action_id(),
        kind=kind,
        payload=dict(payload),
        result_summary=result_summary,
        accepted=accepted,
        recorded_at=_now_iso(),
        error=error,
    )
    return replace(
        state,
        actions=state.actions + (action,),
        action_sequence=state.action_sequence + 1,
    )


def _create_root_node(question: Question, node_id: str, *, mechanism: str) -> Node:
    return Node(
        node_id=node_id,
        label=question.target_measure,
        role=NodeRole.TARGET,
        status=NodeStatus.OPEN,
        parent_id=None,
        unit=question.unit,
        scope=question.scope,
        description=question.target_subject,
        mechanism=mechanism,
    )


def create_decomposition(
    question: Question,
    *,
    now: str | None = None,
) -> DecompositionState:
    """Create a fresh decomposition from a raw :class:`Question`.

    The target node is materialised immediately; no formula, variables, or
    domain label is required.  The single default branch is created
    alongside the question root.  Any unresolved fields declared on the
    question are surfaced as an :class:`Gap` of kind
    :attr:`GapKind.INCOMPLETE_ROOT`.
    """

    if not isinstance(question, Question):
        raise DecompositionError("create_decomposition requires a Question instance")
    now = now or _now_iso()
    root_id = "n_0001"
    root = _create_root_node(question, root_id, mechanism="raw question root")
    branch_id = "br_0001"
    branch = Branch(
        branch_id=branch_id,
        root_question_id=question.question_id,
        expansion_ids=(),
        divergent_at_expansion_id=None,
        note="default branch from raw question",
    )
    state = DecompositionState(
        question=question,
        nodes=(root,),
        relationships=(),
        expansions=(),
        branches=(branch,),
        gaps=(),
        actions=(),
        current_branch_id=branch_id,
        pruned_node_ids=frozenset(),
        node_sequence=1,
        relationship_sequence=0,
        expansion_sequence=0,
        branch_sequence=1,
        gap_sequence=0,
        action_sequence=0,
    )
    state = _append_action(
        state,
        kind=ActionKind.CREATE_QUESTION,
        payload={
            "question_id": question.question_id,
            "root_node_id": root_id,
            "timestamp": now,
        },
        result_summary=f"created decomposition root {root_id} for question {question.question_id}",
        accepted=True,
    )
    if not question.is_minimally_complete() or question.unresolved_fields:
        gap_id = "gap_0001"
        gap = Gap(
            gap_id=gap_id,
            kind=GapKind.INCOMPLETE_ROOT,
            target=question.question_id,
            explanation=(
                "question root has unresolved or under-specified fields: "
                + ", ".join(question.unresolved_fields or ("scope anchors",))
            ),
            blocking=True,
        )
        state = replace(
            state,
            gaps=state.gaps + (gap,),
            gap_sequence=1,
        )
    return state


def register_gap(
    state: DecompositionState,
    *,
    kind: GapKind,
    target: str,
    explanation: str,
    blocking: bool = True,
) -> DecompositionState:
    """Append a :class:`Gap` to the state and log an action."""

    if not explanation.strip():
        raise DecompositionError("gap explanation is required")
    gap_id = state._next_gap_id()
    gap = Gap(
        gap_id=gap_id,
        kind=kind,
        target=target,
        explanation=explanation.strip(),
        blocking=blocking,
    )
    new_state = replace(
        state,
        gaps=state.gaps + (gap,),
        gap_sequence=state.gap_sequence + 1,
    )
    return _append_action(
        new_state,
        kind=ActionKind.REGISTER_GAP,
        payload={"gap_id": gap_id, "kind": str(kind), "target": target},
        result_summary=f"recorded gap {gap_id} on {target}",
        accepted=True,
    )


# Public tree operations live behind this stable module facade.
from .tree_mutation import (
    _reachable_node_ids,
    current_branch_projection,
    expand_state,
    frontier,
    mark_node_unresolved,
    prune,
    pruning_projection,
    propose_alternative,
    propose_atom,
    register_node,
)
from .tree_validation import (
    ChildSpec,
    ExpansionRequest,
    _dimension_key_for,
    evaluate_expansion,
)

__all__ = [
    "ChildSpec",
    "DecompositionError",
    "DecompositionState",
    "ExpansionRequest",
    "current_branch_projection",
    "create_decomposition",
    "evaluate_expansion",
    "expand_state",
    "frontier",
    "mark_node_unresolved",
    "prune",
    "pruning_projection",
    "propose_alternative",
    "propose_atom",
    "register_gap",
    "register_node",
]
