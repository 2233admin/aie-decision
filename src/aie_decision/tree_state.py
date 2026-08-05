"""Immutable state records and export projections for decomposition trees."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Mapping

from .fermi_contracts import (
    ActionKind,
    ActionRecord,
    Branch,
    Expansion,
    FermiContractError,
    Gap,
    Node,
    NodeRole,
    NodeStatus,
    Question,
    RedundancyReason,
    Relationship,
)


_SCHEMA_VERSION = "1.0.0"


class DecompositionError(FermiContractError):
    """Raised when a structural mutation cannot be admitted to the tree."""


@dataclass(frozen=True, slots=True)
class DecompositionState:
    """Immutable decomposition-tree state.

    Every mutation returns a new state with copied tuples and updated sequence
    counters. The persisted record therefore never rewrites history;
    trajectory recording replays from :attr:`actions`.
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
        return tuple(self.node(child_id) for child_id in expansions[0].child_node_ids)

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

        # Imported lazily to keep the state record independent of mutation
        # policy while preserving the method-level public API.
        from .tree_mutation import _reachable_node_ids

        branch_expansion_ids = set(self.current_branch().expansion_ids)
        reachable = _reachable_node_ids(self, branch_expansion_ids)
        frontier: list[Node] = []
        for node in self.nodes:
            if node.node_id not in reachable or node.node_id in self.pruned_node_ids:
                continue
            if node.status in {
                NodeStatus.ATOMIC_LEAF,
                NodeStatus.UNRESOLVED,
                NodeStatus.REJECTED,
                NodeStatus.PRUNED,
            }:
                continue
            expansions_here = [
                expansion
                for expansion in self.expansions
                if expansion.target_node_id == node.node_id
                and expansion.expansion_id in branch_expansion_ids
            ]
            if expansions_here and node.status is NodeStatus.EXPANDED:
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
        branches = [
            {
                "branch_id": branch.branch_id,
                "root_question_id": branch.root_question_id,
                "expansion_ids": list(branch.expansion_ids),
                "divergent_at_expansion_id": branch.divergent_at_expansion_id,
                "note": branch.note,
                "is_current": branch.branch_id == current.branch_id,
            }
            for branch in self.branches
        ]
        frontier = [
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
            "branches": branches,
            "nodes": [_node_to_dict(node) for node in self.nodes],
            "relationships": [_relationship_to_dict(item) for item in self.relationships],
            "expansions": [_expansion_to_dict(item) for item in self.expansions],
            "frontier": frontier,
            "gaps": [_gap_to_dict(gap) for gap in self.gaps],
            "actions": [action.to_dict() for action in self.actions],
            "pruned_node_ids": sorted(self.pruned_node_ids),
        }


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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _append_action(
    state: DecompositionState,
    *,
    kind: ActionKind,
    payload: Mapping[str, Any],
    result_summary: str,
    accepted: bool,
    error: str | None = None,
) -> DecompositionState:
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
