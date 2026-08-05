"""Validation policy for decomposition-tree expansion requests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .decomposition_tree import DecompositionError, DecompositionState
from .fermi_contracts import (
    CompoundUnit,
    Expansion,
    Node,
    NodeStatus,
    Relationship,
    RestrictedExpression,
    Scope,
    evaluate_restricted_expression,
    expressions_are_equivalent,
    parse_restricted_expression,
)


@dataclass(frozen=True, slots=True)
class ChildSpec:
    """Specification of a child node to materialise with an expansion."""

    label: str
    unit: str
    scope: Scope
    description: str = ""
    mechanism: str = ""


@dataclass(frozen=True, slots=True)
class ExpansionRequest:
    """Payload submitted to :func:`expand_state` and :func:`propose_alternative`."""

    target_node_id: str
    parent_unit: str
    expression: str
    child_specs: tuple[ChildSpec, ...]
    rationale: str


def _build_relationship(
    state: DecompositionState,
    *,
    expansion_id: str,
    parent_node_id: str,
    request: ExpansionRequest,
    child_node_ids: tuple[str, ...],
    bound_expression: RestrictedExpression,
) -> tuple[DecompositionState, Relationship]:
    relationship_id = state._next_relationship_id()
    relationship = Relationship(
        relationship_id=relationship_id,
        parent_node_id=parent_node_id,
        parent_unit=request.parent_unit.strip(),
        expression=request.expression.strip(),
        child_node_ids=child_node_ids,
        child_units=tuple(spec.unit for spec in request.child_specs),
        rationale=request.rationale.strip(),
        expression_ast=bound_expression,
    )
    new_state = replace(
        state,
        relationships=state.relationships + (relationship,),
        relationship_sequence=state.relationship_sequence + 1,
    )
    return new_state, relationship


def _redundancy_vs(
    *,
    request: ExpansionRequest,
    alternative_of: Expansion,
    relationship: Relationship,
) -> tuple[bool, str | None]:
    request_expression = parse_restricted_expression(request.expression)
    candidate_expression = parse_restricted_expression(relationship.expression)
    request_units = sorted(tuple(spec.unit for spec in request.child_specs))
    candidate_units = sorted(tuple(relationship.child_units))
    if request_units != candidate_units:
        return False, None
    if expressions_are_equivalent(request_expression, candidate_expression):
        return True, (
            f"matches {alternative_of.expansion_id} by commutative-canonical "
            f"signature {request_expression.signature!r}"
        )
    return False, None


def _pick_dominant_expansion(state: DecompositionState, candidates: Sequence[Expansion]) -> Expansion:
    if not candidates:
        raise DecompositionError("no candidate expansion to use as alternative anchor")
    if len(candidates) == 1:
        return candidates[0]
    branch = state.current_branch()
    for item in candidates:
        if item.expansion_id in branch.expansion_ids:
            return item
    return candidates[0]


def _corresponding_relationship(state: DecompositionState, expansion: Expansion) -> Relationship:
    for rel in state.relationships:
        if rel.relationship_id == expansion.relationship_id:
            return rel
    raise DecompositionError(
        f"orphan expansion {expansion.expansion_id}: no matching relationship"
    )


def _validated_request(request: ExpansionRequest) -> ExpansionRequest:
    """Reject malformed payloads before they reach the dimension checker."""

    if not isinstance(request, ExpansionRequest):
        raise DecompositionError("expansion requires an ExpansionRequest")
    if not request.rationale.strip():
        raise DecompositionError("expansion rationale is required")
    if not request.child_specs:
        raise DecompositionError("expansion must declare at least one child")
    if not request.parent_unit.strip():
        raise DecompositionError("parent_unit is required for an expansion")
    if not request.expression.strip():
        raise DecompositionError("expression is required for an expansion")
    # ``parse_restricted_expression`` already rejects calls, attribute access,
    # comprehensions, and non-numeric constants, so we lean on it here to
    # surface restricted-form violations at the request boundary.
    parse_restricted_expression(request.expression)
    return request


def _ensure_expandable(state: DecompositionState, node_id: str) -> Node:
    node = state.node(node_id)
    if node.status in (NodeStatus.ATOMIC_LEAF,):
        raise DecompositionError(f"node {node_id} is already an atomic leaf")
    if node.status in (NodeStatus.PRUNED,):
        raise DecompositionError(f"node {node_id} has been pruned")
    return node


def _dimension_key_for(unit: CompoundUnit) -> str:
    if unit.is_dimensionless:
        return "dimensionless"
    return "*".join(f"{label}^{exp}" for label, exp in unit.to_canonical())


def evaluate_expansion(
    state: DecompositionState,
    *,
    expansion_id: str,
    values: Mapping[str, float],
) -> float:
    """Numerically evaluate an expansion's bound expression.

    ``values`` may map either child node identifiers (``n_0002``) or
    the position-based variable names (``_child_0``) used by the
    relationship.  Numeric constants in the expression are honoured.
    Domain-neutral; Track B must supply distributions.
    """

    expansion = state.expansion(expansion_id)
    relationship = state.relationship(expansion.relationship_id)
    if relationship.expression_ast is None:
        raise DecompositionError(
            f"relationship {relationship.relationship_id} was not bound to dimensions"
        )
    bound: dict[str, float] = {}
    for index, child_id in enumerate(expansion.child_node_ids):
        suffix = f"_{index}" if len(expansion.child_node_ids) > 1 else "_0"
        variable_name = f"_child{suffix}"
        try:
            bound[variable_name] = float(values[child_id])
        except KeyError:
            bound[variable_name] = float(values[variable_name])
    return evaluate_restricted_expression(relationship.expression_ast, bound)


__all__ = ["ChildSpec", "ExpansionRequest", "evaluate_expansion"]
