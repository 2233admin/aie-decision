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


# ---------------------------------------------------------------------------
# Node & relationship registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChildSpec:
    """Specification of a child node to materialise with an expansion."""

    label: str
    unit: str
    scope: Scope
    description: str = ""
    mechanism: str = ""


def register_node(
    state: DecompositionState,
    *,
    label: str,
    unit: str | None,
    scope: Scope | None,
    description: str = "",
    role: NodeRole = NodeRole.CHILD,
    parent_id: str | None = None,
    mechanism: str = "",
) -> DecompositionState:
    """Attach an unregistered node to the tree."""

    if not label.strip():
        raise DecompositionError("node label is required")
    if scope is not None and not isinstance(scope, Scope):
        raise DecompositionError("scope must be a Scope instance")
    if unit is not None:
        parse_compound_unit(unit)
    node_id = state._next_node_id()
    node = Node(
        node_id=node_id,
        label=label.strip(),
        role=role,
        status=NodeStatus.OPEN,
        parent_id=parent_id,
        unit=unit,
        scope=scope,
        description=description.strip(),
        mechanism=mechanism.strip(),
    )
    new_state = replace(state, nodes=state.nodes + (node,), node_sequence=state.node_sequence + 1)
    return _append_action(
        new_state,
        kind=ActionKind.REGISTER_NODE,
        payload={"node_id": node_id, "label": label, "parent_id": parent_id},
        result_summary=f"registered node {node_id} ({label})",
        accepted=True,
    )


# ---------------------------------------------------------------------------
# Expansion (the core structural mutation)
# ---------------------------------------------------------------------------


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


def _materialise_children(
    state: DecompositionState,
    *,
    parent_node_id: str,
    specs: Sequence[ChildSpec],
) -> tuple[DecompositionState, tuple[str, ...]]:
    new_state = state
    ids: list[str] = []
    for spec in specs:
        new_state = register_node(
            new_state,
            label=spec.label,
            unit=spec.unit,
            scope=spec.scope,
            description=spec.description,
            role=NodeRole.CHILD,
            parent_id=parent_node_id,
            mechanism=spec.mechanism,
        )
        ids.append(new_state.nodes[-1].node_id)
    return new_state, tuple(ids)


def _expand(
    state: DecompositionState,
    *,
    request: ExpansionRequest,
    alternative_of: Expansion | None,
    is_redundant: tuple[bool, str | None],
) -> DecompositionState:
    target_node = _ensure_expandable(state, request.target_node_id)

    parent_unit_text = request.parent_unit.strip()
    parent_unit = parse_compound_unit(parent_unit_text)
    child_units = tuple(spec.unit for spec in request.child_specs)
    if not child_units:
        raise DecompositionError("expansion must declare at least one child")
    try:
        bound_expression = project_dimensional_closure(
            parent_unit_text,
            request.expression,
            child_units,
        )
    except DimensionalError as exc:
        gap_state = register_gap(
            state,
            kind=GapKind.UNIT_MISMATCH,
            target=request.target_node_id,
            explanation=f"relationship {request.expression!r} cannot produce parent unit {parent_unit_text}: {exc}",
            blocking=True,
        )
        return gap_state

    # Materialise children + register the relationship.
    materialised_state, child_node_ids = _materialise_children(
        state,
        parent_node_id=request.target_node_id,
        specs=request.child_specs,
    )
    relationship_state, relationship = _build_relationship(
        materialised_state,
        expansion_id="placeholder",  # generated below
        parent_node_id=request.target_node_id,
        request=request,
        child_node_ids=child_node_ids,
        bound_expression=bound_expression,
    )

    # Materialise the expansion entry.
    new_expansion_seq = relationship_state.expansion_sequence + 1
    expansion_id = f"exp_{new_expansion_seq:04d}"
    expansion = Expansion(
        expansion_id=expansion_id,
        target_node_id=request.target_node_id,
        relationship_id=relationship.relationship_id,
        parent_unit=parent_unit_text,
        projected_unit=_dimension_key_for(parent_unit),
        child_node_ids=child_node_ids,
        rationale=request.rationale.strip(),
        is_alternative=alternative_of is not None,
        alternative_of_expansion_id=alternative_of.expansion_id if alternative_of else None,
        is_redundant=is_redundant[0],
        redundancy_reason=is_redundant[1],
    )
    branch_state = _attach_to_branch(
        relationship_state,
        expansion=expansion,
        target_node_id=request.target_node_id,
        is_alternative=alternative_of is not None,
    )
    expansion_state = replace(
        branch_state,
        expansions=branch_state.expansions + (expansion,),
        expansion_sequence=new_expansion_seq,
    )

    # Update the parent node's status.
    target_index = next(
        index
        for index, node in enumerate(expansion_state.nodes)
        if node.node_id == request.target_node_id
    )
    updated_nodes = list(expansion_state.nodes)
    updated_nodes[target_index] = replace(target_node, status=NodeStatus.EXPANDED)
    expansion_state = replace(expansion_state, nodes=tuple(updated_nodes))

    # Mark children as atom candidates.
    final_nodes: list[Node] = []
    for node in expansion_state.nodes:
        if node.node_id in child_node_ids and node.status is NodeStatus.OPEN:
            final_nodes.append(replace(node, role=NodeRole.ATOM_CANDIDATE))
        else:
            final_nodes.append(node)
    final_state = replace(expansion_state, nodes=tuple(final_nodes))

    summary = expansion.describe()
    return _append_action(
        final_state,
        kind=ActionKind.PROPOSE_ALTERNATIVE if alternative_of is not None else ActionKind.EXPAND,
        payload={
            "expansion_id": expansion_id,
            "target_node_id": request.target_node_id,
            "child_node_ids": list(child_node_ids),
            "expression": request.expression,
            "parent_unit": parent_unit_text,
            "is_alternative": alternative_of is not None,
            "is_redundant": expansion.is_redundant,
        },
        result_summary=summary,
        accepted=True,
    )


def expand_state(state: DecompositionState, *, request: ExpansionRequest) -> DecompositionState:
    """Expand ``request.target_node_id`` with a new relationship and children.

    The expansion extends the current branch.  If the target node already
    had an expansion on the current branch the new expansion becomes the
    dominant alternative on a freshly forked alternative branch.
    """

    request = _validated_request(request)
    target_existing = state.expansions_at(request.target_node_id)
    branch = state.current_branch()
    alternative_on_current_branch = [
        item for item in target_existing if item.expansion_id in branch.expansion_ids
    ]
    if alternative_on_current_branch:
        # The AI is replacing or adding an alternative at the same node on
        # the current branch.  Treat the request as an alternative.
        current_alternative = alternative_on_current_branch[0]
        return _propose_alternative_inner(state, request=request, alternative_of=current_alternative)
    return _expand(state, request=request, alternative_of=None, is_redundant=(False, None))


def propose_alternative(state: DecompositionState, *, request: ExpansionRequest) -> DecompositionState:
    """Register an alternative expansion at any node."""

    request = _validated_request(request)
    target_existing = state.expansions_at(request.target_node_id)
    if not target_existing:
        raise DecompositionError(
            "an alternative must compete with at least one existing expansion"
        )
    dominant = _pick_dominant_expansion(state, target_existing)
    return _propose_alternative_inner(state, request=request, alternative_of=dominant)


def _propose_alternative_inner(
    state: DecompositionState,
    *,
    request: ExpansionRequest,
    alternative_of: Expansion,
) -> DecompositionState:
    """Submit ``request`` as an alternative to ``alternative_of``.

    Redundancy is evaluated against ``alternative_of`` first (the most
    common case: an algebraically equivalent rewrite).  If that does not
    match, additional sibling expansions at the same node are also
    considered.
    """

    anchor = _corresponding_relationship(state, alternative_of)
    redundant = _redundancy_vs(request=request, alternative_of=alternative_of, relationship=anchor)
    if not redundant[0]:
        candidate_existing = [
            (item, _corresponding_relationship(state, item))
            for item in state.expansions_at(request.target_node_id)
            if item.expansion_id != alternative_of.expansion_id
        ]
        for _expansion, relationship in candidate_existing:
            redundant = _redundancy_vs(request=request, alternative_of=alternative_of, relationship=relationship)
            if redundant[0]:
                break
    new_state = _expand(
        state,
        request=request,
        alternative_of=alternative_of,
        is_redundant=redundant,
    )
    if redundant[0]:
        new_state = register_gap(
            new_state,
            kind=GapKind.REDUNDANT_ALTERNATIVE,
            target=request.target_node_id,
            explanation=(
                "alternative is algebraically equivalent to existing expansion "
                f"{alternative_of.expansion_id}: {redundant[1]}"
            ),
            blocking=False,
        )
    return new_state


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


def _attach_to_branch(
    state: DecompositionState,
    *,
    expansion: Expansion,
    target_node_id: str,
    is_alternative: bool,
) -> DecompositionState:
    branches = list(state.branches)
    if is_alternative:
        # Fork a new branch whose lineage diverges at this expansion.
        dominant = state.current_branch()
        new_id = state._next_branch_id()
        lineage: tuple[str, ...] = dominant.expansion_ids + (expansion.expansion_id,)
        new_branch = Branch(
            branch_id=new_id,
            root_question_id=state.question.question_id,
            expansion_ids=lineage,
            divergent_at_expansion_id=expansion.expansion_id,
            note=f"alternative branch over {target_node_id}",
        )
        branches.append(new_branch)
        new_sequence = state.branch_sequence + 1
    else:
        # Extend the current branch lineage.
        current = state.current_branch()
        updated = replace(current, expansion_ids=current.expansion_ids + (expansion.expansion_id,))
        branches = [
            updated if branch.branch_id == current.branch_id else branch
            for branch in branches
        ]
        new_sequence = state.branch_sequence
    return replace(state, branches=tuple(branches), branch_sequence=new_sequence)


def _dimension_key_for(unit: CompoundUnit) -> str:
    if unit.is_dimensionless:
        return "dimensionless"
    return "*".join(f"{label}^{exp}" for label, exp in unit.to_canonical())


# ---------------------------------------------------------------------------
# Frontier, branches, pruning
# ---------------------------------------------------------------------------


def _reachable_node_ids(state: DecompositionState, expansion_ids: set[str]) -> set[str]:
    """All node ids reachable via expansions the current branch endorses.

    The target root is always reachable regardless of expansion lineage so
    that the very first question (which has no expansions yet) still
    surfaces on the frontier for the AI to act on.
    """

    reachable: set[str] = {node.node_id for node in state.nodes if node.parent_id is None}
    for expansion in state.expansions:
        if expansion.expansion_id not in expansion_ids:
            continue
        reachable.add(expansion.target_node_id)
        reachable.update(expansion.child_node_ids)
    return reachable


def frontier(state: DecompositionState) -> tuple[Node, ...]:
    """Public wrapper returning the active frontier nodes."""

    return state.frontier()


def current_branch_projection(state: DecompositionState) -> DecompositionState:
    """Return a state where only the current branch is materialised."""

    branch = state.current_branch()
    kept_expansions = tuple(
        item for item in state.expansions if item.expansion_id in branch.expansion_ids
    )
    kept_child_ids = set()
    for expansion in kept_expansions:
        kept_child_ids.update(expansion.child_node_ids)
    kept_nodes = tuple(
        node
        for node in state.nodes
        if node.node_id == "n_0001" or node.node_id in kept_child_ids or node.parent_id is None
    )
    new_state = replace(state, nodes=kept_nodes, expansions=kept_expansions)
    return _append_action(
        new_state,
        kind=ActionKind.ACTIVATE_BRANCH,
        payload={"branch_id": branch.branch_id},
        result_summary=f"projected current branch {branch.branch_id}",
        accepted=True,
    )


def pruning_projection(state: DecompositionState) -> DecompositionState:
    """Return a copy of the state that excises the pruned subtree from the
    active branch line.

    Pruned nodes remain in the historical record (the tuples still carry
    them) but the current frontier excludes them and the active branch's
    reachable set is recomputed without going through them.
    """

    if not state.pruned_node_ids:
        return state
    pruned = state.pruned_node_ids
    branch = state.current_branch()
    reachable: set[str] = set()
    for expansion in state.expansions:
        if expansion.expansion_id not in branch.expansion_ids:
            continue
        if expansion.target_node_id in pruned:
            continue
        reachable.add(expansion.target_node_id)
        for child_id in expansion.child_node_ids:
            if child_id not in pruned:
                reachable.add(child_id)
    surviving_nodes = tuple(node for node in state.nodes if node.node_id not in pruned)
    frontier_candidates = [node for node in surviving_nodes if node.node_id in reachable]
    if frontier_candidates or surviving_nodes:
        new_state = replace(state, nodes=surviving_nodes)
    else:
        new_state = state
    return _append_action(
        new_state,
        kind=ActionKind.PRUNE,
        payload={"pruned_node_ids": sorted(pruned)},
        result_summary=f"applied pruning projection over {len(pruned)} node(s)",
        accepted=True,
    )


def prune(state: DecompositionState, *, node_id: str, reason: str) -> DecompositionState:
    """Mark ``node_id`` (and its descendants) as pruned and log a gap."""

    if not node_id.strip():
        raise DecompositionError("prune requires a node_id")
    if not reason.strip():
        raise DecompositionError("prune requires a reason")
    if not any(node.node_id == node_id for node in state.nodes):
        raise DecompositionError(f"cannot prune unknown node {node_id}")
    descendants = _descendants(state, node_id)
    new_pruned = state.pruned_node_ids | {node_id, *descendants}
    updated_nodes = tuple(
        replace(node, status=NodeStatus.PRUNED) if node.node_id in new_pruned else node
        for node in state.nodes
    )
    new_state = replace(state, pruned_node_ids=new_pruned, nodes=updated_nodes)
    new_state = register_gap(
        new_state,
        kind=GapKind.UNRESOLVED_NODE,
        target=node_id,
        explanation=f"node pruned: {reason.strip()}",
        blocking=False,
    )
    return _append_action(
        new_state,
        kind=ActionKind.PRUNE,
        payload={"node_id": node_id, "pruned_node_ids": sorted(new_pruned), "reason": reason.strip()},
        result_summary=f"pruned subtree rooted at {node_id} ({len(descendants)} descendants)",
        accepted=True,
    )


def _descendants(state: DecompositionState, node_id: str) -> set[str]:
    descendants: set[str] = set()
    work: list[str] = [node_id]
    while work:
        current = work.pop()
        expansions_here = state.expansions_at(current)
        for expansion in expansions_here:
            for child in expansion.child_node_ids:
                if child not in descendants and child != node_id:
                    descendants.add(child)
                    work.append(child)
    return descendants


# ---------------------------------------------------------------------------
# Atomic claims
# ---------------------------------------------------------------------------


def propose_atom(state: DecompositionState, *, node_id: str, claim: AtomicClaim) -> DecompositionState:
    """Promote a node to an atomic leaf on the strength of an :class:`AtomicClaim`."""

    if not isinstance(claim, AtomicClaim):
        raise DecompositionError("propose_atom requires an AtomicClaim")
    if claim.node_id != node_id:
        raise DecompositionError("claim node_id must match the node being promoted")
    node = state.node(node_id)
    errors = validate_atomic_claim(claim)
    if errors:
        gap_state = register_gap(
            state,
            kind=GapKind.ATOM_REJECTED,
            target=node_id,
            explanation="atomic claim rejected: " + "; ".join(errors),
            blocking=True,
        )
        return gap_state
    if claim.unit and node.unit:
        if not units_close(parse_compound_unit(claim.unit), parse_compound_unit(node.unit)):
            return register_gap(
                state,
                kind=GapKind.UNIT_MISMATCH,
                target=node_id,
                explanation=(
                    f"atomic claim unit {claim.unit!r} does not match node "
                    f"unit {node.unit!r}"
                ),
                blocking=True,
            )
    updated = tuple(
        replace(item, status=NodeStatus.ATOMIC_LEAF) if item.node_id == node_id else item
        for item in state.nodes
    )
    new_question = state.question
    if _question_can_be_marked_atomic(state):
        new_question = replace(new_question, status=QuestionStatus.ATOMIC_LEAF)
    promoted = replace(state, nodes=updated, question=new_question)
    return _append_action(
        promoted,
        kind=ActionKind.PROPOSE_ATOM,
        payload={
            "node_id": node_id,
            "target_object": claim.target_object,
            "unit": claim.unit,
            "measurement_kind": str(claim.measurement_kind),
            "source": claim.source,
            "procedure_excerpt": claim.procedure[:120],
        },
        result_summary=f"node {node_id} accepted as atomic leaf",
        accepted=True,
    )


def _question_can_be_marked_atomic(state: DecompositionState) -> bool:
    """Decide whether promoting one leaf should also mark the question atomic.

    The root reaches an atomic/completed state only when an explicit
    tree-level condition is met: every node reachable on the current
    branch is already an atomic leaf or a pruned/unresolved gap that
    blocks further evaluation.  Promoting a single descendant is *not*
    sufficient — the question would otherwise flip to ``ATOMIC_LEAF``
    every time one child becomes measurable, which contradicts the
    recursive decomposition the product direction mandates.
    """

    if state.question.status is not QuestionStatus.OPEN:
        return False
    branch = state.current_branch()
    reachable = _reachable_node_ids(state, set(branch.expansion_ids))
    if not reachable:
        return False
    for node in state.nodes:
        if node.node_id not in reachable:
            continue
        if node.node_id in state.pruned_node_ids:
            continue
        if node.status is NodeStatus.ATOMIC_LEAF:
            continue
        if node.status in (NodeStatus.PRUNED, NodeStatus.UNRESOLVED, NodeStatus.REJECTED):
            continue
        return False
    return True


def mark_node_unresolved(state: DecompositionState, *, node_id: str, reason: str) -> DecompositionState:
    if not any(node.node_id == node_id for node in state.nodes):
        raise DecompositionError(f"cannot mark unknown node {node_id} unresolved")
    updated = tuple(
        replace(node, status=NodeStatus.UNRESOLVED) if node.node_id == node_id else node
        for node in state.nodes
    )
    new_state = replace(state, nodes=updated)
    return register_gap(
        new_state,
        kind=GapKind.UNRESOLVED_NODE,
        target=node_id,
        explanation=reason.strip(),
        blocking=True,
    )


# ---------------------------------------------------------------------------
# Evaluation helper for restricted numeric evaluation
# ---------------------------------------------------------------------------


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
