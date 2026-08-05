"""Immutable mutation operations for decomposition-tree state."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .decomposition_tree import (
    DecompositionError,
    DecompositionState,
    _append_action,
    register_gap,
)
from .fermi_contracts import (
    ActionKind,
    AtomicClaim,
    Branch,
    DimensionalError,
    Expansion,
    GapKind,
    Node,
    NodeRole,
    NodeStatus,
    ObservationKind,
    QuestionStatus,
    RedundancyReason,
    Relationship,
    Scope,
    parse_compound_unit,
    project_dimensional_closure,
    units_close,
    validate_atomic_claim,
)
from .tree_validation import (
    ChildSpec,
    ExpansionRequest,
    _build_relationship,
    _corresponding_relationship,
    _dimension_key_for,
    _ensure_expandable,
    _pick_dominant_expansion,
    _redundancy_vs,
    _validated_request,
)


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


__all__ = [
    "current_branch_projection",
    "expand_state",
    "frontier",
    "mark_node_unresolved",
    "prune",
    "pruning_projection",
    "propose_alternative",
    "propose_atom",
    "register_node",
]

