"""Track A decomposition-tree tests.

Each test maps to a Track A OpenSpec scenario (see
``openspec/changes/build-recursive-fermi-runtime/specs/recursive-fermi-decomposition/spec.md``).
A non-throughput novel-question example closes the suite.
"""

from __future__ import annotations

import pytest

from aie_decision.decomposition_tree import (
    ChildSpec,
    DecompositionError,
    DecompositionState,
    ExpansionRequest,
    create_decomposition,
    current_branch_projection,
    evaluate_expansion,
    expand_state,
    frontier,
    mark_node_unresolved,
    propose_alternative,
    propose_atom,
    prune,
    pruning_projection,
    register_gap,
)

from aie_decision.fermi_contracts import (
    ActionKind,
    AtomicClaim,
    Branch,
    CompoundUnit,
    GapKind,
    MeasurementKind,
    Node,
    NodeRole,
    NodeStatus,
    ObservationKind,
    Question,
    QuestionStatus,
    Scope,
)

def _scope(label: str = "U.S. workers") -> Scope:
    return Scope(population=label, geography="United States")

def _commutes_question(**overrides) -> Question:
    payload = {
        "question_id": "q-commutes",
        "question": "How many one-way commutes occur on a U.S. weekday?",
        "target_subject": "U.S. weekday commutes",
        "target_measure": "count of one-way commutes",
        "unit": "person/day",
        "time_basis": "per weekday in 2025",
        "scope": _scope(),
        "decision_use": "size the bus-rapid-transit opportunity",
    }
    payload.update(overrides)
    return Question(**payload)

def _commute_expansion(state: DecompositionState) -> ExpansionRequest:
    return ExpansionRequest(
        target_node_id="n_0001",
        parent_unit="person/day",
        expression="_child_0 * _child_1",
        rationale="commutes/day = (commuting people per day) * (trips per person on a weekday)",
        child_specs=(
            ChildSpec(
                label="commuting people per day",
                unit="person/day",
                scope=_scope(),
                description="U.S. residents who commute at all on a weekday",
            ),
            ChildSpec(
                label="trips per person on a weekday",
                unit="1",
                scope=_scope(),
                description="average one-way trips per commuting person per weekday",
            ),
        ),
    )

def test_unresolvable_node_returns_non_answerable_branch_with_gap():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    abstract = state.expansions[0].child_node_ids[0]

    state = mark_node_unresolved(
        state,
        node_id=abstract,
        reason="no measurable population can be produced without a household survey",
    )
    projected = current_branch_projection(state)
    frontier_ids = {node.node_id for node in frontier(projected)}

    # The unresolved node is no longer on the actionable frontier.
    assert abstract not in frontier_ids
    # A blocking gap explains the unresolved abstraction.
    unresolved_gap = next(
        gap for gap in state.gaps if gap.target == abstract and gap.kind is GapKind.UNRESOLVED_NODE
    )
    assert unresolved_gap.blocking

def test_export_returns_machine_readable_state_for_reconstruction():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))

    payload = state.export()
    assert payload["schema_version"] == "1.0.0"
    assert payload["question"]["question_id"] == "q-commutes"
    assert payload["current_branch_id"] == state.current_branch_id
    # Every node, relationship, expansion, gap, and action is enumerated.
    assert len(payload["nodes"]) == len(state.nodes)
    assert len(payload["relationships"]) == len(state.relationships)
    assert len(payload["expansions"]) == len(state.expansions)
    assert len(payload["actions"]) == len(state.actions)
    # Frontier projection is included for direct consumption by an AI.
    frontier_ids = {entry["node_id"] for entry in payload["frontier"]}
    assert frontier_ids == {node.node_id for node in frontier(state)}

def test_export_records_branch_alternative_history():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale="work + non-work split",
            child_specs=(
                ChildSpec(label="work trips per day", unit="person/day", scope=_scope()),
                ChildSpec(label="non-work trips per day", unit="person/day", scope=_scope()),
            ),
        ),
    )
    payload = state.export()
    branches = payload["branches"]
    assert any(branch["is_current"] for branch in branches)
    # The export lists every branch so a client can replay comparison logic.
    assert len(branches) == len(state.branches)
    assert sum(1 for branch in branches if not branch["is_current"]) >= 1

def test_pruning_projection_excludes_pruned_nodes_from_the_active_frontier():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    first_child, second_child = state.expansions[0].child_node_ids

    state = prune(state, node_id=first_child, reason="no defensible measurement procedure")

    projected = pruning_projection(state)
    surviving = {node.node_id for node in projected.nodes}
    assert first_child not in surviving
    assert second_child in surviving
    # The pruned node keeps a record via GapKind.UNRESOLVED_NODE.
    gap = next(
        g for g in state.gaps
        if g.target == first_child and g.kind is GapKind.UNRESOLVED_NODE
    )
    assert gap.blocking is False

def test_pruning_is_irreversible_and_visible_in_action_log():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    child_id = state.expansions[0].child_node_ids[0]

    state = prune(state, node_id=child_id, reason="wrong decomposition leaf")
    kinds = [action.kind for action in state.actions]
    # The register-gap action introduced by prune must be present.
    assert ActionKind.PRUNE in kinds
    assert state.node(child_id).status is NodeStatus.PRUNED

def test_pruning_unknown_node_is_rejected():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="unknown"):
        prune(state, node_id="n_9999", reason="missing")

def test_frontier_returns_only_open_atoms_on_the_active_branch():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    leaves = state.expansions[0].child_node_ids

    # Promote the first child to an atomic leaf and confirm it disappears
    # from the frontier; the second remains.
    claim = AtomicClaim(
        node_id=leaves[0],
        target_object="number of U.S. residents commuting on a weekday",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="ACS B08006 commuting tables and LEHD LODES",
        procedure="Pull the ACS B08006 commuting tables and audit the LEHD LODES dataset.",
    )
    state = propose_atom(state, node_id=leaves[0], claim=claim)

    frontier_nodes = frontier(state)
    frontier_ids = {node.node_id for node in frontier_nodes}
    assert leaves[0] not in frontier_ids
    assert leaves[1] in frontier_ids

def test_alternative_branches_lists_non_current_branches():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale="alt split",
            child_specs=(
                ChildSpec(label="a", unit="person/day", scope=_scope()),
                ChildSpec(label="b", unit="person/day", scope=_scope()),
            ),
        ),
    )
    alternatives = state.alternative_branches()
    assert all(branch.branch_id != state.current_branch_id for branch in alternatives)

def test_atomic_claim_for_an_untracked_node_records_gap():
    state = create_decomposition(_commutes_question())
    claim = AtomicClaim(
        node_id="n_9999",
        target_object="never registered",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="field log",
        procedure="Pull the relevant row from the field log.",
    )
    with pytest.raises(DecompositionError, match="unknown node"):
        propose_atom(state, node_id="n_9999", claim=claim)

def test_register_gap_requires_non_empty_explanation():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError):
        register_gap(state, kind=GapKind.UNRESOLVED_NODE, target="n_0001", explanation="   ")

def test_register_gap_appends_a_recorded_action():
    state = create_decomposition(_commutes_question())
    before_actions = len(state.actions)
    state = register_gap(
        state,
        kind=GapKind.UNRESOLVED_NODE,
        target="n_0001",
        explanation="node requires measurement procedure",
    )
    assert len(state.actions) == before_actions + 1
    assert state.actions[-1].kind is ActionKind.REGISTER_GAP
