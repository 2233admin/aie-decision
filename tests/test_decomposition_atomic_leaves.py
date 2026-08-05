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

def test_abstract_label_submitted_as_leaf_is_rejected_and_kept_on_frontier():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    abstract_node_id = state.expansions[0].child_node_ids[0]

    claim = AtomicClaim(
        node_id=abstract_node_id,
        target_object="operating efficiency",
        unit="ratio",
        scope=_scope(),
        time_basis="per day",
        # A bare claim that still omits a source and procedure must be
        # rejected, even though the textual content reads as an action.
        measurement_kind=MeasurementKind.DIRECT_OBSERVATION,
        source="",
        procedure="",
    )
    before_gap_count = len(state.gaps)
    state = propose_atom(state, node_id=abstract_node_id, claim=claim)
    after_gap_count = len(state.gaps)

    assert after_gap_count > before_gap_count
    new_gap = state.gaps[-1]
    assert new_gap.kind is GapKind.ATOM_REJECTED
    assert new_gap.target == abstract_node_id
    assert new_gap.blocking is True
    # The abstract node stays open on the frontier.
    frontier_ids = {node.node_id for node in frontier(state)}
    assert abstract_node_id in frontier_ids
    assert state.node(abstract_node_id).status is NodeStatus.OPEN

def test_concrete_atom_is_accepted_and_recorded_with_procedure():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    target = state.expansions[0].child_node_ids[0]

    claim = AtomicClaim(
        node_id=target,
        target_object="number of U.S. residents commuting on a weekday",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="Census Bureau ACS commuting-flow tables (B08006) and LEHD LODES",
        procedure=(
            "Pull the ACS B08006 commuting-flow table and cross-check the LEHD "
            "origin-destination employment statistics to obtain a daily count."
        ),
        observation_kind=ObservationKind.OBSERVED,
    )
    state = propose_atom(state, node_id=target, claim=claim)
    promoted = state.node(target)

    assert promoted.status is NodeStatus.ATOMIC_LEAF
    # The question does NOT automatically become atomic just because a child
    # did — the second child is still open and the tree is not whole.
    assert state.question.status is QuestionStatus.OPEN
    assert any(action.kind is ActionKind.PROPOSE_ATOM for action in state.actions)

def test_concrete_atom_with_unit_mismatch_records_blocking_gap():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    target = state.expansions[0].child_node_ids[0]
    claim = AtomicClaim(
        node_id=target,
        target_object="number of U.S. residents commuting on a weekday",
        unit="kg",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="ACS B08006",
        procedure="Pull the ACS B08006 commuting-flow table for the daily count.",
    )
    before_gaps = len(state.gaps)
    state = propose_atom(state, node_id=target, claim=claim)
    assert len(state.gaps) > before_gaps
    gap = state.gaps[-1]
    assert gap.kind is GapKind.UNIT_MISMATCH
    assert gap.target == target

def test_promoting_one_child_does_not_mark_the_question_atomic():
    """Defect: the previous implementation flipped the question to ATOMIC_LEAF
    whenever a single child became a leaf, hiding the remaining frontier.
    """

    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    leaves = state.expansions[0].child_node_ids

    claim = AtomicClaim(
        node_id=leaves[0],
        target_object="number of U.S. residents commuting on a weekday",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="ACS B08006",
        procedure="Pull the ACS B08006 commuting-flow table for the daily count.",
    )
    state = propose_atom(state, node_id=leaves[0], claim=claim)
    assert state.node(leaves[0]).status is NodeStatus.ATOMIC_LEAF
    assert state.question.status is QuestionStatus.OPEN
    # The other child is still on the frontier.
    frontier_ids = {node.node_id for node in frontier(state)}
    assert leaves[1] in frontier_ids

def test_question_reaches_atomic_only_when_every_branch_leaf_is_admissible():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    leaves = list(state.expansions[0].child_node_ids)

    def _claim(node_id: str) -> AtomicClaim:
        node = state.node(node_id)
        return AtomicClaim(
            node_id=node_id,
            target_object="commuting people count",
            unit=node.unit,
            scope=_scope(),
            time_basis="per weekday in 2025",
            measurement_kind=MeasurementKind.COUNT,
            source="ACS B08006",
            procedure="Pull the ACS B08006 commuting-flow table for the daily count.",
        )

    state = propose_atom(state, node_id=leaves[0], claim=_claim(leaves[0]))
    assert state.question.status is QuestionStatus.OPEN
    state = propose_atom(state, node_id=leaves[1], claim=_claim(leaves[1]))
    assert state.node(leaves[0]).status is NodeStatus.ATOMIC_LEAF
    assert state.node(leaves[1]).status is NodeStatus.ATOMIC_LEAF
    # Question remains OPEN until an explicit root-level condition is met;
    # a child becoming ATOMIC_LEAF does not automatically mark the question complete.
    assert state.question.status is QuestionStatus.OPEN
