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

def test_create_decomposition_records_target_fields_without_a_formula():
    question = _commutes_question()
    state = create_decomposition(question)

    assert state.question.question_id == question.question_id
    assert state.question.target_subject == "U.S. weekday commutes"
    assert state.question.target_measure == "count of one-way commutes"
    assert state.question.unit == "person/day"
    assert state.question.time_basis == "per weekday in 2025"
    assert state.question.scope.geography == "United States"
    assert state.question.is_minimally_complete()
    assert len(state.branches) == 1
    assert state.branches[0].branch_id == "br_0001"
    assert state.branches[0].expansion_ids == ()
    assert state.current_branch_id == "br_0001"

    # The root node is materialised without an expansion or a formula.
    root = state.nodes[0]
    assert root.node_id == "n_0001"
    assert root.role is NodeRole.TARGET
    assert root.parent_id is None
    assert root.status is NodeStatus.OPEN
    assert root.unit == "person/day"
    assert root.mechanism == "raw question root"

    # The action log records the creation as the very first event.
    assert state.actions[0].kind is ActionKind.CREATE_QUESTION
    assert state.actions[0].accepted

def test_create_decomposition_surfaces_incomplete_root_gap():
    question = _commutes_question(scope=Scope())
    state = create_decomposition(question)

    assert any(
        gap.kind is GapKind.INCOMPLETE_ROOT and gap.target == question.question_id
        for gap in state.gaps
    )
