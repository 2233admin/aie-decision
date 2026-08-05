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

def test_expand_an_abstract_target_into_three_children_with_unit_closure():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))

    expansion = state.expansions[0]
    relationship = state.relationships[0]
    assert expansion.target_node_id == "n_0001"
    assert relationship.parent_unit == "person/day"
    assert len(expansion.child_node_ids) == 2
    assert state.node("n_0001").status is NodeStatus.EXPANDED
    for child_id in expansion.child_node_ids:
        child = state.node(child_id)
        assert child.parent_id == "n_0001"
        assert child.role is NodeRole.ATOM_CANDIDATE
    # Numeric evaluation reproduces the parent unit when child values are bound.
    numeric = evaluate_expansion(
        state,
        expansion_id=expansion.expansion_id,
        values={"n_0002": 50_000_000, "n_0003": 2.0},
    )
    assert numeric == 100_000_000.0
    # Frontier exposes the freshly materialised children for further work.
    frontier_nodes = frontier(state)
    assert {node.node_id for node in frontier_nodes} == set(expansion.child_node_ids)

def test_expansion_extends_the_current_branch_lineage():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    assert state.branches[0].expansion_ids == (state.expansions[0].expansion_id,)

def test_existing_branch_and_current_branch_are_stable_across_reads():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    branch = state.current_branch()
    assert isinstance(branch, Branch)
    assert branch.branch_id == state.current_branch_id

def test_expansion_with_unit_mismatch_leaves_the_prior_tree_intact():
    state = create_decomposition(_commutes_question())
    baseline_nodes = tuple(state.nodes)
    baseline_expansions = tuple(state.expansions)

    mismatched = ExpansionRequest(
        target_node_id="n_0001",
        parent_unit="person/day",
        expression="_child_0 + _child_1",
        rationale="addition is invalid across heterogeneous units",
        child_specs=(
            ChildSpec(label="mass", unit="kg", scope=_scope()),
            ChildSpec(label="length", unit="m", scope=_scope()),
        ),
    )
    state = expand_state(state, request=mismatched)

    # The tree is unchanged — no children were added.
    assert tuple(state.nodes) == baseline_nodes
    assert tuple(state.expansions) == baseline_expansions
    # But a gap records why the attempt was rejected.
    assert any(
        gap.kind is GapKind.UNIT_MISMATCH and gap.target == "n_0001" and gap.blocking
        for gap in state.gaps
    )


def test_rejected_unit_mapping_cannot_leak_attacker_children_into_the_tree():
    """Adversarial: rejected child payloads never become visible nodes."""
    baseline = create_decomposition(_commutes_question())
    rejected = expand_state(
        baseline,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 * _child_1",
            rationale="attempt to smuggle incompatible child nodes",
            child_specs=(
                ChildSpec(label="poison-mass", unit="kg", scope=_scope()),
                ChildSpec(label="poison-length", unit="m", scope=_scope()),
            ),
        ),
    )

    assert all(
        node.label not in {"poison-mass", "poison-length"}
        for node in rejected.nodes
    )
    assert rejected.current_branch() == baseline.current_branch()
    assert rejected.frontier() == baseline.frontier()

def test_expansion_with_incompatible_unit_records_blocking_gap():
    state = create_decomposition(_commutes_question())
    state = expand_state(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="kg",
            expression="_child_0 + _child_1",
            rationale="incompatible",
            child_specs=(
                ChildSpec(label="a", unit="kg", scope=_scope()),
                ChildSpec(label="b", unit="m", scope=_scope()),
            ),
        ),
    )
    gap = next(gap for gap in state.gaps if gap.target == "n_0001")
    assert gap.kind is GapKind.UNIT_MISMATCH
    assert gap.blocking is True
