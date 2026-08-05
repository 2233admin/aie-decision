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

def test_alternative_expansion_records_a_new_branch_without_replacing_the_default():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    baseline_branch_count = len(state.branches)

    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale=(
                "split commutes into work commutes and non-work commutes because the "
                "underlying drivers differ"
            ),
            child_specs=(
                ChildSpec(label="work trips per day", unit="person/day", scope=_scope()),
                ChildSpec(label="non-work trips per day", unit="person/day", scope=_scope()),
            ),
        ),
    )

    assert len(state.branches) == baseline_branch_count + 1
    assert state.expansions[-1].is_alternative
    # The new branch diverges at the alternative expansion.
    divergent = state.branches[-1]
    assert divergent.divergent_at_expansion_id == state.expansions[-1].expansion_id
    assert divergent.expansion_ids[0] == state.expansions[0].expansion_id
    # The alternative and the dominant both remain accessible for comparison.
    alt_id = state.expansions[-1].expansion_id
    assert any(
        branch.branch_id != state.current_branch_id and alt_id in branch.expansion_ids
        for branch in state.branches
    )

def test_algebraically_equivalent_alternative_is_marked_redundant():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    dominant_id = state.expansions[0].expansion_id

    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_1 * _child_0",  # commutative rewrite
            rationale="identical algebra with renamed children",
            child_specs=(
                ChildSpec(label="trips per person on a weekday (alt)", unit="1", scope=_scope()),
                ChildSpec(label="commuting people per day (alt)", unit="person/day", scope=_scope()),
            ),
        ),
    )

    new_expansion = state.expansions[-1]
    assert new_expansion.is_alternative
    assert new_expansion.is_redundant
    assert new_expansion.alternative_of_expansion_id == dominant_id
    assert "matches" in (new_expansion.redundancy_reason or "")
    # The current branch stays unchanged: the redundant alt does not become a
    # distinct search branch the runtime needs to compare against.
    current = state.current_branch()
    assert dominant_id in current.expansion_ids
    assert new_expansion.expansion_id not in current.expansion_ids
    # A non-blocking gap captures the redundancy so the AI can see why.
    assert any(
        gap.kind is GapKind.REDUNDANT_ALTERNATIVE and gap.target == "n_0001"
        and gap.blocking is False
        for gap in state.gaps
    )
    # is_redundant_alternative reports the same conclusion from a query API.
    is_redundant, reason = state.is_redundant_alternative(new_expansion.expansion_id)
    assert is_redundant
    assert reason is not None

def test_familiar_domain_keywords_do_not_auto_complete_the_decomposition():
    # The question deliberately uses operational vocabulary.  The runtime
    # must not synthesise a fixed formula or default children.
    question = Question(
        question_id="q-production-volume",
        question="Estimate the daily production volume of a fabrication workshop.",
        target_subject="fabrication-workshop daily production volume",
        target_measure="orders per day",
        unit="order/day",
        time_basis="per weekday",
        scope=Scope(population="fulfillment centres", geography="United States"),
    )
    state = create_decomposition(question)

    # No children were auto-injected by keyword matching.
    assert len(state.nodes) == 1
    assert len(state.expansions) == 0
    assert state.node("n_0001").status is NodeStatus.OPEN
    # The frontier exposes the still-open root for an AI to act on.
    frontier_ids = {node.node_id for node in frontier(state)}
    assert frontier_ids == {"n_0001"}
