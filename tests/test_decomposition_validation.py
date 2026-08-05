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

def test_register_node_requires_a_label():
    from aie_decision.decomposition_tree import register_node

    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="label"):
        register_node(state, label="   ", unit="person/day", scope=_scope())

def test_expansion_request_must_include_rationale():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="rationale"):
        expand_state(
            state,
            request=ExpansionRequest(
                target_node_id="n_0001",
                parent_unit="person/day",
                expression="_child_0",
                rationale="",
                child_specs=(ChildSpec(label="x", unit="person/day", scope=_scope()),),
            ),
        )

def test_propose_alternative_requires_a_prior_expansion():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="at least one existing"):
        propose_alternative(
            state,
            request=ExpansionRequest(
                target_node_id="n_0001",
                parent_unit="person/day",
                expression="_child_0",
                rationale="r",
                child_specs=(ChildSpec(label="x", unit="person/day", scope=_scope()),),
            ),
        )

def test_expanding_a_pruned_node_is_rejected():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = prune(state, node_id=state.expansions[0].child_node_ids[0], reason="not actionable")
    with pytest.raises(DecompositionError, match="pruned"):
        expand_state(
            state,
            request=ExpansionRequest(
                target_node_id=state.expansions[0].child_node_ids[0],
                parent_unit="person/day",
                expression="_child_0",
                rationale="r",
                child_specs=(ChildSpec(label="x", unit="person/day", scope=_scope()),),
            ),
        )

def test_node_helper_exposes_unknown_node_lookup_errors():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="unknown node_id"):
        state.node("n_9999")

def test_current_branch_projection_filters_to_the_active_lineage():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale="alt",
            child_specs=(
                ChildSpec(label="work trips", unit="person/day", scope=_scope()),
                ChildSpec(label="non-work trips", unit="person/day", scope=_scope()),
            ),
        ),
    )
    projected = current_branch_projection(state)
    branch = projected.current_branch()
    for expansion in projected.expansions:
        assert expansion.expansion_id in branch.expansion_ids

def test_canonical_dimension_key_for_dimensionless_unit():
    from aie_decision.decomposition_tree import _dimension_key_for

    assert _dimension_key_for(CompoundUnit()) == "dimensionless"
    assert "kg" in _dimension_key_for(CompoundUnit({"kg": 1}))
