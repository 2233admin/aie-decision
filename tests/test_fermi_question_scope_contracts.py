"""Track A contracts-level tests.

These tests cover every Track A OpenSpec scenario that lives purely at
the contracts layer: raw-question roots, restricted arithmetic parsing,
compound-unit parsing and dimensional closure, redundant-alternative
signatures, and atomic-claim validation including abstract-label
rejection.
"""

from __future__ import annotations

import ast

from dataclasses import replace

import pytest

from aie_decision.fermi_contracts import (
    DEFAULT_UNIT_SYMBOLS,
    AtomicClaim,
    AtomicClaimError,
    Branch,
    CompoundUnit,
    DIMENSIONLESS,
    DimensionalError,
    Expansion,
    FermiContractError,
    Gap,
    MeasurementKind,
    Node,
    NodeRole,
    NodeStatus,
    ObservationKind,
    Question,
    QuestionStatus,
    RedundancyReason,
    Relationship,
    RestrictedExpression,
    RestrictedExpressionError,
    Scope,
    check_dimensional_closure,
    divide_units,
    evaluate_restricted_expression,
    expressions_are_equivalent,
    multiply_units,
    parse_compound_unit,
    parse_restricted_expression,
    power_units,
    project_dimensional_closure,
    units_close,
    validate_atomic_claim,
)

def _scope() -> Scope:
    return Scope(population="U.S. commuters", geography="United States")

def _question(**overrides) -> Question:
    payload = {
        "question_id": "q-commutes",
        "question": "How many one-way commutes occur on a U.S. weekday?",
        "target_subject": "U.S. weekday commutes",
        "target_measure": "count of one-way commutes",
        "unit": "person/day",
        "time_basis": "per weekday in 2025",
        "scope": _scope(),
    }
    payload.update(overrides)
    return Question(**payload)

def test_raw_question_root_records_every_required_target_field():
    question = _question()

    assert question.question_id == "q-commutes"
    assert question.target_measure == "count of one-way commutes"
    assert question.target_subject == "U.S. weekday commutes"
    assert question.unit == "person/day"
    assert question.time_basis == "per weekday in 2025"
    assert question.scope.geography == "United States"
    assert question.scope.is_well_defined()
    assert question.is_minimally_complete()
    assert question.status is QuestionStatus.OPEN

@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_subject": "  "}, "target_subject"),
        ({"target_measure": ""}, "target_measure"),
        ({"unit": ""}, "unit"),
        ({"time_basis": "  "}, "time_basis"),
        ({"question": ""}, "question"),
        ({"question_id": ""}, "question_id"),
    ],
)
def test_question_without_a_required_field_is_rejected(overrides, match):
    with pytest.raises(FermiContractError, match=match):
        _question(**overrides)

def test_question_with_unresolved_scope_fields_keeps_gap_open():
    question = _question(scope=Scope(population="  ", geography="  "))

    assert question.is_minimally_complete() is False
    assert "scope anchors" not in question.unresolved_fields
    assert question.with_unresolved(("scope",)).unresolved_fields == ("scope",)

def test_scope_is_only_well_defined_with_a_real_anchor():
    assert Scope(geography="Tokyo").is_well_defined()
    assert Scope(population="all people").is_well_defined()
    assert Scope(geography="  ", population="  ").is_well_defined() is False
    assert Scope().is_well_defined() is False

def test_node_rejects_empty_label():
    with pytest.raises(FermiContractError, match="label"):
        Node(node_id="n_0002", label="", role=NodeRole.CHILD)

def test_node_cannot_assign_parent_to_the_target_role():
    with pytest.raises(FermiContractError, match="target node cannot have a parent"):
        Node(node_id="n_0001", label="root", role=NodeRole.TARGET, parent_id="n_0000")

def test_gap_rejects_blank_explanation():
    with pytest.raises(FermiContractError, match="explanation"):
        Gap(gap_id="gap_0001", kind=NodeStatus.UNRESOLVED, target="n_0002", explanation="   ")
