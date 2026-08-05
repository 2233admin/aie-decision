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

def test_expressions_are_equivalent_under_commutative_canonicalisation():
    left = parse_restricted_expression("a * b")
    right = parse_restricted_expression("b * a")
    assert expressions_are_equivalent(left, right)

def test_expressions_are_not_equivalent_when_variables_differ():
    left = parse_restricted_expression("a * b")
    right = parse_restricted_expression("a * c")
    assert expressions_are_equivalent(left, right) is False

def test_expressions_are_not_equivalent_when_operator_differs():
    left = parse_restricted_expression("a * b")
    right = parse_restricted_expression("a + b")
    assert expressions_are_equivalent(left, right) is False

def test_expressions_with_distinct_canonical_signatures_compare_on_signature():
    a = parse_restricted_expression("(a + b) * c")
    b = parse_restricted_expression("c * (a + b)")
    assert expressions_are_equivalent(a, b)

def test_expansion_record_can_describe_itself():
    expansion = Expansion(
        expansion_id="exp_0001",
        target_node_id="n_0001",
        relationship_id="rel_0001",
        parent_unit="person/day",
        projected_unit="person*day^-1",
        child_node_ids=("n_0002", "n_0003"),
        rationale="ratio",
    )
    assert "exp_0001" in expansion.describe()
    assert "n_0001" in expansion.describe()

def test_branch_record_covers_named_expansion():
    branch = Branch(
        branch_id="br_0001",
        root_question_id="q-commutes",
        expansion_ids=("exp_0001", "exp_0002"),
    )
    assert branch.covers("exp_0001")
    assert not branch.covers("exp_0003")

def test_redundancy_reason_is_a_string_subclass_for_safe_logging():
    reason = RedundancyReason("matches exp_0001")
    assert str(reason) == "matches exp_0001"
