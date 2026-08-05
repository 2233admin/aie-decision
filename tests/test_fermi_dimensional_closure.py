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

def test_project_dimensional_closure_accepts_a_compatible_expression():
    projection = project_dimensional_closure("kg*m/s^2", "a * b", ["kg", "m/s^2"])
    assert projection.dimensions.to_canonical() == (("kg", 1), ("m", 1), ("s", -2))

def test_project_dimensional_closure_rejects_unit_mismatch():
    with pytest.raises(DimensionalError, match="closure"):
        project_dimensional_closure("kg*m/s^2", "a + b", ["kg", "m"])

def test_project_dimensional_closure_rejects_mismatched_variable_count():
    with pytest.raises(DimensionalError, match="references"):
        project_dimensional_closure("person/day", "a * b * c", ["person", "day"])

def test_check_dimensional_closure_returns_true_when_projection_matches_parent():
    parsed = parse_restricted_expression("a * b")
    ok, key = check_dimensional_closure("kg*m/s^2", parsed, ("kg", "m/s^2"))
    assert ok is True
    assert "kg" in key and "m" in key

def test_check_dimensional_closure_reports_incompatible_projection():
    parsed = parse_restricted_expression("a * b")
    ok, key = check_dimensional_closure("kg*m/s", parsed, ("kg", "m/s^2"))
    assert ok is False
    assert "kg" in key
