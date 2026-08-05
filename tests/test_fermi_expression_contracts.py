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

def ast_dump(tree):
    return ast.dump(tree)

@pytest.mark.parametrize(
    "expression",
    [
        "a + b",
        "a - b",
        "a * b",
        "a / b",
        "a * b + c",
        "a * (b + c)",
        "a^2 + b^2",
        "-a + b",
        "+1.5 * a",
        "(a + b) * (c - d)",
    ],
)
def test_parse_restricted_expression_accepts_pure_arithmetic(expression):
    parsed = parse_restricted_expression(expression)
    assert isinstance(parsed, RestrictedExpression)
    assert parsed.variables
    assert parsed.signature

@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "foo()",
        "lambda x: x",
        "[a for a in b]",
        "a if a > 0 else b",
        "a and b",
        "a == b",
        "a[0]",
        "a.b",
        "True",
        "None",
    ],
)
def test_parse_restricted_expression_rejects_anything_beyond_arithmetic(expression):
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression(expression)

def test_parse_restricted_expression_accepts_caret_power_notation():
    parsed = parse_restricted_expression("a ^ 2")
    assert "Pow" in ast_dump(parsed.tree)

def test_parse_restricted_expression_rejects_dunder_names():
    with pytest.raises(RestrictedExpressionError, match="dunder"):
        parse_restricted_expression("__class__")

def test_parse_restricted_expression_rejects_non_numeric_or_non_finite_constants():
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression("a + 'b'")
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression("a + True")
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression("a + 1e400 * 1e400")

def test_parse_restricted_expression_records_variable_count():
    parsed = parse_restricted_expression("a * b * c / d")
    assert sorted(parsed.variables) == ["a", "b", "c", "d"]

def test_parse_restricted_expression_variables_follow_source_order():
    # ``ast.walk`` visits in BFS order and would yield ``b`` before ``a``
    # when the variable is nested inside a sub-expression on the right;
    # the project must collect variables in textual source order so that
    # positional unit bindings line up with the relationship's child list.
    parsed = parse_restricted_expression("b * a")
    assert list(parsed.variables) == ["b", "a"]

def test_parse_restricted_expression_dimension_evaluation_requires_unit_bindings():
    parsed = parse_restricted_expression(
        "a * b",
        variable_units={
            "a": parse_compound_unit("kg"),
            "b": parse_compound_unit("m/s^2"),
        },
    )
    assert parsed.dimensions.to_canonical() == (("kg", 1), ("m", 1), ("s", -2))

def test_parse_restricted_expression_rejects_undeclared_variables_when_units_pinned():
    with pytest.raises(RestrictedExpressionError, match="undeclared"):
        parse_restricted_expression("a * b", variable_units={"a": DIMENSIONLESS})

def test_evaluate_restricted_expression_returns_numeric_value():
    parsed = parse_restricted_expression("a * b + c")
    assert evaluate_restricted_expression(parsed, {"a": 2.0, "b": 3.0, "c": 4.0}) == 10.0

def test_evaluate_restricted_expression_rejects_division_by_zero():
    parsed = parse_restricted_expression("a / b")
    with pytest.raises(RestrictedExpressionError, match="division by zero"):
        evaluate_restricted_expression(parsed, {"a": 1.0, "b": 0.0})

def test_evaluate_restricted_expression_rejects_non_numeric_value():
    parsed = parse_restricted_expression("a + 1")
    with pytest.raises(RestrictedExpressionError, match="non-numeric"):
        evaluate_restricted_expression(parsed, {"a": "x"})

def test_evaluate_restricted_expression_rejects_non_integer_exponent():
    parsed = parse_restricted_expression("a ^ 1.5")
    with pytest.raises(RestrictedExpressionError, match="integer constant"):
        evaluate_restricted_expression(parsed, {"a": 2.0})
