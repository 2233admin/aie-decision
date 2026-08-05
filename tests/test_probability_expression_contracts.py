"""Tests for the probability and joint-sampling module."""

from __future__ import annotations

import math

import pytest

from aie_decision.probability import (
    CalibrationLabel,
    CompiledExpression,
    ConstantMarginal,
    DependenceCase,
    DistributionFamily,
    ExpressionError,
    JointModel,
    LeafSpec,
    MarginalKind,
    QuantileFittedMarginal,
    UnknownMarginal,
    compile_expression,
    evaluate_compiled,
    joint_sample,
    marginal_kind,
    rank_width_reduction,
    reducible_uncertainty,
)

def _constant_leaf(leaf_id: str, value: float) -> LeafSpec:
    return LeafSpec(leaf_id=leaf_id, marginal=ConstantMarginal(value=value))

def _fitted_leaf(
    leaf_id: str, p05: float, p50: float, p95: float, family=DistributionFamily.NORMAL
) -> LeafSpec:
    return LeafSpec(
        leaf_id=leaf_id,
        marginal=QuantileFittedMarginal(
            p05=p05, p50=p50, p95=p95, family=family
        ),
    )

def _unknown_leaf(leaf_id: str, domain=(0.0, 1.0)) -> LeafSpec:
    return LeafSpec(
        leaf_id=leaf_id, marginal=UnknownMarginal(reason="no_evidence", domain=domain)
    )

def test_compile_expression_accepts_arithmetic() -> None:
    compiled = compile_expression("a * b + c / d - 1")
    assert compiled.variables == ("a", "b", "c", "d")
    assert evaluate_compiled(compiled, {"a": 2, "b": 3, "c": 10, "d": 2}) == 2 * 3 + 10 / 2 - 1

def test_compile_expression_supports_power_and_unary() -> None:
    compiled = compile_expression("-(a ** 2) + b")
    assert evaluate_compiled(compiled, {"a": 3, "b": 10}) == -9 + 10

def test_compile_expression_rejects_unknown_syntax() -> None:
    with pytest.raises(ExpressionError):
        compile_expression("import os")
    with pytest.raises(ExpressionError):
        compile_expression("a and b")
    with pytest.raises(ExpressionError):
        compile_expression("a > 0")

def test_compile_expression_requires_at_least_one_variable() -> None:
    with pytest.raises(ExpressionError):
        compile_expression("1 + 2")

def test_compile_expression_rejects_non_finite_constants() -> None:
    with pytest.raises(ExpressionError):
        compile_expression("float('inf') * a")

def test_compile_expression_rejects_non_integer_exponent() -> None:
    """Non-integer power exponents are rejected at compile time so the
    restricted expression never silently evaluates ``**`` with a float
    exponent that would be undefined behaviour on negative bases."""

    with pytest.raises(ExpressionError, match="integer"):
        compile_expression("a ** 1.5")
    with pytest.raises(ExpressionError, match="integer"):
        compile_expression("a ** b")

def test_evaluate_compiled_surfaces_overflow_as_expression_error() -> None:
    """Numeric overflow during evaluation must surface as ExpressionError
    so the frontier treats it as a meaningful evaluation failure rather
    than an unexpected exception."""

    compiled = compile_expression("a ** 1000")
    with pytest.raises(ExpressionError, match="overflow"):
        evaluate_compiled(compiled, {"a": 10.0})

def test_unknown_marginal_short_circuits_before_negative_equicorrelation_check() -> None:
    """When an unknown marginal is present, the summary must reach the
    scenario-bounds branch even if the declared negative equicorrelation
    would otherwise be invalid.  Without this precedence, an honest
    unknown-marginal gap is masked behind a confusing PD failure."""

    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 1.0, 2.0, 3.0),
        _unknown_leaf("mystery"),
    ]
    expression = compile_expression("a * b * mystery")
    # A correlation that would normally be rejected for n=3 leaves
    # (limit = 1/(3-1) = 0.5).  With the unknown marginal present the
    # validation must not run; the copula dimension is 2 anyway.
    model = JointModel(
        dependence=DependenceCase.NEGATIVE,
        correlation=0.6,
        sample_count=64,
        seed=200,
    )
    summary = joint_sample(leaves, expression, model)
    assert summary.probability_interval_valid is False
    assert summary.coverage_semantics == "scenario_bounds_only"
    assert summary.unknown_leaves == ("mystery",)

def test_copula_dimension_excludes_unknown_and_constant_leaves() -> None:
    """Negative equicorrelation is validated against the copula dimension
    only — constant and unknown leaves must not inflate the PD bound."""

    # Two fitted leaves + a constant.  Copula dimension is 2, so any
    # correlation < 1.0 is PD.  With the OLD count (3 non-constant
    # leaves) the bound would be 0.5, but constants don't matter.
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 1.0, 2.0, 3.0),
        _constant_leaf("scale", 1.0),
    ]
    expression = compile_expression("a * b * scale")
    model = JointModel(
        dependence=DependenceCase.NEGATIVE,
        correlation=0.6,  # valid for n=2 copula leaves
        sample_count=64,
        seed=201,
    )
    summary = joint_sample(leaves, expression, model)
    assert summary.probability_interval_valid is True

def test_evaluate_compiled_detects_missing_variable() -> None:
    compiled = compile_expression("a + b")
    with pytest.raises(ExpressionError):
        evaluate_compiled(compiled, {"a": 1})

def test_evaluate_compiled_detects_division_by_zero() -> None:
    compiled = compile_expression("a / b")
    with pytest.raises(ExpressionError):
        evaluate_compiled(compiled, {"a": 1, "b": 0})
