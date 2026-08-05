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

def _fitted_leaf(
    leaf_id: str, p05: float, p50: float, p95: float, family=DistributionFamily.NORMAL
) -> LeafSpec:
    return LeafSpec(
        leaf_id=leaf_id,
        marginal=QuantileFittedMarginal(
            p05=p05, p50=p50, p95=p95, family=family
        ),
    )

def test_negative_equicorrelation_rejects_too_large() -> None:
    """Negative equicorrelation with off-diagonal >= 1/(n-1) is not PD."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    # For n=2 uncertain leaves, limit = 1/(2-1) = 1.0; any negative
    # correlation >= 1.0 is impossible; use a value that triggers the
    # guard: correlation 0.6 would require 0.6 < 1.0 so it passes.
    # Use 0.6 for n=3 leaves: limit = 1/(3-1) = 0.5, so 0.6 >= 0.5 → reject.
    model = JointModel(
        dependence=DependenceCase.NEGATIVE,
        correlation=0.6,
        sample_count=256,
        seed=12,
    )
    # The validation happens before sampling; we need a third uncertain leaf
    # so the guard fires.
    leaves3 = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
        _fitted_leaf("c", 1.0, 2.0, 3.0),
    ]
    expression3 = compile_expression("a * b * c")
    with pytest.raises(ValueError, match="not positive definite"):
        joint_sample(leaves3, expression3, model)


def test_negative_equicorrelation_rejects_the_exact_singular_boundary() -> None:
    """Adversarial: equality at -1/(n-1) is singular, not merely risky."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 1.0, 2.0, 3.0),
        _fitted_leaf("c", 1.0, 2.0, 3.0),
    ]
    expression = compile_expression("a * b * c")
    model = JointModel(
        dependence=DependenceCase.NEGATIVE,
        correlation=0.5,
        sample_count=128,
        seed=71,
    )

    with pytest.raises(ValueError, match="not positive definite"):
        joint_sample(leaves, expression, model)

def test_negative_equicorrelation_accepts_valid() -> None:
    """Valid negative equicorrelation passes and narrows the interval."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    independent = joint_sample(
        leaves,
        expression,
        JointModel(dependence=DependenceCase.INDEPENDENT, sample_count=1024, seed=13),
    )
    negative = joint_sample(
        leaves,
        expression,
        JointModel(
            dependence=DependenceCase.NEGATIVE,
            correlation=0.5,  # valid: 0.5 < 1/(2-1) = 1.0
            sample_count=1024,
            seed=13,
        ),
    )
    assert negative.probability_interval_valid is True
    assert negative.width < independent.width

def test_joint_sample_deterministic_under_fixed_seed() -> None:
    """Two calls with the same seed produce identical 90% output."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    model = JointModel(sample_count=2048, seed=42)
    first = joint_sample(leaves, expression, model)
    second = joint_sample(leaves, expression, model)
    assert first.p05 == second.p05
    assert first.p50 == second.p50
    assert first.p95 == second.p95
    assert first.width == second.width
    assert first.probability_interval_valid is True

def test_valid_joint_model_quantiles_inside_cartesian_envelope() -> None:
    """Under valid joint assumptions, the 90% interval must lie strictly
    inside the Cartesian product of the marginal P05/P95 ranges."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    summary = joint_sample(leaves, expression, JointModel(sample_count=4096, seed=99))
    assert summary.probability_interval_valid is True
    # The middle 90% must be inside the Cartesian envelope.
    assert summary.p05 > 1.0 * 4.0
    assert summary.p95 < 3.0 * 6.0
    # Width must be strictly smaller than the Cartesian range.
    cartesian_width = (3.0 * 6.0) - (1.0 * 4.0)
    assert summary.width < cartesian_width
    assert summary.marginal_summary.startswith("a=")
