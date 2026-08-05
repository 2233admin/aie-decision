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

def test_joint_sample_constant_product_is_exact() -> None:
    leaves = [_constant_leaf("price", 10.0), _constant_leaf("qty", 5.0)]
    expression = compile_expression("price * qty")
    summary = joint_sample(leaves, expression, JointModel())
    assert summary.p05 == summary.p50 == summary.p95 == 50.0
    assert summary.width == 0.0

def test_joint_sample_reproducible_under_seed() -> None:
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    first = joint_sample(leaves, expression, JointModel(sample_count=512, seed=42))
    second = joint_sample(leaves, expression, JointModel(sample_count=512, seed=42))
    assert first.p05 == second.p05
    assert first.p50 == second.p50
    assert first.p95 == second.p95
    assert first.width == second.width

def test_joint_sample_different_seeds_diverge() -> None:
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    a = joint_sample(leaves, expression, JointModel(sample_count=512, seed=1))
    b = joint_sample(leaves, expression, JointModel(sample_count=512, seed=2))
    assert a.p05 != b.p05 or a.p95 != b.p95

def test_joint_sample_unknown_marginal_records_gap_and_calibration() -> None:
    leaves = [
        _constant_leaf("scale", 2.0),
        _unknown_leaf("uncertain"),
    ]
    expression = compile_expression("scale * uncertain")
    summary = joint_sample(
        leaves, expression, JointModel(sample_count=512, seed=7)
    )
    assert summary.calibration is CalibrationLabel.UNMEASURED_UNKNOWN_MARGINAL
    assert summary.unknown_leaves == ("uncertain",)
    assert any(g.startswith("unknown_marginal_semantics") for g in summary.dependency_gaps)

def test_joint_sample_unknown_dependence_records_dependency_gap() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0), _fitted_leaf("b", 4.0, 5.0, 6.0)]
    expression = compile_expression("a + b")
    summary = joint_sample(
        leaves, expression, JointModel(dependence=DependenceCase.UNKNOWN, sample_count=512, seed=11)
    )
    assert summary.calibration is CalibrationLabel.UNMEASURED_WITH_DEPENDENCY_GAP
    assert "leaf_dependence_undeclared" in summary.dependency_gaps

def test_joint_sample_does_not_treat_cartesian_endpoints_as_target_interval() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0), _fitted_leaf("b", 4.0, 5.0, 6.0)]
    expression = compile_expression("a * b")
    summary = joint_sample(leaves, expression, JointModel(sample_count=4096, seed=99))
    # Cartesian range would be [4, 18] for product of fitted leaves.
    cartesian_width = (3 * 6) - (1 * 4)
    # The Monte Carlo summary must be much narrower than the Cartesian endpoint range.
    assert summary.width < cartesian_width
    # The middle 90% interval must lie strictly inside the Cartesian envelope.
    assert summary.p05 > 1 * 4
    assert summary.p95 < 3 * 6

def test_joint_sample_positive_dependence_widens_product() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0), _fitted_leaf("b", 4.0, 5.0, 6.0)]
    expression = compile_expression("a * b")
    independent = joint_sample(
        leaves, expression, JointModel(dependence=DependenceCase.INDEPENDENT, sample_count=2048, seed=3)
    )
    positive = joint_sample(
        leaves, expression, JointModel(dependence=DependenceCase.POSITIVE, sample_count=2048, seed=3)
    )
    # With strong positive copula, low pairs with low and high pairs with high,
    # so the product's 90% interval must widen past the independent baseline.
    assert positive.width > independent.width

def test_joint_sample_negative_dependence_narrows_product() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0), _fitted_leaf("b", 4.0, 5.0, 6.0)]
    expression = compile_expression("a * b")
    independent = joint_sample(
        leaves, expression, JointModel(dependence=DependenceCase.INDEPENDENT, sample_count=2048, seed=5)
    )
    negative = joint_sample(
        leaves, expression, JointModel(dependence=DependenceCase.NEGATIVE, sample_count=2048, seed=5)
    )
    # Negative dependence pairs low with high and vice versa, dampening the
    # extremes of the product so its 90% interval must narrow.
    assert negative.width < independent.width

def test_joint_sample_rejects_expression_referencing_unknown_leaf() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0)]
    expression = compile_expression("a + missing")
    with pytest.raises(ValueError):
        joint_sample(leaves, expression, JointModel())

def test_joint_sample_requires_at_least_one_leaf() -> None:
    expression = compile_expression("a + 1")
    with pytest.raises(ValueError):
        joint_sample([], expression, JointModel())

def test_joint_sample_lognormal_requires_positive_quantiles() -> None:
    leaves = [_fitted_leaf("a", 0.1, 1.0, 5.0, family=DistributionFamily.LOGNORMAL)]
    expression = compile_expression("a")
    summary = joint_sample(leaves, expression, JointModel(sample_count=256, seed=4))
    assert summary.p50 == pytest.approx(1.0, abs=0.5)
