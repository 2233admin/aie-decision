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

def test_reducible_uncertainty_flags_dominant_leaf() -> None:
    leaves = [
        _fitted_leaf("tight", 9.9, 10.0, 10.1),
        _fitted_leaf("loose", 1.0, 5.0, 9.0),
    ]
    expression = compile_expression("tight * loose")
    model = JointModel(sample_count=1024, seed=12)
    summary = joint_sample(leaves, expression, model)
    contributions = reducible_uncertainty(leaves, expression, model, summary)
    assert [c.leaf_id for c in contributions] == ["loose", "tight"]
    loose_share = contributions[0].narrowing_fraction
    tight_share = contributions[1].narrowing_fraction
    assert loose_share > tight_share

def test_rank_width_reduction_is_deterministic() -> None:
    leaves = [
        _fitted_leaf("tight", 9.9, 10.0, 10.1),
        _fitted_leaf("loose", 1.0, 5.0, 9.0),
    ]
    expression = compile_expression("tight * loose")
    model = JointModel(sample_count=512, seed=1)
    summary = joint_sample(leaves, expression, model)
    contributions = reducible_uncertainty(leaves, expression, model, summary)
    ranked = rank_width_reduction(contributions)
    assert [r.leaf_id for r in ranked] == ["loose", "tight"]
    assert all(r.priority <= 1.0 + 1e-9 for r in ranked)
    assert ranked[0].priority >= ranked[-1].priority - 1e-9

def test_reducible_uncertainty_handles_only_constants() -> None:
    leaves = [_constant_leaf("a", 1.0), _constant_leaf("b", 2.0)]
    expression = compile_expression("a + b")
    model = JointModel()
    summary = joint_sample(leaves, expression, model)
    contributions = reducible_uncertainty(leaves, expression, model, summary)
    assert contributions == ()

def test_joint_sample_records_sample_metadata() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0)]
    expression = compile_expression("a")
    summary = joint_sample(
        leaves,
        expression,
        JointModel(sample_count=1024, seed=2025, dependence=DependenceCase.POSITIVE),
    )
    assert summary.sample_count == 1024
    assert summary.seed == 2025
    assert summary.method == "monte_carlo_joint_sampling"
    assert summary.dependence is DependenceCase.POSITIVE
