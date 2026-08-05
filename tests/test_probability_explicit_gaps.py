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

def test_unknown_marginal_probability_interval_valid_is_false() -> None:
    """Unknown marginal must produce probability_interval_valid=False."""
    leaves = [
        _constant_leaf("scale", 2.0),
        _unknown_leaf("uncertain", domain=(0.5, 1.5)),
    ]
    expression = compile_expression("scale * uncertain")
    summary = joint_sample(leaves, expression, JointModel(sample_count=512, seed=7))
    assert summary.probability_interval_valid is False
    # Quantiles must be None, not silently filled with scenario endpoints.
    assert summary.p05 is None
    assert summary.p50 is None
    assert summary.p95 is None
    assert summary.width is None
    # Scenario envelope is exposed for sensitivity analysis.
    assert summary.scenario_bounds is not None
    assert summary.scenario_bounds[0] <= summary.scenario_bounds[1]
    assert summary.coverage_semantics == "scenario_bounds_only"

def test_unknown_marginal_scenario_bounds_not_named_as_p05_p95() -> None:
    """Scenario endpoints must not be stored under p05/p95 fields."""
    leaves = [
        _constant_leaf("scale", 2.0),
        _unknown_leaf("uncertain", domain=(0.0, 10.0)),
    ]
    expression = compile_expression("scale * uncertain")
    summary = joint_sample(leaves, expression, JointModel(sample_count=512, seed=8))
    # p05/p95 are None — scenario numbers are NOT masquerading as quantiles.
    assert summary.p05 is None
    assert summary.p95 is None
    # Scenario envelope is available separately.
    assert summary.scenario_bounds == (0.0, 20.0)  # scale * domain extremes

def test_unknown_marginal_reducible_uncertainty_returns_empty() -> None:
    """When baseline has no valid probability interval, reducible uncertainty
    cannot be expressed as a fraction of a 90 percent width."""
    leaves = [
        _constant_leaf("scale", 2.0),
        _unknown_leaf("uncertain"),
    ]
    expression = compile_expression("scale * uncertain")
    model = JointModel(sample_count=512, seed=9)
    summary = joint_sample(leaves, expression, model)
    assert summary.probability_interval_valid is False
    contributions = reducible_uncertainty(leaves, expression, model, summary)
    assert contributions == ()

def test_unknown_dependence_with_two_uncertain_leaves_invalid() -> None:
    """UNKNOWN dependence with >=2 non-constant leaves makes the interval
    invalid regardless of how narrow the scenario envelope is."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    summary = joint_sample(
        leaves,
        expression,
        JointModel(
            dependence=DependenceCase.UNKNOWN,
            sample_count=512,
            seed=10,
        ),
    )
    assert summary.probability_interval_valid is False
    assert summary.p05 is None
    assert summary.p95 is None
    assert summary.coverage_semantics == "invalid_unknown_dependence"
    # Scenario envelope is still provided as a reference.
    assert summary.scenario_bounds is not None

def test_unknown_dependence_single_leaf_is_valid() -> None:
    """UNKNOWN dependence with only one non-constant leaf is still valid because
    joint propagation is not required."""
    leaves = [_fitted_leaf("a", 1.0, 2.0, 3.0)]
    expression = compile_expression("a")
    summary = joint_sample(
        leaves,
        expression,
        JointModel(
            dependence=DependenceCase.UNKNOWN,
            sample_count=512,
            seed=11,
        ),
    )
    assert summary.probability_interval_valid is True
    assert summary.p05 is not None
    assert summary.p95 is not None
    assert summary.coverage_semantics == "monte_carlo_joint_sampling"
