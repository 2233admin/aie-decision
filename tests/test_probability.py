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


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------


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


def test_evaluate_compiled_detects_missing_variable() -> None:
    compiled = compile_expression("a + b")
    with pytest.raises(ExpressionError):
        evaluate_compiled(compiled, {"a": 1})


def test_evaluate_compiled_detects_division_by_zero() -> None:
    compiled = compile_expression("a / b")
    with pytest.raises(ExpressionError):
        evaluate_compiled(compiled, {"a": 1, "b": 0})


# ---------------------------------------------------------------------------
# Marginal constructors
# ---------------------------------------------------------------------------


def test_constant_marginal_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        ConstantMarginal(value=float("inf"))


def test_quantile_fitted_marginal_enforces_order() -> None:
    with pytest.raises(ValueError):
        QuantileFittedMarginal(p05=5, p50=1, p95=10)
    with pytest.raises(ValueError):
        QuantileFittedMarginal(p05=1, p50=2, p95=2)


def test_unknown_marginal_requires_reason_and_ordered_domain() -> None:
    with pytest.raises(ValueError):
        UnknownMarginal(reason="", domain=(0.0, 1.0))
    with pytest.raises(ValueError):
        UnknownMarginal(reason="missing", domain=(1.0, 0.0))
    with pytest.raises(ValueError):
        UnknownMarginal(reason="missing", domain=(0.0, 0.0))


def test_marginal_kind_dispatch() -> None:
    assert marginal_kind(ConstantMarginal(value=1)) is MarginalKind.CONSTANT
    assert (
        marginal_kind(QuantileFittedMarginal(p05=1, p50=2, p95=3))
        is MarginalKind.QUANTILE_FITTED
    )
    assert (
        marginal_kind(UnknownMarginal(reason="gap", domain=(0, 1)))
        is MarginalKind.UNKNOWN
    )


# ---------------------------------------------------------------------------
# Joint sampling — basic properties
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reducible uncertainty and ranking
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unknown marginal — scenario-only, cannot certify 90% probability interval
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unknown dependence — invalid when joint propagation is required
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Negative equicorrelation validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Deterministic 90% output under valid joint assumptions
# ---------------------------------------------------------------------------


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