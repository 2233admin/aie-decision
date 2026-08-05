"""Tests for the frontier certification module."""

from __future__ import annotations

from typing import Optional, Sequence

import pytest

from aie_decision.frontier import (
    CoarseningCallback,
    DeletionCallback,
    FrontierStatus,
    InterventionResult,
    NecessityEvidence,
    RefinementCallback,
    RefinementProposal,
    RefinementResult,
    SaturationEvidence,
    SufficiencyEvidence,
    SufficiencyStatus,
    SubstitutionCallback,
    Tolerance,
    certify_frontier,
    evaluate_necessity,
    evaluate_saturation,
    evaluate_sufficiency,
)

from aie_decision.probability import (
    CalibrationLabel,
    CompiledExpression,
    ConstantMarginal,
    DependenceCase,
    DistributionFamily,
    JointModel,
    LeafSpec,
    QuantileFittedMarginal,
    TargetSummary,
    UnknownMarginal,
    compile_expression,
    joint_sample,
)

def _constant_leaf(leaf_id: str, value: float) -> LeafSpec:
    return LeafSpec(leaf_id=leaf_id, marginal=ConstantMarginal(value=value))

def _fitted_leaf(
    leaf_id: str, p05: float, p50: float, p95: float
) -> LeafSpec:
    return LeafSpec(
        leaf_id=leaf_id,
        marginal=QuantileFittedMarginal(
            p05=p05, p50=p50, p95=p95, family=DistributionFamily.NORMAL
        ),
    )

def _unknown_leaf(leaf_id: str, domain=(0.0, 1.0)) -> LeafSpec:
    return LeafSpec(
        leaf_id=leaf_id,
        marginal=UnknownMarginal(reason="no_evidence", domain=domain),
    )

def _build(
    leaves: Sequence[LeafSpec],
    expression_source: str | None = None,
    sample_count: int = 512,
    seed: int = 0,
    dependence: DependenceCase = DependenceCase.INDEPENDENT,
) -> tuple[TargetSummary, CompiledExpression, JointModel, list[LeafSpec]]:
    if expression_source is None:
        ids = [leaf.leaf_id for leaf in leaves]
        if len(ids) == 1:
            expression_source = ids[0]
        else:
            expression_source = " * ".join(ids)
    expression = compile_expression(expression_source)
    model = JointModel(
        sample_count=sample_count, seed=seed, dependence=dependence
    )
    summary = joint_sample(leaves, expression, model)
    return summary, expression, model, list(leaves)

def test_sufficiency_passes_when_width_below_tolerance() -> None:
    summary, _, _, _ = _build(
        [_fitted_leaf("a", 9.9, 10.0, 10.1), _constant_leaf("b", 1.0)],
        expression_source="a * b",
        sample_count=1024,
        seed=1,
    )
    tolerance = Tolerance(acceptable_width=summary.width + 1e-9)
    evidence = evaluate_sufficiency(summary, tolerance)
    assert evidence.status is SufficiencyStatus.PASSES
    assert evidence.passes is True

def test_sufficiency_fails_when_width_exceeds_tolerance() -> None:
    summary, _, _, _ = _build(
        [_fitted_leaf("a", 1.0, 5.0, 9.0), _constant_leaf("b", 1.0)],
        expression_source="a * b",
        sample_count=1024,
        seed=2,
    )
    tolerance = Tolerance(acceptable_width=max(summary.width * 0.5, 1e-12))
    evidence = evaluate_sufficiency(summary, tolerance)
    assert evidence.status is SufficiencyStatus.FAILS
    assert not evidence.passes
    assert evidence.reasons

def test_sufficiency_relative_width_uses_reference_value() -> None:
    summary, _, _, _ = _build(
        [_fitted_leaf("a", 90.0, 100.0, 110.0)],
        expression_source="a",
        sample_count=1024,
        seed=3,
    )
    tolerance = Tolerance(acceptable_relative_width=0.2, reference_value=100.0)
    evidence = evaluate_sufficiency(summary, tolerance)
    assert evidence.relative_width is not None
    # Either PASSES or FAILS depending on realised Monte Carlo width; the
    # contract is just that we exposed a measured relative width.
    assert evidence.status in {SufficiencyStatus.PASSES, SufficiencyStatus.FAILS}

def test_sufficiency_decision_thresholds_fail_on_span() -> None:
    summary, _, _, _ = _build(
        [_fitted_leaf("a", -5.0, 5.0, 15.0)],
        expression_source="a",
        sample_count=2048,
        seed=4,
    )
    tolerance = Tolerance(decision_thresholds=(2.0, 8.0))
    evidence = evaluate_sufficiency(summary, tolerance)
    assert evidence.status is SufficiencyStatus.FAILS
    assert any("spans" in reason for reason in evidence.reasons)

def test_sufficiency_returns_not_assessable_without_tolerance() -> None:
    summary, _, _, _ = _build([_constant_leaf("a", 1.0)])
    tolerance = Tolerance()
    evidence = evaluate_sufficiency(summary, tolerance)
    assert evidence.status is SufficiencyStatus.NOT_ASSESSABLE
    assert evidence.reasons == ("no_explicit_tolerance",)

def test_sufficiency_require_calibration_blocks_unknown_semantics() -> None:
    # An unknown marginal makes probability_interval_valid=False, so
    # sufficiency is NOT_ASSESSABLE regardless of require_calibration.
    summary, _, _, _ = _build(
        [_unknown_leaf("a"), _constant_leaf("b", 1.0)],
        expression_source="a * b",
        sample_count=256,
        seed=5,
    )
    assert summary.probability_interval_valid is False
    tolerance = Tolerance(
        acceptable_width=10.0,
        require_calibration=True,
    )
    evidence = evaluate_sufficiency(summary, tolerance)
    # With invalid probability interval, NOT_ASSESSABLE takes precedence
    # over require_calibration check.
    assert evidence.status is SufficiencyStatus.NOT_ASSESSABLE
    assert "probability_interval_invalid" in evidence.reasons
