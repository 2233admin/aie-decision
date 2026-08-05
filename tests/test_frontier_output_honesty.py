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

def test_certify_unknown_marginal_frontier_not_certified() -> None:
    """Frontier with an unknown marginal cannot be certified because
    probability_interval_valid is False."""
    leaves = [
        _constant_leaf("scale", 2.0),
        _unknown_leaf("uncertain", domain=(0.5, 1.5)),
    ]
    expression = compile_expression("scale * uncertain")
    model = JointModel(sample_count=512, seed=50)
    summary = joint_sample(leaves, expression, model)
    assert summary.probability_interval_valid is False

    tolerance = Tolerance(acceptable_width=10.0)
    sufficiency = evaluate_sufficiency(summary, tolerance)
    # Sufficiency must be NOT_ASSESSABLE, not PASSES.
    assert sufficiency.status is SufficiencyStatus.NOT_ASSESSABLE
    assert "probability_interval_invalid" in sufficiency.reasons

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        return None

    necessity = (
        evaluate_necessity(
            leaf=leaves[0],
            other_leaves=[leaves[1]],
            expression=expression,
            model=model,
            baseline=summary,
            material_degradation=0.5,
            delete=_delete,
        ),
    )
    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        summary,
        material_improvement_threshold=0.5,
        next_refinement=_refine,
    )

    certification = certify_frontier(summary, sufficiency, necessity, saturation)
    # Certification must fail at the sufficiency gate.
    assert certification.status is FrontierStatus.INSUFFICIENT
    assert certification.certified is False

def test_certify_unknown_dependence_frontier_not_certified() -> None:
    """Frontier with UNKNOWN dependence and two uncertain leaves cannot be
    certified because probability_interval_valid is False."""
    leaves = [
        _fitted_leaf("a", 1.0, 2.0, 3.0),
        _fitted_leaf("b", 4.0, 5.0, 6.0),
    ]
    expression = compile_expression("a * b")
    model = JointModel(
        dependence=DependenceCase.UNKNOWN,
        sample_count=512,
        seed=51,
    )
    summary = joint_sample(leaves, expression, model)
    assert summary.probability_interval_valid is False
    assert summary.coverage_semantics == "invalid_unknown_dependence"

    tolerance = Tolerance(acceptable_width=100.0)
    sufficiency = evaluate_sufficiency(summary, tolerance)
    assert sufficiency.status is SufficiencyStatus.NOT_ASSESSABLE

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        return None

    necessity = (
        evaluate_necessity(
            leaf=leaves[0],
            other_leaves=[leaves[1]],
            expression=expression,
            model=model,
            baseline=summary,
            material_degradation=0.5,
            delete=_delete,
        ),
    )
    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        summary,
        material_improvement_threshold=0.5,
        next_refinement=_refine,
    )

    certification = certify_frontier(summary, sufficiency, necessity, saturation)
    assert certification.status is FrontierStatus.INSUFFICIENT
    assert certification.certified is False

def test_certify_valid_joint_model_produces_deterministic_certified_output() -> None:
    """Under valid joint assumptions with a tight tolerance, the frontier
    can be certified and two calls with the same seed produce identical results."""
    leaves = [
        _constant_leaf("anchor", 1.0),
        _fitted_leaf("narrow", 4.95, 5.0, 5.05),
    ]
    expression = compile_expression("anchor * narrow")
    model = JointModel(sample_count=2048, seed=52)

    first = joint_sample(leaves, expression, model)
    second = joint_sample(leaves, expression, model)
    assert first.p05 == second.p05
    assert first.p95 == second.p95
    assert first.width == second.width
    assert first.probability_interval_valid is True
    assert second.probability_interval_valid is True

    tolerance = Tolerance(acceptable_width=first.width + 1e-9)
    sufficiency = evaluate_sufficiency(first, tolerance)
    assert sufficiency.status is SufficiencyStatus.PASSES

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        return None

    necessity_anchor = evaluate_necessity(
        leaf=leaves[0],
        other_leaves=[leaves[1]],
        expression=expression,
        model=model,
        baseline=first,
        material_degradation=0.5,
        delete=_delete,
    )
    necessity_narrow = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=first,
        material_degradation=0.5,
        delete=_delete,
    )
    # Force both necessary.
    necessity_anchor = type(necessity_anchor)(
        leaf_id=necessity_anchor.leaf_id,
        is_necessary=True,
        deletion=necessity_anchor.deletion,
        coarsening=necessity_anchor.coarsening,
        substitution=necessity_anchor.substitution,
        material_threshold=necessity_anchor.material_threshold,
        baseline_width=necessity_anchor.baseline_width,
        reasons=("forced",),
    )
    necessity_narrow = type(necessity_narrow)(
        leaf_id=necessity_narrow.leaf_id,
        is_necessary=True,
        deletion=necessity_narrow.deletion,
        coarsening=necessity_narrow.coarsening,
        substitution=necessity_narrow.substitution,
        material_threshold=necessity_narrow.material_threshold,
        baseline_width=necessity_narrow.baseline_width,
        reasons=("forced",),
    )

    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        first,
        material_improvement_threshold=0.5,
        next_refinement=_refine,
    )

    certification = certify_frontier(
        first,
        sufficiency,
        (necessity_anchor, necessity_narrow),
        saturation,
    )
    assert certification.status is FrontierStatus.CERTIFIED
    assert certification.certified is True
