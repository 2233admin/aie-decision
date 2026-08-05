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

def _build_passing(
    sample_count: int = 1024, seed: int = 30
) -> tuple[
    TargetSummary,
    CompiledExpression,
    JointModel,
    list[LeafSpec],
    Tolerance,
    SufficiencyEvidence,
]:
    leaves = [_constant_leaf("anchor", 1.0), _fitted_leaf("narrow", 4.95, 5.0, 5.05)]
    expression = compile_expression("anchor * narrow")
    model = JointModel(sample_count=sample_count, seed=seed)
    summary = joint_sample(leaves, expression, model)
    tolerance = Tolerance(acceptable_width=summary.width * 10.0 + 1e-9)
    sufficiency = evaluate_sufficiency(summary, tolerance)
    assert sufficiency.status is SufficiencyStatus.PASSES
    return summary, expression, model, leaves, tolerance, sufficiency

def test_certify_frontier_certifies_when_all_three_gates_pass() -> None:
    summary, expression, model, leaves, tolerance, sufficiency = _build_passing()

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    # anchor: deletion destroys the branch because expression requires it
    # narrow: deletion collapses width to 0 → within tolerance → redundant
    necessity_anchor = evaluate_necessity(
        leaf=leaves[0],
        other_leaves=[leaves[1]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.5,
        delete=_delete,
    )
    necessity_narrow = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.5,
        delete=_delete,
    )
    # We want a clean "all necessary" picture. Force the narrow leaf to be
    # necessary by tightening the threshold.
    necessity_narrow_strict = NecessityEvidence(
        leaf_id=necessity_narrow.leaf_id,
        is_necessary=True,
        deletion=necessity_narrow.deletion,
        coarsening=necessity_narrow.coarsening,
        substitution=necessity_narrow.substitution,
        material_threshold=0.0001,
        baseline_width=necessity_narrow.baseline_width,
        reasons=("forced_for_certification",),
    )

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        return None

    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        summary,
        material_improvement_threshold=0.5,
        next_refinement=_refine,
    )

    certification = certify_frontier(
        summary,
        sufficiency,
        (necessity_anchor, necessity_narrow_strict),
        saturation,
    )
    assert certification.status is FrontierStatus.CERTIFIED
    assert certification.certified is True
    assert certification.reasons == ()

def test_certify_frontier_reports_insufficient_when_sufficiency_fails() -> None:
    leaves = [_fitted_leaf("a", 1.0, 2.0, 4.0)]
    expression = compile_expression("a")
    model = JointModel(sample_count=512, seed=31)
    summary = joint_sample(leaves, expression, model)
    tolerance = Tolerance(acceptable_width=0.01)  # any reasonable width fails
    sufficiency = evaluate_sufficiency(summary, tolerance)
    assert sufficiency.status is SufficiencyStatus.FAILS

    certification = certify_frontier(
        summary,
        sufficiency,
        (),
        SaturationEvidence(
            saturated=True,
            material_threshold=0.0,
            explored=(),
            best_observed_narrowing=0.0,
        ),
    )
    assert certification.status is FrontierStatus.INSUFFICIENT
    assert any("sufficiency" in r for r in certification.reasons)

def test_certify_frontier_reports_structurally_complete_when_redundant_leaf() -> None:
    leaves = [_constant_leaf("a", 1.0), _constant_leaf("b", 1.0)]
    expression = compile_expression("a")
    model = JointModel(sample_count=64, seed=32)
    summary = joint_sample(leaves, expression, model)
    tolerance = Tolerance(acceptable_width=summary.width + 1e-9)
    sufficiency = evaluate_sufficiency(summary, tolerance)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        if leaf_id == "a":
            return None
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    necessity_a = evaluate_necessity(
        leaf=leaves[0],
        other_leaves=[leaves[1]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.0,
        delete=_delete,
    )
    necessity_b = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.0,
        delete=_delete,
    )
    assert necessity_a.is_necessary is True
    assert necessity_b.is_necessary is False  # deletion leaves branch intact

    saturation = SaturationEvidence(
        saturated=True,
        material_threshold=0.0,
        explored=(),
        best_observed_narrowing=0.0,
    )
    certification = certify_frontier(
        summary,
        sufficiency,
        (necessity_a, necessity_b),
        saturation,
    )
    assert certification.status is FrontierStatus.STRUCTURALLY_COMPLETE
    assert not certification.certified
    assert any("necessity_redundant:b" in r for r in certification.reasons)

def test_certify_frontier_reports_structurally_complete_when_not_saturated() -> None:
    summary, expression, model, leaves, tolerance, sufficiency = _build_passing()

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    necessity_a = evaluate_necessity(
        leaf=leaves[0],
        other_leaves=[leaves[1]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.5,
        delete=_delete,
    )
    necessity_b = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.5,
        delete=_delete,
    )
    # Force both leaves to be necessary.
    forced = []
    for evidence in (necessity_a, necessity_b):
        forced.append(
            NecessityEvidence(
                leaf_id=evidence.leaf_id,
                is_necessary=True,
                deletion=evidence.deletion,
                coarsening=evidence.coarsening,
                substitution=evidence.substitution,
                material_threshold=evidence.material_threshold,
                baseline_width=evidence.baseline_width,
                reasons=evidence.reasons,
            )
        )

    saturation = SaturationEvidence(
        saturated=False,
        material_threshold=0.5,
        explored=(
            RefinementResult(
                leaf_id="narrow",
                description="expected_to_help",
                expected_narrowing=0.6,
                observed_narrowing=0.6,
                over_material_threshold=True,
            ),
        ),
        best_observed_narrowing=0.6,
    )
    certification = certify_frontier(
        summary,
        sufficiency,
        tuple(forced),
        saturation,
    )
    assert certification.status is FrontierStatus.STRUCTURALLY_COMPLETE
    assert any("saturation" in r for r in certification.reasons)
