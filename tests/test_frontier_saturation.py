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

def test_saturation_returns_saturated_when_no_useful_refinement() -> None:
    leaves = [_constant_leaf("a", 1.0), _fitted_leaf("b", 4.9, 5.0, 5.1)]
    expression = compile_expression("a * b")
    model = JointModel(sample_count=256, seed=20)
    summary = joint_sample(leaves, expression, model)

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
    assert saturation.saturated is True
    assert saturation.explored == ()

def test_saturation_reports_valuable_refinement_when_present() -> None:
    leaves = [_constant_leaf("a", 1.0), _fitted_leaf("loose", 1.0, 5.0, 9.0)]
    expression = compile_expression("a * loose")
    model = JointModel(sample_count=1024, seed=21)
    summary = joint_sample(leaves, expression, model)

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        return RefinementProposal(
            leaf_id="loose",
            proposed_leaves=[_fitted_leaf("loose", 4.5, 5.0, 5.5)],
            description="tighten_loose_leaf",
            expected_target_narrowing=None,
        )

    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        summary,
        material_improvement_threshold=0.01,
        next_refinement=_refine,
    )
    assert saturation.saturated is False
    assert saturation.explored
    assert saturation.explored[0].observed_narrowing > 0.0
    assert saturation.explored[0].over_material_threshold is True

def test_saturation_exhausts_when_callback_yields_only_marginal_gains() -> None:
    leaves = [_constant_leaf("a", 1.0), _fitted_leaf("narrow", 4.95, 5.0, 5.05)]
    expression = compile_expression("a * narrow")
    model = JointModel(sample_count=512, seed=22)
    summary = joint_sample(leaves, expression, model)

    iteration = {"count": 0}

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        iteration["count"] += 1
        if iteration["count"] > 4:
            return None
        return RefinementProposal(
            leaf_id="narrow",
            proposed_leaves=[_fitted_leaf("narrow", 4.95, 5.0, 5.05)],
            description="noop_refinement",
            expected_target_narrowing=0.0,
        )

    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        summary,
        material_improvement_threshold=1.0,
        next_refinement=_refine,
    )
    assert saturation.saturated is True
    assert all(not r.over_material_threshold for r in saturation.explored)

def test_saturation_records_callback_error_as_note() -> None:
    leaves = [_constant_leaf("a", 1.0)]
    expression = compile_expression("a")
    model = JointModel(sample_count=64, seed=23)
    summary = joint_sample(leaves, expression, model)

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        raise RuntimeError("simulated_failure")

    saturation = evaluate_saturation(
        leaves,
        expression,
        model,
        summary,
        material_improvement_threshold=0.0,
        next_refinement=_refine,
    )
    assert any("refinement_callback_error" in note for note in saturation.notes)

def test_saturation_catches_replacement_failure_without_crashing() -> None:
    leaves = [_constant_leaf("anchor", 1.0), _fitted_leaf("b", 1.0, 2.0, 3.0)]
    expression, model = compile_expression("anchor * b"), JointModel(sample_count=64, seed=80)
    proposal = lambda *_: RefinementProposal("b", [_constant_leaf("anchor", 0.5)], "collision", 10.0)
    result = evaluate_saturation(leaves, expression, model, joint_sample(leaves, expression, model), material_improvement_threshold=0.5, next_refinement=proposal)
    assert result.explored == ()
    assert any("refinement_replacement_failed" in note for note in result.notes)

@pytest.mark.parametrize("invalid_baseline", [True, False])
def test_saturation_invalid_width_never_claims_narrowing(invalid_baseline: bool) -> None:
    unknown = [_unknown_leaf("a", domain=(0.5, 1.5)), _constant_leaf("scale", 2.0)]
    fitted = [_fitted_leaf("a", 0.9, 1.0, 1.1), _constant_leaf("scale", 2.0)]
    expression, model = compile_expression("a * scale"), JointModel(sample_count=64, seed=82)
    leaves, replacement = (unknown, fitted) if invalid_baseline else (fitted, unknown)
    proposal = lambda *_: RefinementProposal("a", [replacement[0]], "replace", 0.5)
    result = evaluate_saturation(leaves, expression, model, joint_sample(leaves, expression, model), material_improvement_threshold=0.01, next_refinement=proposal)
    expected = "baseline_invalid" if invalid_baseline else "refinement_invalid_interval"
    assert any(expected in note for note in result.notes)
    assert all(item.observed_narrowing == 0.0 for item in result.explored)
