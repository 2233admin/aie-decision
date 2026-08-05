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

def test_tolerance_rejects_negative_width() -> None:
    with pytest.raises(ValueError):
        Tolerance(acceptable_width=-1.0)

def test_tolerance_requires_reference_for_relative_width() -> None:
    with pytest.raises(ValueError):
        Tolerance(acceptable_relative_width=0.2)
    with pytest.raises(ValueError):
        Tolerance(acceptable_relative_width=0.2, reference_value=0.0)

def test_evaluate_necessity_rejects_bad_material_degradation() -> None:
    leaves = [_constant_leaf("a", 1.0)]
    summary, expression, model, _ = _build(leaves, sample_count=64, seed=40)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return None

    with pytest.raises(ValueError):
        evaluate_necessity(
            leaf=leaves[0],
            other_leaves=[],
            expression=expression,
            model=model,
            baseline=summary,
            material_degradation=-0.1,
            delete=_delete,
        )

def test_evaluate_saturation_rejects_bad_threshold() -> None:
    leaves = [_constant_leaf("a", 1.0)]
    summary, expression, model, _ = _build(leaves, sample_count=64, seed=41)

    def _refine(
        current: Sequence[LeafSpec], baseline: TargetSummary
    ) -> Optional[RefinementProposal]:
        return None

    with pytest.raises(ValueError):
        evaluate_saturation(
            leaves,
            expression,
            model,
            summary,
            material_improvement_threshold=-1.0,
            next_refinement=_refine,
        )
