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

def test_necessity_deletion_marks_redundant_leaf_when_width_stays_similar() -> None:
    leaves = [
        _constant_leaf("anchor", 1.0),
        _fitted_leaf("narrow", 0.99, 1.0, 1.01),
    ]
    expression = compile_expression("anchor")
    model = JointModel(sample_count=512, seed=10)
    summary = joint_sample(leaves, expression, model)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        if leaf_id == "anchor":
            return None
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    evidence = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=10.0,
        delete=_delete,
    )
    assert evidence.is_necessary is False
    assert evidence.deletion.applicable is True
    assert any("within_material_tolerance" in reason for reason in evidence.reasons)

def test_necessity_deletion_marks_necessary_when_branch_breaks() -> None:
    leaves = [_constant_leaf("a", 1.0), _constant_leaf("b", 2.0)]
    expression = compile_expression("a + b")
    model = JointModel(sample_count=128, seed=11)
    summary = joint_sample(leaves, expression, model)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        if leaf_id == "b":
            return None
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    evidence = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.01,
        delete=_delete,
    )
    assert evidence.is_necessary is True
    assert evidence.deletion.new_status == "non_computable"

def test_necessity_destructive_degradation_is_recorded_as_necessary() -> None:
    leaves = [_constant_leaf("anchor", 100.0), _fitted_leaf("vol", 1.0, 5.0, 9.0)]
    expression = compile_expression("anchor * vol")
    model = JointModel(sample_count=512, seed=12)
    summary = joint_sample(leaves, expression, model)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    evidence = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.01,
        delete=_delete,
    )
    # The expression references vol, so deletion makes the branch
    # non-computable — this destructive degradation must be recorded as
    # necessary evidence.
    assert evidence.is_necessary is True
    assert any("deletion_makes_branch_non_computable" in reason for reason in evidence.reasons)

def test_necessity_coarsening_within_tolerance_marks_redundant() -> None:
    leaves = [_constant_leaf("anchor", 1.0), _fitted_leaf("narrow", 4.9, 5.0, 5.1)]
    expression = compile_expression("anchor * narrow")
    model = JointModel(sample_count=256, seed=13)
    summary = joint_sample(leaves, expression, model)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    def _coarsen(leaf_id: str) -> Optional[LeafSpec]:
        if leaf_id == "narrow":
            return _fitted_leaf("narrow", 4.0, 5.0, 6.0)
        return None

    evidence = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=10.0,  # tolerant
        delete=_delete,
        coarsen=_coarsen,
    )
    assert evidence.is_necessary is False
    assert evidence.coarsening is not None
    assert evidence.coarsening.applicable is True

def test_necessity_substitution_returns_not_applicable_when_callback_returns_none() -> None:
    leaves = [_constant_leaf("a", 1.0), _constant_leaf("b", 2.0)]
    expression = compile_expression("a + b")
    model = JointModel(sample_count=128, seed=14)
    summary = joint_sample(leaves, expression, model)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    def _substitute(leaf_id: str) -> Optional[LeafSpec]:
        return None

    evidence = evaluate_necessity(
        leaf=leaves[0],
        other_leaves=[leaves[1]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.5,
        delete=_delete,
        substitute=_substitute,
    )
    assert evidence.substitution is not None
    assert evidence.substitution.applicable is False
    assert evidence.substitution.new_status == "not_applicable"

def test_necessity_substitution_destruction_is_recorded() -> None:
    leaves = [_constant_leaf("a", 1.0), _fitted_leaf("b", 1.0, 5.0, 9.0)]
    expression = compile_expression("a * b")
    model = JointModel(sample_count=512, seed=15)
    summary = joint_sample(leaves, expression, model)

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    def _substitute(leaf_id: str) -> Optional[LeafSpec]:
        if leaf_id == "b":
            # Replace a wide interval with a much tighter one; we expect
            # material degradation to be tiny, so the leaf is redundant.
            return _fitted_leaf("b", 4.9, 5.0, 5.1)
        return None

    evidence = evaluate_necessity(
        leaf=leaves[1],
        other_leaves=[leaves[0]],
        expression=expression,
        model=model,
        baseline=summary,
        material_degradation=0.5,
        delete=_delete,
        substitute=_substitute,
    )
    # Substitution made b narrower; branch still answerable, so leaf is redundant.
    assert evidence.is_necessary is False

@pytest.mark.parametrize("kind", ["delete", "coarsen"])
def test_necessity_invalid_intervention_width_is_honest(kind: str) -> None:
    leaves = [_unknown_leaf("a", domain=(0.5, 1.5)), _unknown_leaf("b", domain=(2.0, 3.0))]
    expression, model = compile_expression("a"), JointModel(sample_count=64, seed=70)
    kwargs = {"delete": lambda leaf_id: [leaf for leaf in leaves if leaf.leaf_id != leaf_id]}
    target, others = leaves[1], [leaves[0]]
    if kind == "coarsen":
        target, others = leaves[0], [leaves[1]]
        kwargs["coarsen"] = lambda leaf_id: _unknown_leaf("a", domain=(0.0, 2.0))
    evidence = evaluate_necessity(target, others, expression, model, joint_sample(leaves, expression, model), material_degradation=0.1, **kwargs)
    result = evidence.deletion if kind == "delete" else evidence.coarsening
    assert result is not None and result.new_status == "invalid_interval"
    assert any("width_after=invalid" in note for note in result.notes)
