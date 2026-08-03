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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Necessity
# ---------------------------------------------------------------------------


def _identity_delete_factory(leaves: Sequence[LeafSpec]) -> DeletionCallback:
    by_id = {leaf.leaf_id: leaf for leaf in leaves}

    def _delete(leaf_id: str) -> Optional[Sequence[LeafSpec]]:
        if leaf_id == "structural":
            return None  # branch non-computable
        return [leaf for leaf in leaves if leaf.leaf_id != leaf_id]

    return _delete


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


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Frontier certification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Certification-blocking: unknown marginal and unknown dependence
# ---------------------------------------------------------------------------


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
