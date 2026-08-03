"""Sufficiency, necessity, saturation, and frontier certification.

This module accepts plain immutable inputs from :mod:`aie_decision.probability`
and the destructive / refinement callbacks provided by the calling runtime.
It deliberately avoids any reference to the decomposition tree track so the
two can evolve in parallel.

* **Sufficiency** checks the target summary against an explicit
  :class:`Tolerance`.  When no tolerance is supplied, sufficiency is reported
  as ``not_assessable`` rather than silently passing.
* **Necessity** executes a deletion callback plus an applicable coarsening
  or substitution callback for every retained leaf.  A leaf is *locally
  necessary* only when deletion (or coarsening / substitution, when
  applicable) destroys answerability or causes a material degradation in
  target width or decision stability.
* **Saturation** evaluates a sequence of refinement callbacks and reports
  whether the expected target-width improvement of any explored refinement
  exceeds the supplied material-improvement threshold.  Saturation is
  explicitly *conditional* on the explored refinements.

Frontier certification requires all three gates to pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Mapping, Optional, Protocol, Sequence

from .probability import (
    CalibrationLabel,
    CompiledExpression,
    JointModel,
    LeafSpec,
    TargetSummary,
    joint_sample,
)


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tolerance:
    """An explicit answerability tolerance for the target quantity.

    Either ``acceptable_width`` or ``acceptable_relative_width`` may be set.
    ``decision_thresholds`` enables an explicit decision-stability check
    independent of an absolute width.
    """

    acceptable_width: Optional[float] = None
    acceptable_relative_width: Optional[float] = None
    reference_value: Optional[float] = None
    decision_thresholds: tuple[float, ...] = ()
    require_calibration: bool = False

    def __post_init__(self) -> None:
        if self.acceptable_width is not None:
            if not isfinite(self.acceptable_width) or self.acceptable_width < 0:
                raise ValueError("acceptable_width must be finite and non-negative")
        if self.acceptable_relative_width is not None:
            if (
                not isfinite(self.acceptable_relative_width)
                or self.acceptable_relative_width < 0
            ):
                raise ValueError(
                    "acceptable_relative_width must be finite and non-negative"
                )
            if self.reference_value is None or self.reference_value == 0:
                raise ValueError(
                    "acceptable_relative_width requires a non-zero reference_value"
                )


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------


class SufficiencyStatus(str, Enum):
    PASSES = "passes"
    FAILS = "fails"
    NOT_ASSESSABLE = "not_assessable"


@dataclass(frozen=True, slots=True)
class SufficiencyEvidence:
    """Record of a sufficiency evaluation against an explicit tolerance."""

    status: SufficiencyStatus
    target_width: float
    relative_width: Optional[float]
    tolerance: Tolerance
    calibration: CalibrationLabel
    reasons: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        return self.status is SufficiencyStatus.PASSES


def _action_regions(lower: float, upper: float, thresholds: Sequence[float]) -> int:
    ordered = tuple(sorted(set(float(value) for value in thresholds)))
    low_region = sum(lower >= threshold for threshold in ordered)
    high_region = sum(upper >= threshold for threshold in ordered)
    return high_region - low_region + 1


def evaluate_sufficiency(
    summary: TargetSummary, tolerance: Tolerance
) -> SufficiencyEvidence:
    """Evaluate ``summary`` against the explicit ``tolerance``.

    The summary's ``probability_interval_valid`` flag is the primary gate:
    if the target 90 percent interval is invalid because a required leaf
    marginal is unknown or the joint dependence is unresolved, this
    function reports :attr:`SufficiencyStatus.NOT_ASSESSABLE` regardless
    of the requested tolerance, regardless of the ``require_calibration``
    flag, and regardless of how narrow the scenario envelope happens to
    be.  A narrow scenario range is not a certified probability interval.
    """

    if not summary.probability_interval_valid or summary.width is None:
        return SufficiencyEvidence(
            status=SufficiencyStatus.NOT_ASSESSABLE,
            target_width=summary.width,
            relative_width=None,
            tolerance=tolerance,
            calibration=summary.calibration,
            reasons=("probability_interval_invalid",),
        )

    reasons: list[str] = []
    has_tolerance = (
        tolerance.acceptable_width is not None
        or tolerance.acceptable_relative_width is not None
        or bool(tolerance.decision_thresholds)
    )
    if not has_tolerance:
        return SufficiencyEvidence(
            status=SufficiencyStatus.NOT_ASSESSABLE,
            target_width=summary.width,
            relative_width=None,
            tolerance=tolerance,
            calibration=summary.calibration,
            reasons=("no_explicit_tolerance",),
        )

    relative_width: Optional[float] = None
    if tolerance.acceptable_relative_width is not None and tolerance.reference_value:
        reference = abs(tolerance.reference_value)
        if reference > 0:
            assert summary.width is not None
            relative_width = summary.width / reference

    passes_width = True
    if tolerance.acceptable_width is not None:
        assert summary.width is not None
        if summary.width > tolerance.acceptable_width:
            passes_width = False
            reasons.append(
                f"target_width {summary.width:.6g} exceeds acceptable_width "
                f"{tolerance.acceptable_width:.6g}"
            )
    if (
        tolerance.acceptable_relative_width is not None
        and relative_width is not None
    ):
        if relative_width > tolerance.acceptable_relative_width:
            passes_width = False
            reasons.append(
                f"relative_width {relative_width:.6g} exceeds acceptable_relative_width "
                f"{tolerance.acceptable_relative_width:.6g}"
            )

    passes_decision = True
    if tolerance.decision_thresholds:
        ordered = tuple(sorted(set(tolerance.decision_thresholds)))
        p05 = summary.p05 if summary.p05 is not None else 0.0
        p95 = summary.p95 if summary.p95 is not None else 0.0
        action_regions = _action_regions(p05, p95, ordered)
        if action_regions > 1:
            passes_decision = False
            reasons.append(
                f"interval spans {action_regions} action regions across {ordered}"
            )

    if tolerance.require_calibration:
        if summary.calibration is not CalibrationLabel.UNMEASURED:
            passes_width = False
            reasons.append(
                f"calibration {summary.calibration.value} is not an unmeasured pure sample"
            )

    status = (
        SufficiencyStatus.PASSES
        if passes_width and passes_decision and not reasons
        else SufficiencyStatus.FAILS
    )

    return SufficiencyEvidence(
        status=status,
        target_width=summary.width,
        relative_width=relative_width,
        tolerance=tolerance,
        calibration=summary.calibration,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Necessity callbacks and evidence
# ---------------------------------------------------------------------------


class DeletionCallback(Protocol):
    """Return the leaf set that results from deleting the named leaf.

    Returning ``None`` means the deletion made the branch non-computable.
    """

    def __call__(self, leaf_id: str) -> Optional[Sequence[LeafSpec]]: ...


class CoarseningCallback(Protocol):
    """Return a coarsened replacement leaf for ``leaf_id``.

    Returning ``None`` means coarsening is not applicable.
    """

    def __call__(self, leaf_id: str) -> Optional[LeafSpec]: ...


class SubstitutionCallback(Protocol):
    """Return a same-target substitute leaf for ``leaf_id``.

    Returning ``None`` means substitution is not applicable.
    """

    def __call__(self, leaf_id: str) -> Optional[LeafSpec]: ...


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """Result of executing one destructive intervention against the frontier."""

    intervention: str
    applicable: bool
    target_width: Optional[float]
    new_status: str  # "computed", "non_computable", "not_applicable"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NecessityEvidence:
    """Locally executed necessity record for a single retained leaf."""

    leaf_id: str
    is_necessary: bool
    deletion: InterventionResult
    coarsening: Optional[InterventionResult]
    substitution: Optional[InterventionResult]
    material_threshold: float
    baseline_width: float
    reasons: tuple[str, ...] = ()


def evaluate_necessity(
    leaf: LeafSpec,
    other_leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
    baseline: TargetSummary,
    *,
    material_degradation: float,
    delete: DeletionCallback,
    coarsen: Optional[CoarseningCallback] = None,
    substitute: Optional[SubstitutionCallback] = None,
) -> NecessityEvidence:
    """Execute destructive interventions for one retained leaf.

    ``material_degradation`` is the fractional increase in target width
    (relative to ``baseline.width``) above which removal is considered
    destructive.
    """

    if not isfinite(material_degradation) or material_degradation < 0:
        raise ValueError("material_degradation must be finite and non-negative")
    if leaf.leaf_id in {other.leaf_id for other in other_leaves}:
        raise ValueError("leaf appears in both leaf and other_leaves")

    all_leaves = (leaf, *other_leaves)
    full_leaves = tuple(all_leaves)
    deletion_result = _execute_deletion(
        leaf.leaf_id, full_leaves, expression, model, delete
    )

    coarsening_result = (
        _execute_coarsening(
            leaf.leaf_id,
            full_leaves,
            expression,
            model,
            baseline,
            material_degradation,
            coarsen,
        )
        if coarsen is not None
        else None
    )
    substitution_result = (
        _execute_substitution(
            leaf.leaf_id,
            full_leaves,
            expression,
            model,
            baseline,
            material_degradation,
            substitute,
        )
        if substitute is not None
        else None
    )

    reasons: list[str] = []
    is_necessary = True

    if not deletion_result.applicable:
        reasons.append("deletion_not_applicable")
        is_necessary = False
    else:
        if deletion_result.new_status == "non_computable":
            reasons.append("deletion_makes_branch_non_computable")
        else:
            assert deletion_result.target_width is not None
            degradation = (
                deletion_result.target_width - baseline.width
            ) / baseline.width if baseline.width else 0.0
            if degradation > material_degradation:
                reasons.append(
                    f"deletion_degrades_width_by_{degradation:.3f}_over_threshold"
                )
            else:
                reasons.append("deletion_within_material_tolerance")
                is_necessary = False

    if coarsening_result is not None:
        if not coarsening_result.applicable:
            pass  # coarsening not applicable does not weaken necessity
        elif coarsening_result.new_status == "non_computable":
            reasons.append("coarsening_makes_branch_non_computable")
        else:
            assert coarsening_result.target_width is not None
            degradation = (
                coarsening_result.target_width - baseline.width
            ) / baseline.width if baseline.width else 0.0
            if degradation > material_degradation:
                reasons.append(
                    f"coarsening_degrades_width_by_{degradation:.3f}_over_threshold"
                )
            else:
                reasons.append("coarsening_within_material_tolerance")
                is_necessary = False

    if substitution_result is not None:
        if not substitution_result.applicable:
            pass
        elif substitution_result.new_status == "non_computable":
            reasons.append("substitution_makes_branch_non_computable")
        else:
            assert substitution_result.target_width is not None
            degradation = (
                substitution_result.target_width - baseline.width
            ) / baseline.width if baseline.width else 0.0
            if degradation > material_degradation:
                reasons.append(
                    f"substitution_degrades_width_by_{degradation:.3f}_over_threshold"
                )
            else:
                reasons.append("substitution_within_material_tolerance")
                is_necessary = False

    return NecessityEvidence(
        leaf_id=leaf.leaf_id,
        is_necessary=is_necessary,
        deletion=deletion_result,
        coarsening=coarsening_result,
        substitution=substitution_result,
        material_threshold=material_degradation,
        baseline_width=baseline.width,
        reasons=tuple(reasons),
    )


def _execute_deletion(
    leaf_id: str,
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
    delete: DeletionCallback,
) -> InterventionResult:
    try:
        replacement = delete(leaf_id)
    except Exception as exc:  # pragma: no cover - defensive
        return InterventionResult(
            intervention="deletion",
            applicable=False,
            target_width=None,
            new_status="not_applicable",
            notes=(f"deletion_callback_error:{exc!r}",),
        )
    if replacement is None:
        return InterventionResult(
            intervention="deletion",
            applicable=True,
            target_width=None,
            new_status="non_computable",
            notes=("deletion_returns_none",),
        )
    referenced = set(expression.variables)
    available = {leaf.leaf_id for leaf in replacement}
    if not referenced.issubset(available):
        missing = referenced - available
        return InterventionResult(
            intervention="deletion",
            applicable=True,
            target_width=None,
            new_status="non_computable",
            notes=(f"missing_referenced_leaves:{','.join(sorted(missing))}",),
        )
    summary = joint_sample(replacement, expression, model)
    return InterventionResult(
        intervention="deletion",
        applicable=True,
        target_width=summary.width,
        new_status="computed",
        notes=(f"width_after_deletion={summary.width:.6g}",),
    )


def _execute_coarsening(
    leaf_id: str,
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
    baseline: TargetSummary,
    material_degradation: float,
    coarsen: CoarseningCallback,
) -> Optional[InterventionResult]:
    try:
        coarsened = coarsen(leaf_id)
    except Exception as exc:  # pragma: no cover - defensive
        return InterventionResult(
            intervention="coarsening",
            applicable=False,
            target_width=None,
            new_status="not_applicable",
            notes=(f"coarsening_callback_error:{exc!r}",),
        )
    if coarsened is None:
        return InterventionResult(
            intervention="coarsening",
            applicable=False,
            target_width=None,
            new_status="not_applicable",
            notes=("coarsening_returns_none",),
        )
    replaced = tuple(
        coarsened if leaf.leaf_id == leaf_id else leaf for leaf in leaves
    )
    summary = joint_sample(replaced, expression, model)
    return InterventionResult(
        intervention="coarsening",
        applicable=True,
        target_width=summary.width,
        new_status="computed",
        notes=(f"width_after_coarsening={summary.width:.6g}",),
    )


def _execute_substitution(
    leaf_id: str,
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
    baseline: TargetSummary,
    material_degradation: float,
    substitute: SubstitutionCallback,
) -> Optional[InterventionResult]:
    try:
        replacement = substitute(leaf_id)
    except Exception as exc:  # pragma: no cover - defensive
        return InterventionResult(
            intervention="substitution",
            applicable=False,
            target_width=None,
            new_status="not_applicable",
            notes=(f"substitution_callback_error:{exc!r}",),
        )
    if replacement is None:
        return InterventionResult(
            intervention="substitution",
            applicable=False,
            target_width=None,
            new_status="not_applicable",
            notes=("substitution_returns_none",),
        )
    replaced = tuple(
        replacement if leaf.leaf_id == leaf_id else leaf for leaf in leaves
    )
    summary = joint_sample(replaced, expression, model)
    return InterventionResult(
        intervention="substitution",
        applicable=True,
        target_width=summary.width,
        new_status="computed",
        notes=(f"width_after_substitution={summary.width:.6g}",),
    )


# ---------------------------------------------------------------------------
# Saturation callbacks and evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefinementProposal:
    """A refinement candidate offered by the calling runtime.

    ``proposed_leaves`` is the new leaf set that would replace ``leaf_id``.
    ``expected_description`` records the rationale so saturation evidence
    can be audited later.
    """

    leaf_id: str
    proposed_leaves: Sequence[LeafSpec]
    description: str
    expected_target_narrowing: Optional[float] = None


class RefinementCallback(Protocol):
    """Return the next refinement proposal to evaluate, or ``None`` when exhausted."""

    def __call__(
        self,
        current_leaves: Sequence[LeafSpec],
        baseline: TargetSummary,
    ) -> Optional[RefinementProposal]: ...


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """Record of evaluating one explored refinement proposal."""

    leaf_id: str
    description: str
    expected_narrowing: Optional[float]
    observed_narrowing: float
    over_material_threshold: bool


@dataclass(frozen=True, slots=True)
class SaturationEvidence:
    """Saturation record built from a sequence of explored refinements."""

    saturated: bool
    material_threshold: float
    explored: tuple[RefinementResult, ...]
    best_observed_narrowing: float
    notes: tuple[str, ...] = ()


def evaluate_saturation(
    current_leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
    baseline: TargetSummary,
    *,
    material_improvement_threshold: float,
    next_refinement: RefinementCallback,
    max_iterations: int = 32,
) -> SaturationEvidence:
    """Walk the refinement callback and aggregate observed improvements.

    Saturation is conditional on the refinements the runtime actually
    explored; unexplored alternatives are recorded as ``notes`` so the
    certification report does not overclaim.
    """

    if not isfinite(material_improvement_threshold) or material_improvement_threshold < 0:
        raise ValueError(
            "material_improvement_threshold must be finite and non-negative"
        )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    explored: list[RefinementResult] = []
    best_observed = 0.0
    notes: list[str] = []
    cursor: tuple[LeafSpec, ...] = tuple(current_leaves)
    exhausted = False

    for iteration in range(max_iterations):
        try:
            proposal = next_refinement(cursor, baseline)
        except StopIteration:
            exhausted = True
            break
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"refinement_callback_error_iter{iteration}:{exc!r}")
            break
        if proposal is None:
            exhausted = True
            break

        if proposal.leaf_id not in {leaf.leaf_id for leaf in cursor}:
            notes.append(
                f"proposed_leaf_id_not_in_frontier:{proposal.leaf_id}"
            )
            continue

        replaced = _replace_leaf(cursor, proposal.leaf_id, proposal.proposed_leaves)
        try:
            new_summary = joint_sample(replaced, expression, model)
        except Exception as exc:
            notes.append(
                f"refinement_evaluation_failed:{proposal.leaf_id}:{exc!r}"
            )
            continue
        observed = max(0.0, baseline.width - new_summary.width)
        best_observed = max(best_observed, observed)
        expected = proposal.expected_target_narrowing
        if expected is None:
            expected_value = observed
        else:
            expected_value = max(0.0, float(expected))
        explored.append(
            RefinementResult(
                leaf_id=proposal.leaf_id,
                description=proposal.description,
                expected_narrowing=expected_value,
                observed_narrowing=observed,
                over_material_threshold=observed > material_improvement_threshold,
            )
        )
        if observed > material_improvement_threshold:
            notes.append(
                f"refinement_over_threshold:{proposal.leaf_id}:{observed:.6g}"
            )
            break

    saturated = (
        not any(result.over_material_threshold for result in explored)
        and exhausted
    )

    if not exhausted and not any(result.over_material_threshold for result in explored):
        notes.append("refinement_iterations_exhausted_without_pass")

    return SaturationEvidence(
        saturated=saturated,
        material_threshold=material_improvement_threshold,
        explored=tuple(explored),
        best_observed_narrowing=best_observed,
        notes=tuple(notes),
    )


def _replace_leaf(
    leaves: Sequence[LeafSpec],
    leaf_id: str,
    proposed: Sequence[LeafSpec],
) -> tuple[LeafSpec, ...]:
    kept = tuple(leaf for leaf in leaves if leaf.leaf_id != leaf_id)
    proposed_ids = {leaf.leaf_id for leaf in proposed}
    if any(leaf.leaf_id in proposed_ids for leaf in kept):
        raise ValueError("refinement introduces a leaf_id already on the frontier")
    return kept + tuple(proposed)


# ---------------------------------------------------------------------------
# Frontier certification
# ---------------------------------------------------------------------------


class FrontierStatus(str, Enum):
    """Final frontier status after sufficiency, necessity, and saturation."""

    CERTIFIED = "certified"
    STRUCTURALLY_COMPLETE = "structurally_complete"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class FrontierCertification:
    """Combined frontier certification record.

    ``summary`` is the honest propagated target interval, ``sufficiency``
    the gate against the explicit tolerance, ``necessity`` the executed
    destructive interventions, and ``saturation`` the explored refinements.
    """

    status: FrontierStatus
    summary: TargetSummary
    sufficiency: SufficiencyEvidence
    necessity: tuple[NecessityEvidence, ...]
    saturation: SaturationEvidence

    @property
    def certified(self) -> bool:
        return self.status is FrontierStatus.CERTIFIED

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.sufficiency.status is not SufficiencyStatus.PASSES:
            reasons.append(f"sufficiency:{self.sufficiency.status.value}")
            reasons.extend(self.sufficiency.reasons)
        for evidence in self.necessity:
            if not evidence.is_necessary:
                reasons.append(
                    f"necessity_redundant:{evidence.leaf_id}:{'/'.join(evidence.reasons)}"
                )
        if not self.saturation.saturated:
            reasons.append("saturation:not_saturated")
            reasons.extend(self.saturation.notes)
        return tuple(dict.fromkeys(reasons))


def certify_frontier(
    summary: TargetSummary,
    sufficiency: SufficiencyEvidence,
    necessity: Sequence[NecessityEvidence],
    saturation: SaturationEvidence,
) -> FrontierCertification:
    """Combine the three gates into a single certification verdict."""

    if sufficiency.status is not SufficiencyStatus.PASSES:
        status = FrontierStatus.INSUFFICIENT
    elif any(not evidence.is_necessary for evidence in necessity):
        status = FrontierStatus.STRUCTURALLY_COMPLETE
    elif not saturation.saturated:
        status = FrontierStatus.STRUCTURALLY_COMPLETE
    else:
        status = FrontierStatus.CERTIFIED

    return FrontierCertification(
        status=status,
        summary=summary,
        sufficiency=sufficiency,
        necessity=tuple(necessity),
        saturation=saturation,
    )


__all__ = [
    "Tolerance",
    "SufficiencyStatus",
    "SufficiencyEvidence",
    "evaluate_sufficiency",
    "DeletionCallback",
    "CoarseningCallback",
    "SubstitutionCallback",
    "InterventionResult",
    "NecessityEvidence",
    "evaluate_necessity",
    "RefinementProposal",
    "RefinementCallback",
    "RefinementResult",
    "SaturationEvidence",
    "evaluate_saturation",
    "FrontierStatus",
    "FrontierCertification",
    "certify_frontier",
]