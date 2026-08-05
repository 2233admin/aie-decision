"""Executed deletion, coarsening, and substitution policy for a frontier."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Optional, Protocol, Sequence

from .probability import CompiledExpression, JointModel, LeafSpec, TargetSummary, joint_sample


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
            degradation = _degradation_or_invalid(
                baseline.width, deletion_result.target_width
            )
            if degradation is None:
                reasons.append("deletion_target_width_unavailable")
                # An undeleted branch that cannot produce a finite interval
                # is destructive enough to record as necessary — without
                # it we cannot certify a probability interval.
            elif degradation > material_degradation:
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
            degradation = _degradation_or_invalid(
                baseline.width, coarsening_result.target_width
            )
            if degradation is None:
                reasons.append("coarsening_target_width_unavailable")
            elif degradation > material_degradation:
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
            degradation = _degradation_or_invalid(
                baseline.width, substitution_result.target_width
            )
            if degradation is None:
                reasons.append("substitution_target_width_unavailable")
            elif degradation > material_degradation:
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
        baseline_width=baseline.width if baseline.width is not None else 0.0,
        reasons=tuple(reasons),
    )


def _degradation_or_invalid(
    baseline_width: Optional[float], new_width: Optional[float]
) -> Optional[float]:
    """Compute the fractional width degradation, treating ``None`` as invalid.

    Returns ``None`` when either side is unavailable (the baseline or
    the post-intervention summary did not yield a valid probability
    interval).  Callers must interpret a ``None`` return value as
    "the intervention produced an interval we cannot evaluate" and
    record an honest reason rather than coercing a numeric comparison.
    """

    if baseline_width is None or new_width is None:
        return None
    if baseline_width == 0:
        # A zero-width baseline cannot express a fractional degradation
        # meaningfully; treat any positive post-intervention width as a
        # material regression and a zero-width new result as no change.
        return float("inf") if new_width > 0 else 0.0
    return (new_width - baseline_width) / baseline_width


def _format_width_note(value: Optional[float]) -> str:
    """Format an intervention-result width for a note, surviving ``None``."""

    if value is None:
        return "width_after=invalid"
    return f"width_after={value:.6g}"


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
        new_status="computed" if summary.probability_interval_valid else "invalid_interval",
        notes=(_format_width_note(summary.width),),
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
        new_status="computed" if summary.probability_interval_valid else "invalid_interval",
        notes=(_format_width_note(summary.width),),
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
        new_status="computed" if summary.probability_interval_valid else "invalid_interval",
        notes=(_format_width_note(summary.width),),
    )




__all__ = [
    "DeletionCallback",
    "CoarseningCallback",
    "SubstitutionCallback",
    "InterventionResult",
    "NecessityEvidence",
    "evaluate_necessity",
]

