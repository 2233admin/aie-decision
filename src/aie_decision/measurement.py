"""Bounded measurement and uncertainty propagation for missing conditions.

The module deliberately has no point-imputation API.  A missing condition stays a
bounded value with provenance until an actual observation is supplied elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from math import isfinite
from typing import Callable, Iterable, Mapping, Sequence


class BoundBasis(str, Enum):
    HARD_CONSTRAINT = "hard_constraint"
    REFERENCE_CLASS = "reference_class"
    MODEL_ESTIMATE = "model_estimate"
    EXPERT_ASSUMPTION = "expert_assumption"
    USER_OVERRIDE = "user_override"


class FillStatus(str, Enum):
    MISSING = "missing"
    BOUNDED = "bounded"


class DependenceCase(str, Enum):
    INDEPENDENT = "independent"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    CONDITIONAL = "conditionally_answerable"
    NOT_ANSWERABLE = "not_answerable"


@dataclass(frozen=True)
class BoundDerivation:
    basis: BoundBasis
    method: str
    provenance: tuple[str, ...]
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _validate_bounds(self.lower, self.upper)
        if not self.method.strip() or not self.provenance:
            raise ValueError("bound derivation requires a method and provenance")


@dataclass(frozen=True)
class MissingCondition:
    condition_id: str
    reason: str
    domain: tuple[float, float]
    interval: tuple[float, float]
    derivations: tuple[BoundDerivation, ...]
    revision: int = 1
    required: bool = True
    fill_status: FillStatus = FillStatus.BOUNDED

    def __post_init__(self) -> None:
        if not self.condition_id.strip() or not self.reason.strip():
            raise ValueError("condition_id and reason are required")
        _validate_bounds(*self.domain)
        _validate_bounds(*self.interval)
        if self.interval[0] < self.domain[0] or self.interval[1] > self.domain[1]:
            raise ValueError("proposed interval must be inside the admissible domain")
        if not self.derivations:
            raise ValueError("a missing condition requires bound provenance")
        if self.fill_status not in (FillStatus.MISSING, FillStatus.BOUNDED):
            raise ValueError("missing conditions cannot be marked as observed")

    def revise(self, derivation: BoundDerivation) -> "MissingCondition":
        """Return a new revision while preserving every previous derivation."""
        if derivation.lower < self.domain[0] or derivation.upper > self.domain[1]:
            raise ValueError("revised interval must be inside the admissible domain")
        return replace(
            self,
            interval=(derivation.lower, derivation.upper),
            derivations=self.derivations + (derivation,),
            revision=self.revision + 1,
            fill_status=FillStatus.BOUNDED,
        )


@dataclass(frozen=True)
class PropagatedCase:
    dependence: DependenceCase
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class Sensitivity:
    condition_id: str
    expected_narrowing: float
    narrowing_fraction: float
    decision_resolution: float


@dataclass(frozen=True)
class ThresholdRobustness:
    robust: bool
    action_regions: tuple[int, ...]
    crossed_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class InformationTarget:
    condition_id: str
    expected_narrowing: float
    decision_resolution: float
    priority: float


@dataclass(frozen=True)
class StopRule:
    max_unresolved_required: int = 0
    max_sensitivity_fraction: float = 0.1
    require_decision_robustness: bool = True


@dataclass(frozen=True)
class StopDecision:
    status: Answerability
    reasons: tuple[str, ...]
    observations_needed: tuple[str, ...]


Evaluator = Callable[[Mapping[str, float], DependenceCase], float]


def propagate_uncertainty(
    conditions: Sequence[MissingCondition],
    evaluator: Evaluator,
    dependence_cases: Sequence[DependenceCase],
) -> tuple[PropagatedCase, ...]:
    """Propagate corner bounds under explicitly named dependence cases.

    ``UNKNOWN`` expands to positive, negative and independence alternatives; it
    is never treated as invisible independence.
    """
    if not conditions:
        raise ValueError("at least one missing condition is required")
    if not dependence_cases:
        raise ValueError("material dependence cases must be declared")
    expanded: list[DependenceCase] = []
    for case in dependence_cases:
        choices = (
            (DependenceCase.POSITIVE, DependenceCase.NEGATIVE, DependenceCase.INDEPENDENT)
            if case is DependenceCase.UNKNOWN
            else (case,)
        )
        for choice in choices:
            if choice not in expanded:
                expanded.append(choice)

    output: list[PropagatedCase] = []
    for case in expanded:
        assignments = _corner_assignments(conditions, case)
        values = [float(evaluator(assignment, case)) for assignment in assignments]
        if not values or not all(isfinite(value) for value in values):
            raise ValueError("evaluator must return finite values")
        output.append(PropagatedCase(case, min(values), max(values)))
    return tuple(output)


def analyze_sensitivity(
    conditions: Sequence[MissingCondition],
    evaluator: Evaluator,
    dependence: DependenceCase = DependenceCase.INDEPENDENT,
    thresholds: Sequence[float] = (),
) -> tuple[Sensitivity, ...]:
    """Estimate value-of-information without labelling an analytic probe observed.

    Each condition is fixed at both of its endpoints in turn.  The mean remaining
    output width is compared with the full bounded width.
    """
    full = propagate_uncertainty(conditions, evaluator, (dependence,))[0]
    results: list[Sensitivity] = []
    for target in conditions:
        widths: list[float] = []
        resolutions: list[float] = []
        for endpoint in target.interval:
            assignments = _corner_assignments(conditions, dependence, {target.condition_id: endpoint})
            values = [float(evaluator(assignment, dependence)) for assignment in assignments]
            lower, upper = min(values), max(values)
            widths.append(upper - lower)
            resolutions.append(1.0 if threshold_robustness(lower, upper, thresholds).robust else 0.0)
        narrowing = max(0.0, full.width - sum(widths) / len(widths))
        fraction = narrowing / full.width if full.width else 0.0
        results.append(Sensitivity(target.condition_id, narrowing, fraction, sum(resolutions) / len(resolutions)))
    return tuple(sorted(results, key=lambda item: (-item.expected_narrowing, item.condition_id)))


def threshold_robustness(lower: float, upper: float, thresholds: Sequence[float]) -> ThresholdRobustness:
    _validate_bounds(lower, upper)
    ordered = tuple(sorted(set(float(value) for value in thresholds)))
    crossed = tuple(value for value in ordered if lower <= value <= upper)
    regions = tuple(range(_region(lower, ordered), _region(upper, ordered) + 1))
    return ThresholdRobustness(len(regions) == 1, regions, crossed)


def rank_information_targets(sensitivities: Iterable[Sensitivity]) -> tuple[InformationTarget, ...]:
    items = tuple(sensitivities)
    max_narrowing = max((item.expected_narrowing for item in items), default=0.0)
    ranked = [
        InformationTarget(
            item.condition_id,
            item.expected_narrowing,
            item.decision_resolution,
            (item.expected_narrowing / max_narrowing if max_narrowing else 0.0) * 0.7
            + item.decision_resolution * 0.3,
        )
        for item in items
    ]
    return tuple(sorted(ranked, key=lambda item: (-item.priority, item.condition_id)))


def evaluate_stop_rule(
    conditions: Sequence[MissingCondition],
    sensitivities: Sequence[Sensitivity],
    robustness: ThresholdRobustness,
    rule: StopRule,
) -> StopDecision:
    unresolved = tuple(condition.condition_id for condition in conditions if condition.required)
    high_sensitivity = tuple(
        item.condition_id for item in sensitivities if item.narrowing_fraction > rule.max_sensitivity_fraction
    )
    reasons: list[str] = []
    if len(unresolved) > rule.max_unresolved_required:
        reasons.append("unresolved_required_limit_exceeded")
    if high_sensitivity:
        reasons.append("sensitivity_limit_exceeded")
    if rule.require_decision_robustness and not robustness.robust:
        reasons.append("decision_changes_across_plausible_bounds")
    if not reasons:
        status = Answerability.ANSWERABLE
    elif not robustness.robust:
        status = Answerability.CONDITIONAL
    else:
        status = Answerability.NOT_ANSWERABLE
    observations = tuple(dict.fromkeys(high_sensitivity + unresolved))
    return StopDecision(status, tuple(reasons), observations)


def _corner_assignments(
    conditions: Sequence[MissingCondition],
    dependence: DependenceCase,
    fixed: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], ...]:
    fixed = fixed or {}
    free = [condition for condition in conditions if condition.condition_id not in fixed]
    if dependence is DependenceCase.POSITIVE:
        corners = [tuple(condition.interval[0] for condition in free), tuple(condition.interval[1] for condition in free)]
    elif dependence is DependenceCase.NEGATIVE and len(free) > 1:
        corners = [
            tuple(condition.interval[index % 2] for index, condition in enumerate(free)),
            tuple(condition.interval[(index + 1) % 2] for index, condition in enumerate(free)),
        ]
    else:
        corners = list(product(*(condition.interval for condition in free)))
    return tuple({**fixed, **dict(zip((condition.condition_id for condition in free), values))} for values in corners)


def _region(value: float, thresholds: Sequence[float]) -> int:
    return sum(value >= threshold for threshold in thresholds)


def _validate_bounds(lower: float, upper: float) -> None:
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError("bounds must be finite and ordered")
