"""Forecast-interval semantics, metrics, and historical calibration audits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import fmean
from typing import Iterable, Sequence


class IntervalKind(str, Enum):
    PREDICTION = "future_value_prediction"
    CONFIDENCE = "parameter_confidence"


@dataclass(frozen=True)
class ForecastInterval:
    target: str
    horizon: str
    unit: str
    population: str
    coverage_level: float
    conditional_assumptions: tuple[str, ...]
    generation_method: str
    reference_time: str
    lower: float
    upper: float
    kind: IntervalKind

    def __post_init__(self) -> None:
        required = (self.target, self.horizon, self.unit, self.population, self.generation_method, self.reference_time)
        if not all(value.strip() for value in required):
            raise ValueError("all interval semantics must be declared")
        if not 0.0 < self.coverage_level < 1.0:
            raise ValueError("coverage_level must be between zero and one")
        _validate_bounds(self.lower, self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class DecisionGain:
    score: float
    label: str
    action_regions_spanned: int


@dataclass(frozen=True)
class IntervalAudit:
    absolute_width: float
    normalized_width: float
    baseline_improvement: float | None
    decision_gain: DecisionGain
    informative: bool
    flags: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalInterval:
    cohort: str
    horizon: str
    lower: float
    upper: float
    outcome: float
    period: int = 0

    def __post_init__(self) -> None:
        _validate_bounds(self.lower, self.upper)


@dataclass(frozen=True)
class CalibrationReport:
    cohort: str
    horizon: str
    sample_size: int
    empirical_coverage: float
    sharpness: float
    mean_interval_score: float
    drift: float | None
    flags: tuple[str, ...]


def validate_interval_semantics(interval: ForecastInterval, *, target_is_future_realization: bool) -> None:
    if target_is_future_realization and interval.kind is not IntervalKind.PREDICTION:
        raise ValueError("a future realized value requires a prediction interval, not a confidence interval")
    if not target_is_future_realization and interval.kind is IntervalKind.PREDICTION:
        raise ValueError("a parameter estimate requires confidence-interval semantics")


def normalized_width(interval: ForecastInterval, scale: float) -> float:
    if not isfinite(scale) or scale <= 0:
        raise ValueError("normalization scale must be positive and finite")
    return interval.width / scale


def baseline_improvement(interval: ForecastInterval, baseline_width: float | None) -> float | None:
    if baseline_width is None:
        return None
    if not isfinite(baseline_width) or baseline_width <= 0:
        raise ValueError("baseline width must be positive and finite")
    return (baseline_width - interval.width) / baseline_width


def empirical_coverage(records: Iterable[HistoricalInterval]) -> float:
    items = tuple(records)
    if not items:
        raise ValueError("coverage requires historical outcomes")
    return sum(item.lower <= item.outcome <= item.upper for item in items) / len(items)


def sharpness(records: Iterable[HistoricalInterval]) -> float:
    items = tuple(records)
    if not items:
        raise ValueError("sharpness requires historical intervals")
    return fmean(item.upper - item.lower for item in items)


def proper_interval_score(lower: float, upper: float, outcome: float, coverage_level: float) -> float:
    """Winkler interval score; lower is better."""
    _validate_bounds(lower, upper)
    if not 0.0 < coverage_level < 1.0:
        raise ValueError("coverage_level must be between zero and one")
    alpha = 1.0 - coverage_level
    penalty = 0.0
    if outcome < lower:
        penalty = 2.0 / alpha * (lower - outcome)
    elif outcome > upper:
        penalty = 2.0 / alpha * (outcome - upper)
    return upper - lower + penalty


def decision_information_gain(
    interval: ForecastInterval,
    thresholds: Sequence[float],
    baseline: ForecastInterval | None = None,
) -> DecisionGain:
    ordered = tuple(sorted(set(float(value) for value in thresholds)))
    spanned = _regions_spanned(interval.lower, interval.upper, ordered)
    if baseline is not None:
        baseline_spanned = _regions_spanned(baseline.lower, baseline.upper, ordered)
        score = max(0.0, min(1.0, (baseline_spanned - spanned) / max(1, baseline_spanned - 1)))
    elif ordered:
        score = 1.0 if spanned == 1 else max(0.0, 1.0 - (spanned - 1) / len(ordered))
    else:
        score = 0.0
    label = "resolves_decision" if spanned == 1 else ("narrows_decision" if score > 0 else "low_information_gain")
    return DecisionGain(score, label, spanned)


def audit_interval(
    interval: ForecastInterval,
    *,
    scale: float,
    baseline_width: float | None = None,
    thresholds: Sequence[float] = (),
    baseline: ForecastInterval | None = None,
    documented_decision_context_useful: bool = False,
) -> IntervalAudit:
    width_ratio = normalized_width(interval, scale)
    improvement = baseline_improvement(interval, baseline_width)
    gain = decision_information_gain(interval, thresholds, baseline)
    flags: list[str] = []
    if width_ratio >= 2.0 and not documented_decision_context_useful:
        flags.append("excessive_normalized_width")
    if improvement is not None and improvement <= 0:
        flags.append("no_baseline_improvement")
    if thresholds and gain.label == "low_information_gain":
        flags.append("decision_unresolved")
    informative = documented_decision_context_useful or not flags and (gain.score > 0 or not thresholds)
    if not informative:
        flags.append("uninformative")
    return IntervalAudit(interval.width, width_ratio, improvement, gain, informative, tuple(dict.fromkeys(flags)))


def calibration_by_cohort(
    records: Iterable[HistoricalInterval],
    *,
    coverage_level: float,
    minimum_sample_size: int = 30,
    undercoverage_tolerance: float = 0.05,
    excessive_width: float | None = None,
    drift_tolerance: float = 0.1,
) -> tuple[CalibrationReport, ...]:
    grouped: dict[tuple[str, str], list[HistoricalInterval]] = defaultdict(list)
    for item in records:
        grouped[(item.cohort, item.horizon)].append(item)
    reports: list[CalibrationReport] = []
    for (cohort, horizon), items in sorted(grouped.items()):
        coverage = empirical_coverage(items)
        width = sharpness(items)
        score = fmean(proper_interval_score(i.lower, i.upper, i.outcome, coverage_level) for i in items)
        flags: list[str] = []
        if len(items) < minimum_sample_size:
            flags.append("insufficient_sample_size")
        if coverage < coverage_level - undercoverage_tolerance:
            flags.append("undercoverage")
        if excessive_width is not None and width > excessive_width:
            flags.append("excessive_width")
        drift = _coverage_drift(items)
        if drift is not None and abs(drift) > drift_tolerance:
            flags.append("calibration_drift")
        reports.append(CalibrationReport(cohort, horizon, len(items), coverage, width, score, drift, tuple(flags)))
    return tuple(reports)


def _coverage_drift(items: Sequence[HistoricalInterval]) -> float | None:
    periods = sorted(set(item.period for item in items))
    if len(periods) < 2:
        return None
    older = [item for item in items if item.period == periods[0]]
    recent = [item for item in items if item.period == periods[-1]]
    return empirical_coverage(recent) - empirical_coverage(older)


def _regions_spanned(lower: float, upper: float, thresholds: Sequence[float]) -> int:
    low_region = sum(lower >= threshold for threshold in thresholds)
    high_region = sum(upper >= threshold for threshold in thresholds)
    return high_region - low_region + 1


def _validate_bounds(lower: float, upper: float) -> None:
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError("bounds must be finite and ordered")
