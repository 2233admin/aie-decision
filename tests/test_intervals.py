import pytest

from aie_decision.intervals import (
    ForecastInterval,
    HistoricalInterval,
    IntervalKind,
    audit_interval,
    calibration_by_cohort,
    empirical_coverage,
    proper_interval_score,
    sharpness,
    validate_interval_semantics,
)


def interval(lower=0.0, upper=1000.0, kind=IntervalKind.PREDICTION):
    return ForecastInterval(
        target="tomorrow price",
        horizon="1 day",
        unit="currency",
        population="product SKU-1",
        coverage_level=0.9,
        conditional_assumptions=("market remains open",),
        generation_method="bounded scenario model",
        reference_time="2026-08-01T00:00:00Z",
        lower=lower,
        upper=upper,
        kind=kind,
    )


def test_future_realization_must_use_prediction_not_confidence_semantics():
    validate_interval_semantics(interval(), target_is_future_realization=True)
    with pytest.raises(ValueError, match="prediction interval"):
        validate_interval_semantics(interval(kind=IntervalKind.CONFIDENCE), target_is_future_realization=True)


def test_current_price_500_zero_to_1000_is_normalized_two_and_uninformative():
    audit = audit_interval(interval(), scale=500.0, baseline_width=800.0)
    assert audit.absolute_width == 1000.0
    assert audit.normalized_width == 2.0
    assert audit.baseline_improvement == pytest.approx(-0.25)
    assert not audit.informative
    assert "uninformative" in audit.flags


def test_coverage_sharpness_and_proper_score_penalize_misses():
    records = (
        HistoricalInterval("retail", "1d", 0.0, 2.0, 1.0),
        HistoricalInterval("retail", "1d", 0.0, 2.0, 3.0),
    )
    assert empirical_coverage(records) == 0.5
    assert sharpness(records) == 2.0
    assert proper_interval_score(0.0, 2.0, 3.0, 0.9) > proper_interval_score(0.0, 2.0, 1.0, 0.9)


def test_decision_gain_flags_interval_spanning_all_action_regions():
    baseline = interval(-100.0, 1100.0)
    audit = audit_interval(interval(), scale=500.0, thresholds=(300.0, 700.0), baseline=baseline)
    assert audit.decision_gain.action_regions_spanned == 3
    assert audit.decision_gain.label == "low_information_gain"
    assert "decision_unresolved" in audit.flags


def test_calibration_reports_cohort_horizon_sample_flags_and_drift_separately():
    records = tuple(
        HistoricalInterval("retail", "1d", 0.0, 1.0, 0.5 if period == 0 else 2.0, period)
        for period in (0, 1)
        for _ in range(4)
    )
    report = calibration_by_cohort(records, coverage_level=0.9, minimum_sample_size=10, drift_tolerance=0.2)[0]
    assert report.cohort == "retail" and report.horizon == "1d"
    assert report.sample_size == 8
    assert report.empirical_coverage == 0.5
    assert report.drift == -1.0
    assert set(report.flags) >= {"insufficient_sample_size", "undercoverage", "calibration_drift"}
