import pytest

from aie_decision.measurement import (
    Answerability,
    BoundBasis,
    BoundDerivation,
    DependenceCase,
    FillStatus,
    MissingCondition,
    StopRule,
    analyze_sensitivity,
    evaluate_stop_rule,
    propagate_uncertainty,
    rank_information_targets,
    threshold_robustness,
)


def condition(identifier: str, lower: float, upper: float) -> MissingCondition:
    derivation = BoundDerivation(BoundBasis.REFERENCE_CLASS, "historical range", ("source-1",), lower, upper)
    return MissingCondition(identifier, "not directly observed", (-100.0, 100.0), (lower, upper), (derivation,))


def test_missing_condition_keeps_bound_provenance_and_override_revision():
    original = condition("elasticity", -2.0, -1.0)
    override = BoundDerivation(BoundBasis.USER_OVERRIDE, "user widened range", ("user-input-7",), -3.0, -0.5)
    revised = original.revise(override)

    assert original.revision == 1
    assert revised.revision == 2
    assert revised.interval == (-3.0, -0.5)
    assert revised.derivations == original.derivations + (override,)
    assert revised.fill_status is FillStatus.BOUNDED
    assert not hasattr(revised, "observed_value")


def test_missing_condition_rejects_unexplained_or_point_observed_fill():
    with pytest.raises(ValueError, match="provenance"):
        BoundDerivation(BoundBasis.MODEL_ESTIMATE, "model", (), 0.0, 1.0)
    with pytest.raises(ValueError, match="observed"):
        MissingCondition("x", "missing", (0.0, 2.0), (1.0, 1.0), (BoundDerivation(BoundBasis.EXPERT_ASSUMPTION, "expert", ("memo",), 1.0, 1.0),), fill_status="observed")


def test_unknown_dependence_expands_to_explicit_alternative_cases():
    conditions = (condition("supply", 0.0, 2.0), condition("delay", 0.0, 3.0))
    cases = propagate_uncertainty(conditions, lambda values, _case: values["supply"] - values["delay"], (DependenceCase.UNKNOWN,))
    assert {case.dependence for case in cases} == {
        DependenceCase.POSITIVE,
        DependenceCase.NEGATIVE,
        DependenceCase.INDEPENDENT,
    }
    assert next(case for case in cases if case.dependence is DependenceCase.INDEPENDENT).lower == -3.0


def test_sensitivity_information_priority_threshold_robustness_and_stop_rule():
    inputs = (condition("dominant", 0.0, 10.0), condition("minor", 0.0, 1.0))
    evaluator = lambda values, _case: values["dominant"] + values["minor"]
    sensitivities = analyze_sensitivity(inputs, evaluator, thresholds=(5.0,))
    priorities = rank_information_targets(sensitivities)
    robustness = threshold_robustness(0.0, 11.0, (5.0,))
    stop = evaluate_stop_rule(inputs, sensitivities, robustness, StopRule(max_unresolved_required=3))

    assert sensitivities[0].condition_id == "dominant"
    assert sensitivities[0].expected_narrowing == pytest.approx(10.0)
    assert priorities[0].condition_id == "dominant"
    assert not robustness.robust
    assert stop.status is Answerability.CONDITIONAL
    assert "decision_changes_across_plausible_bounds" in stop.reasons
    assert "dominant" in stop.observations_needed
