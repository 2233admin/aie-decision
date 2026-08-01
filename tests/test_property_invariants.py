"""Deterministic property checks across broad input families."""

import pytest

from aie_decision.intervals import proper_interval_score
from aie_decision.measurement import BoundBasis, BoundDerivation, MissingCondition
from aie_decision.recompute import STAGES, plan_recomputation


def test_property_interval_score_is_nonnegative_and_penalizes_equal_width_misses():
    for coverage in (0.5, 0.8, 0.9, 0.95):
        inside = proper_interval_score(10, 20, 15, coverage)
        low_miss = proper_interval_score(10, 20, 5, coverage)
        high_miss = proper_interval_score(10, 20, 25, coverage)
        assert inside >= 0
        assert low_miss > inside
        assert high_miss > inside
        assert low_miss == pytest.approx(high_miss)


def test_property_missing_condition_never_accepts_inverted_bounds():
    derivation = BoundDerivation(BoundBasis.HARD_CONSTRAINT, "fixture", ("source-1",), 0, 100)
    for lower, upper in ((1, 0), (10, -10), (float("inf"), 1)):
        with pytest.raises(ValueError):
            MissingCondition("gap", "unknown", (0, 100), (lower, upper), (derivation,))


def test_property_recomputation_is_a_downstream_suffix():
    plans = (
        plan_recomputation(answer_changed=True),
        plan_recomputation(sources_changed=True),
        plan_recomputation(assumptions_changed=True),
        plan_recomputation(bounds_changed=True),
        plan_recomputation(sources_changed=True, bounds_changed=True),
    )
    for plan in plans:
        start = STAGES.index(plan.stages[0])
        assert plan.stages == STAGES[start:]
