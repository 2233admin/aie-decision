import pytest

from aie_decision.factors import (
    FactorScores,
    FactorStatus,
    assess_identifiability,
    generate_candidate,
    rank_candidates,
    transition_factor,
)


def candidate(identifier: str = "lock-in"):
    return generate_candidate(
        identifier,
        "Residential lock-in",
        ("housing-burden", "commute-time", "relocation-cost"),
        "high joint exit costs suppress feasible relocation",
        ("relocation falls when joint cost rises",),
        ("relocation remains unchanged after joint cost shock",),
    )


def test_generation_retains_inputs_mechanism_and_falsification():
    item = candidate()
    assert item.status is FactorStatus.PROPOSED
    assert item.contributing_conditions == ("housing-burden", "commute-time", "relocation-cost")
    assert item.mechanism
    assert item.observable_implications and item.rejection_conditions


def test_promotion_requires_independent_support_or_declared_latent_treatment():
    item = candidate()
    with pytest.raises(ValueError, match="independent evidence"):
        transition_factor(item, FactorStatus.SUPPORTED, note="sounds plausible")
    supported = transition_factor(item, FactorStatus.SUPPORTED, evidence=("panel-study-1",), note="replicated")
    contradicted = transition_factor(item, FactorStatus.CONTRADICTED, note="pre-registered implication failed")
    retired = transition_factor(contradicted, FactorStatus.RETIRED, note="superseded")
    assert supported.status is FactorStatus.SUPPORTED
    assert retired.status is FactorStatus.RETIRED
    with pytest.raises(ValueError, match="cannot transition"):
        transition_factor(retired, FactorStatus.PROPOSED, note="revive")


def test_non_identifiable_candidate_cannot_be_promoted():
    item = assess_identifiability(candidate(), ("relocation falls when joint cost rises",))
    assert not item.identifiable
    with pytest.raises(ValueError, match="non-identifiable"):
        transition_factor(item, FactorStatus.SUPPORTED, evidence=("same-inputs",), note="not distinguishing")


def test_ranking_prefers_observable_nonredundant_candidate_and_exposes_components():
    observable = candidate("observable")
    redundant = candidate("redundant")
    common = dict(explanatory_gain=0.8, incremental_predictive_value=0.7, stability=0.8, uncertainty=0.2)
    ranked = rank_candidates((
        (redundant, FactorScores(**common, observability=0.3, redundancy=0.9)),
        (observable, FactorScores(**common, observability=0.9, redundancy=0.1)),
    ))
    assert ranked[0].candidate.factor_id == "observable"
    assert ranked[0].scores.observability == 0.9
    assert ranked[0].utility > ranked[1].utility
