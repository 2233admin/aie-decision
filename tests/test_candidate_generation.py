from dataclasses import dataclass

import pytest

from aie_decision.candidate_generation import (
    FailureDiagnostic,
    GeneratedCandidate,
    MutationTemplate,
    ProposedMutation,
    generate_candidates,
)


@dataclass(frozen=True)
class Parent:
    candidate_id: str = "rough"
    formula: str = "rough_total"
    prior_weight: float = 0.8


def test_extracts_mechanical_failure_diagnostics_from_evaluation():
    diagnostic = FailureDiagnostic.from_evaluation(
        {"within_acceptable_width": False, "next_measurement": "visitors"},
        missing_variables=("conversion_rate",),
    )

    assert diagnostic.reasons == (
        "interval_too_wide",
        "next_measurement",
        "missing_variables",
    )
    assert diagnostic.next_measurement == "visitors"


def test_declared_template_generates_one_structured_candidate_per_missing_variable():
    diagnostic = FailureDiagnostic(
        ("missing_variables",), missing_variables=("visitors", "conversion_rate")
    )
    template = MutationTemplate(
        template_id="multiply-missing",
        formula_template="({parent_formula}) * {missing_variable}",
        diagnostic_reasons=("missing_variables",),
        mutation_kind="expand",
        prior_multiplier=0.5,
    )

    candidates = generate_candidates(Parent(), diagnostic, (template,))

    assert [item.formula for item in candidates] == [
        "(rough_total) * visitors",
        "(rough_total) * conversion_rate",
    ]
    assert all(isinstance(item, GeneratedCandidate) for item in candidates)
    assert all(item.parent_candidate_id == "rough" for item in candidates)
    assert all(item.prior_weight == 0.4 for item in candidates)
    assert all(item.heuristic and not item.calibrated for item in candidates)


def test_next_measurement_drives_a_targeted_revision():
    diagnostic = FailureDiagnostic(
        ("interval_too_wide", "next_measurement"),
        next_measurement="participation_rate",
    )
    template = MutationTemplate(
        template_id="replace-rough",
        formula_template="population * {next_measurement}",
        diagnostic_reasons=("interval_too_wide",),
    )

    candidate = generate_candidates(Parent(), diagnostic, (template,))[0]

    assert candidate.formula == "population * participation_rate"
    assert candidate.variable_names == ("population", "participation_rate")
    assert candidate.generation_method == "declared_template_heuristic"
    assert candidate.as_mapping()["calibrated"] is False


def test_semantic_formula_deduplication_is_deterministic():
    diagnostic = FailureDiagnostic(("interval_too_wide",))
    templates = (
        MutationTemplate("first", "population*rate", ("interval_too_wide",)),
        MutationTemplate("second", "population * rate", ("interval_too_wide",)),
    )

    first = generate_candidates(Parent(), diagnostic, templates)
    second = generate_candidates(Parent(), diagnostic, templates)

    assert len(first) == 1
    assert first[0].candidate_id == second[0].candidate_id
    assert (
        generate_candidates(
            Parent(), diagnostic, templates, existing_formulas=("population * rate",)
        )
        == ()
    )


def test_replaceable_proposer_is_still_labelled_heuristic_and_uncalibrated():
    class StubProposer:
        def propose(self, request):
            assert request.parent_candidate_id == "rough"
            return (
                ProposedMutation(
                    formula="population * rate",
                    mutation_kind="revise",
                    template_id="stub-llm",
                    rationale="proposed by test double",
                    prior_multiplier=0.25,
                ),
            )

    candidate = generate_candidates(
        Parent(),
        FailureDiagnostic(("interval_too_wide",)),
        (),
        proposer=StubProposer(),
    )[0]

    assert candidate.generation_method == "external_proposer_heuristic"
    assert candidate.heuristic is True
    assert candidate.calibrated is False
    assert candidate.prior_weight == 0.2


def test_invalid_template_or_proposer_formula_is_rejected():
    with pytest.raises(ValueError, match="unsupported formula template field"):
        MutationTemplate("bad", "{invented}", ("interval_too_wide",))

    class InvalidProposer:
        def propose(self, request):
            return (ProposedMutation("1 +", "revise", "bad", "invalid"),)

    with pytest.raises(ValueError, match="valid expression"):
        generate_candidates(
            Parent(),
            FailureDiagnostic(("interval_too_wide",)),
            (),
            proposer=InvalidProposer(),
        )
