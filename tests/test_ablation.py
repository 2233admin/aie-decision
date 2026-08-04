import pytest

from aie_decision.ablation import plan_variable_ablations


def test_removes_addition_terms_and_preserves_exact_parent_for_restore():
    formula = " population + adjustment "

    result = plan_variable_ablations(formula, parent_candidate_id="parent")

    assert [
        (item.removed_variable, item.formula, item.operation)
        for item in result.candidates
    ] == [
        ("population", "adjustment", "addition_term"),
        ("adjustment", "population", "addition_term"),
    ]
    assert all(item.parent_formula == formula for item in result.candidates)
    assert all(item.restore_formula == formula for item in result.candidates)
    assert result.rejections == ()


def test_nested_removal_keeps_the_rest_of_the_formula_context():
    result = plan_variable_ablations("(a + b) * c", parent_candidate_id="p")

    assert [(item.removed_variable, item.formula) for item in result.candidates] == [
        ("a", "b * c"),
        ("b", "a * c"),
        ("c", "a + b"),
    ]


def test_removes_only_direct_multiplication_factors():
    result = plan_variable_ablations(
        "base * rate * seasonality", parent_candidate_id="p"
    )

    assert [
        (item.removed_variable, item.formula, item.operation)
        for item in result.candidates
    ] == [
        ("base", "rate * seasonality", "multiplication_factor"),
        ("rate", "base * seasonality", "multiplication_factor"),
        ("seasonality", "base * rate", "multiplication_factor"),
    ]


def test_rejects_subtraction_division_unary_and_root_variables_explicitly():
    result = plan_variable_ablations("root + (a - b) / -c", parent_candidate_id="p")

    assert [(item.removed_variable, item.formula) for item in result.candidates] == [
        ("root", "(a - b) / -c"),
    ]
    assert {item.variable: item.reason for item in result.rejections} == {
        "a": "operator_is_not_safely_ablatable",
        "b": "operator_is_not_safely_ablatable",
        "c": "variable_is_not_a_direct_term_or_factor",
    }


def test_rejects_repeated_variable_instead_of_partially_removing_it():
    result = plan_variable_ablations("a + a * b", parent_candidate_id="p")

    assert [(item.removed_variable, item.formula) for item in result.candidates] == [
        ("b", "a + a"),
    ]
    assert [(item.variable, item.reason) for item in result.rejections] == [
        ("a", "variable_occurs_multiple_times"),
    ]


def test_rejects_a_child_with_no_remaining_variable():
    result = plan_variable_ablations("a * 2", parent_candidate_id="p")

    assert result.candidates == ()
    assert [(item.variable, item.reason) for item in result.rejections] == [
        ("a", "would_remove_last_variable"),
    ]


def test_candidate_ids_are_deterministic_and_parent_scoped():
    first = plan_variable_ablations("a + b", parent_candidate_id="p")
    repeated = plan_variable_ablations("a + b", parent_candidate_id="p")
    other_parent = plan_variable_ablations("a + b", parent_candidate_id="q")

    assert [item.candidate_id for item in first.candidates] == [
        item.candidate_id for item in repeated.candidates
    ]
    assert {item.candidate_id for item in first.candidates}.isdisjoint(
        item.candidate_id for item in other_parent.candidates
    )


@pytest.mark.parametrize("formula", ["f(a) + b", "a ** b", "a and b", "a[0] + b"])
def test_rejects_formula_syntax_outside_the_fermi_arithmetic_subset(formula):
    with pytest.raises(ValueError, match="formula supports only"):
        plan_variable_ablations(formula, parent_candidate_id="p")


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="valid arithmetic"):
        plan_variable_ablations("a +", parent_candidate_id="p")
    with pytest.raises(ValueError, match="parent_candidate_id"):
        plan_variable_ablations("a + b", parent_candidate_id="")
