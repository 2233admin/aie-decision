"""Track A contracts-level tests.

These tests cover every Track A OpenSpec scenario that lives purely at
the contracts layer: raw-question roots, restricted arithmetic parsing,
compound-unit parsing and dimensional closure, redundant-alternative
signatures, and atomic-claim validation including abstract-label
rejection.
"""

from __future__ import annotations

import ast
from dataclasses import replace

import pytest

from aie_decision.fermi_contracts import (
    DEFAULT_UNIT_SYMBOLS,
    AtomicClaim,
    AtomicClaimError,
    Branch,
    CompoundUnit,
    DIMENSIONLESS,
    DimensionalError,
    Expansion,
    FermiContractError,
    Gap,
    MeasurementKind,
    Node,
    NodeRole,
    NodeStatus,
    ObservationKind,
    Question,
    QuestionStatus,
    RedundancyReason,
    Relationship,
    RestrictedExpression,
    RestrictedExpressionError,
    Scope,
    check_dimensional_closure,
    divide_units,
    evaluate_restricted_expression,
    expressions_are_equivalent,
    multiply_units,
    parse_compound_unit,
    parse_restricted_expression,
    power_units,
    project_dimensional_closure,
    units_close,
    validate_atomic_claim,
)


# ---------------------------------------------------------------------------
# Raw question roots
# ---------------------------------------------------------------------------


def _scope() -> Scope:
    return Scope(population="U.S. commuters", geography="United States")


def _question(**overrides) -> Question:
    payload = {
        "question_id": "q-commutes",
        "question": "How many one-way commutes occur on a U.S. weekday?",
        "target_subject": "U.S. weekday commutes",
        "target_measure": "count of one-way commutes",
        "unit": "person/day",
        "time_basis": "per weekday in 2025",
        "scope": _scope(),
    }
    payload.update(overrides)
    return Question(**payload)


def test_raw_question_root_records_every_required_target_field():
    question = _question()

    assert question.question_id == "q-commutes"
    assert question.target_measure == "count of one-way commutes"
    assert question.target_subject == "U.S. weekday commutes"
    assert question.unit == "person/day"
    assert question.time_basis == "per weekday in 2025"
    assert question.scope.geography == "United States"
    assert question.scope.is_well_defined()
    assert question.is_minimally_complete()
    assert question.status is QuestionStatus.OPEN


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_subject": "  "}, "target_subject"),
        ({"target_measure": ""}, "target_measure"),
        ({"unit": ""}, "unit"),
        ({"time_basis": "  "}, "time_basis"),
        ({"question": ""}, "question"),
        ({"question_id": ""}, "question_id"),
    ],
)
def test_question_without_a_required_field_is_rejected(overrides, match):
    with pytest.raises(FermiContractError, match=match):
        _question(**overrides)


def test_question_with_unresolved_scope_fields_keeps_gap_open():
    question = _question(scope=Scope(population="  ", geography="  "))

    assert question.is_minimally_complete() is False
    assert "scope anchors" not in question.unresolved_fields
    assert question.with_unresolved(("scope",)).unresolved_fields == ("scope",)


# ---------------------------------------------------------------------------
# Scope, node, gap records
# ---------------------------------------------------------------------------


def test_scope_is_only_well_defined_with_a_real_anchor():
    assert Scope(geography="Tokyo").is_well_defined()
    assert Scope(population="all people").is_well_defined()
    assert Scope(geography="  ", population="  ").is_well_defined() is False
    assert Scope().is_well_defined() is False


def test_node_rejects_empty_label():
    with pytest.raises(FermiContractError, match="label"):
        Node(node_id="n_0002", label="", role=NodeRole.CHILD)


def test_node_cannot_assign_parent_to_the_target_role():
    with pytest.raises(FermiContractError, match="target node cannot have a parent"):
        Node(node_id="n_0001", label="root", role=NodeRole.TARGET, parent_id="n_0000")


def test_gap_rejects_blank_explanation():
    with pytest.raises(FermiContractError, match="explanation"):
        Gap(gap_id="gap_0001", kind=NodeStatus.UNRESOLVED, target="n_0002", explanation="   ")


# ---------------------------------------------------------------------------
# Compound unit arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kg*m/s^2", (("kg", 1), ("m", 1), ("s", -2))),
        ("m/s", (("m", 1), ("s", -1))),
        ("m^2", (("m", 2),)),
        ("(kg*m)/s^2", (("kg", 1), ("m", 1), ("s", -2))),
        ("person/day", (("day", -1), ("person", 1))),
        ("1/(person*day)", (("day", -1), ("person", -1))),
        ("tonne/year", (("tonne", 1), ("year", -1))),
        ("USD/person", (("USD", 1), ("person", -1))),
        ("", ()),
        ("ratio", (("ratio", 1),)),
    ],
)
def test_parse_compound_unit_handles_real_fermi_identities(text, expected):
    assert parse_compound_unit(text).to_canonical() == expected


def test_parse_compound_unit_rejects_unknown_labels_by_default():
    with pytest.raises(RestrictedExpressionError, match="commute"):
        parse_compound_unit("commute/day")


def test_parse_compound_unit_accepts_unknown_labels_with_empty_registry():
    unit = parse_compound_unit("commute/day", registry=frozenset())
    assert unit.to_canonical() == (("commute", 1), ("day", -1))


def test_compound_unit_arithmetic_combines_exponents():
    mass = parse_compound_unit("kg")
    acceleration = parse_compound_unit("m/s^2")
    force = multiply_units(mass, acceleration)
    assert force.to_canonical() == (("kg", 1), ("m", 1), ("s", -2))
    assert units_close(force, parse_compound_unit("kg*m/s^2"))
    assert not units_close(force, parse_compound_unit("kg*m"))


def test_divide_and_power_apply_exponent_signs():
    velocity = divide_units(parse_compound_unit("m"), parse_compound_unit("s"))
    assert velocity.to_canonical() == (("m", 1), ("s", -1))
    area = power_units(parse_compound_unit("m"), 2)
    assert area.to_canonical() == (("m", 2),)


def test_units_close_is_label_strict():
    # ``N`` is not aliased to ``kg*m/s^2`` because the registry stays
    # domain-neutral; the function is intentionally label-strict.
    assert units_close(parse_compound_unit("kg*m/s^2"), parse_compound_unit("kg*m/s^2"))
    assert not units_close(parse_compound_unit("kg*m/s^2"), parse_compound_unit("N"))


# ---------------------------------------------------------------------------
# Restricted arithmetic parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "a + b",
        "a - b",
        "a * b",
        "a / b",
        "a * b + c",
        "a * (b + c)",
        "a^2 + b^2",
        "-a + b",
        "+1.5 * a",
        "(a + b) * (c - d)",
    ],
)
def test_parse_restricted_expression_accepts_pure_arithmetic(expression):
    parsed = parse_restricted_expression(expression)
    assert isinstance(parsed, RestrictedExpression)
    assert parsed.variables
    assert parsed.signature


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "foo()",
        "lambda x: x",
        "[a for a in b]",
        "a if a > 0 else b",
        "a and b",
        "a == b",
        "a[0]",
        "a.b",
        "True",
        "None",
    ],
)
def test_parse_restricted_expression_rejects_anything_beyond_arithmetic(expression):
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression(expression)


def test_parse_restricted_expression_accepts_caret_power_notation():
    parsed = parse_restricted_expression("a ^ 2")
    assert "Pow" in ast_dump(parsed.tree)


def test_parse_restricted_expression_rejects_dunder_names():
    with pytest.raises(RestrictedExpressionError, match="dunder"):
        parse_restricted_expression("__class__")


def test_parse_restricted_expression_rejects_non_numeric_or_non_finite_constants():
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression("a + 'b'")
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression("a + True")
    with pytest.raises(RestrictedExpressionError):
        parse_restricted_expression("a + 1e400 * 1e400")


def test_parse_restricted_expression_records_variable_count():
    parsed = parse_restricted_expression("a * b * c / d")
    assert sorted(parsed.variables) == ["a", "b", "c", "d"]


def test_parse_restricted_expression_variables_follow_source_order():
    # ``ast.walk`` visits in BFS order and would yield ``b`` before ``a``
    # when the variable is nested inside a sub-expression on the right;
    # the project must collect variables in textual source order so that
    # positional unit bindings line up with the relationship's child list.
    parsed = parse_restricted_expression("b * a")
    assert list(parsed.variables) == ["b", "a"]


def test_parse_restricted_expression_dimension_evaluation_requires_unit_bindings():
    parsed = parse_restricted_expression(
        "a * b",
        variable_units={
            "a": parse_compound_unit("kg"),
            "b": parse_compound_unit("m/s^2"),
        },
    )
    assert parsed.dimensions.to_canonical() == (("kg", 1), ("m", 1), ("s", -2))


def test_parse_restricted_expression_rejects_undeclared_variables_when_units_pinned():
    with pytest.raises(RestrictedExpressionError, match="undeclared"):
        parse_restricted_expression("a * b", variable_units={"a": DIMENSIONLESS})


def test_evaluate_restricted_expression_returns_numeric_value():
    parsed = parse_restricted_expression("a * b + c")
    assert evaluate_restricted_expression(parsed, {"a": 2.0, "b": 3.0, "c": 4.0}) == 10.0


def test_evaluate_restricted_expression_rejects_division_by_zero():
    parsed = parse_restricted_expression("a / b")
    with pytest.raises(RestrictedExpressionError, match="division by zero"):
        evaluate_restricted_expression(parsed, {"a": 1.0, "b": 0.0})


def test_evaluate_restricted_expression_rejects_non_numeric_value():
    parsed = parse_restricted_expression("a + 1")
    with pytest.raises(RestrictedExpressionError, match="non-numeric"):
        evaluate_restricted_expression(parsed, {"a": "x"})


def test_evaluate_restricted_expression_rejects_non_integer_exponent():
    parsed = parse_restricted_expression("a ^ 1.5")
    with pytest.raises(RestrictedExpressionError, match="integer constant"):
        evaluate_restricted_expression(parsed, {"a": 2.0})


# ---------------------------------------------------------------------------
# Dimensional closure
# ---------------------------------------------------------------------------


def test_project_dimensional_closure_accepts_a_compatible_expression():
    projection = project_dimensional_closure("kg*m/s^2", "a * b", ["kg", "m/s^2"])
    assert projection.dimensions.to_canonical() == (("kg", 1), ("m", 1), ("s", -2))


def test_project_dimensional_closure_rejects_unit_mismatch():
    with pytest.raises(DimensionalError, match="closure"):
        project_dimensional_closure("kg*m/s^2", "a + b", ["kg", "m"])


def test_project_dimensional_closure_rejects_mismatched_variable_count():
    with pytest.raises(DimensionalError, match="references"):
        project_dimensional_closure("person/day", "a * b * c", ["person", "day"])


def test_check_dimensional_closure_returns_true_when_projection_matches_parent():
    parsed = parse_restricted_expression("a * b")
    ok, key = check_dimensional_closure("kg*m/s^2", parsed, ("kg", "m/s^2"))
    assert ok is True
    assert "kg" in key and "m" in key


def test_check_dimensional_closure_reports_incompatible_projection():
    parsed = parse_restricted_expression("a * b")
    ok, key = check_dimensional_closure("kg*m/s", parsed, ("kg", "m/s^2"))
    assert ok is False
    assert "kg" in key


# ---------------------------------------------------------------------------
# Atomic claim validation
# ---------------------------------------------------------------------------


def _atomic_claim(**overrides) -> AtomicClaim:
    payload = {
        "node_id": "n_0002",
        "target_object": "number of people entering the bank lobby",
        "unit": "person/day",
        "scope": Scope(population="Boston downtown branches", geography="Boston, MA"),
        "time_basis": "per weekday in 2025",
        "measurement_kind": MeasurementKind.COUNT,
        "source": "bank lobby door sensor log",
        "procedure": (
            "Each weekday the lobby door sensor records a row per entry; "
            "sum the rows to obtain the count of people entering."
        ),
        "observation_kind": ObservationKind.OBSERVED,
    }
    payload.update(overrides)
    return AtomicClaim(**payload)


def test_concrete_atom_is_accepted_with_structured_measurement_fields():
    errors = validate_atomic_claim(_atomic_claim())
    assert errors == ()


def test_chinese_measurement_description_is_accepted_without_english_vocabulary():
    """Operational atomicity must be language-neutral.

    The same structured record expressed entirely in Chinese must pass
    validation, because acceptance is decided by field presence and
    :class:`MeasurementKind` membership, not by English keyword matching.
    """

    claim = AtomicClaim(
        node_id="n_0002",
        target_object="每日进入银行大堂的客户数量",
        unit="person/day",
        scope=Scope(population="波士顿市中心分行", geography="Boston, MA"),
        time_basis="2025年每个工作日",
        measurement_kind=MeasurementKind.COUNT,
        source="大堂门禁传感器日志",
        procedure="读取门禁传感器每日记录，逐行汇总得到当日进入大堂的人数。",
        observation_kind=ObservationKind.OBSERVED,
    )
    errors = validate_atomic_claim(claim)
    assert errors == ()


def test_arabic_measurement_description_is_accepted_without_english_vocabulary():
    claim = AtomicClaim(
        node_id="n_0002",
        target_object="عدد الأشخاص الذين يدخلون بهو البنك يوميًا",
        unit="person/day",
        scope=Scope(population="فروع بوسطن", geography="Boston, MA"),
        time_basis="يوميًا في 2025",
        measurement_kind=MeasurementKind.COUNT,
        source="سجل حساس باب البنك",
        procedure="اجمع الصفوف اليومية من سجل حساس الباب لكل دخول.",
        observation_kind=ObservationKind.OBSERVED,
    )
    assert validate_atomic_claim(claim) == ()


def test_atom_with_unknown_observation_kind_is_still_structurally_admissible():
    """Honest evidence gaps do not invalidate the structural contract."""

    claim = _atomic_claim(observation_kind=ObservationKind.UNKNOWN)
    assert validate_atomic_claim(claim) == ()
    assert claim.has_structured_measurement() is True


def test_atom_missing_measurement_kind_is_rejected():
    claim = _atomic_claim()
    with pytest.raises(AtomicClaimError, match="measurement_kind"):
        replace(claim, measurement_kind=None)  # type: ignore[arg-type]


def test_atom_without_source_is_rejected():
    errors = validate_atomic_claim(_atomic_claim(source=""))
    assert any("source is required" in err for err in errors)


def test_atom_without_procedure_is_rejected():
    errors = validate_atomic_claim(_atomic_claim(procedure=""))
    assert any("procedure is required" in err for err in errors)


def test_atom_without_target_object_is_rejected():
    errors = validate_atomic_claim(_atomic_claim(target_object=""))
    assert any("target_object is required" in err for err in errors)


def test_atom_without_unit_is_rejected():
    errors = validate_atomic_claim(_atomic_claim(unit=""))
    assert any("unit is required" in err for err in errors)


def test_atom_without_scope_anchor_is_rejected():
    errors = validate_atomic_claim(
        _atomic_claim(scope=Scope(population="  ", geography="  ")),
    )
    assert any("scope must declare" in err for err in errors)


def test_timed_measurement_requires_time_basis():
    errors = validate_atomic_claim(
        _atomic_claim(
            measurement_kind=MeasurementKind.TIMED_MEASUREMENT,
            time_basis="",
        )
    )
    assert any("time_basis is required" in err for err in errors)


def test_derived_proxy_requires_assumption_notes():
    errors = validate_atomic_claim(
        _atomic_claim(
            measurement_kind=MeasurementKind.DERIVED_PROXY,
            assumption_notes="",
        )
    )
    assert any("assumption_notes are required" in err for err in errors)


def test_derived_proxy_with_assumption_notes_is_accepted():
    claim = _atomic_claim(
        measurement_kind=MeasurementKind.DERIVED_PROXY,
        assumption_notes="Mapped from analogous cohort; uncertainty noted.",
    )
    assert validate_atomic_claim(claim) == ()


def test_measurement_kind_enum_lists_explicit_kinds():
    expected = {
        MeasurementKind.DIRECT_OBSERVATION,
        MeasurementKind.RECORD_LOOKUP,
        MeasurementKind.COUNT,
        MeasurementKind.TIMED_MEASUREMENT,
        MeasurementKind.INSTRUMENT_MEASUREMENT,
        MeasurementKind.DERIVED_PROXY,
    }
    assert set(MeasurementKind) == expected


def test_atomic_claim_normalises_unit_string_at_construction():
    claim = _atomic_claim(unit="kg*m/s^2")
    errors = validate_atomic_claim(claim)
    assert errors == ()


def test_atomic_claim_rejects_unknown_unit_label():
    with pytest.raises((AtomicClaimError, RestrictedExpressionError)):
        AtomicClaim(
            node_id="n_0099",
            target_object="rainfall",
            unit="inches/month",
            scope=Scope(population="California", geography="California"),
            time_basis="per month",
            measurement_kind=MeasurementKind.INSTRUMENT_MEASUREMENT,
            source="weather station log",
            procedure="read the weather station log",
        )


def test_relationship_record_requires_parallel_child_units():
    with pytest.raises(FermiContractError, match="parallel tuples"):
        Relationship(
            relationship_id="rel_0001",
            parent_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 * _child_1",
            child_node_ids=("n_0002", "n_0003"),
            child_units=("person",),
            rationale="ratio",
        )


# ---------------------------------------------------------------------------
# Redundant alternative signatures
# ---------------------------------------------------------------------------


def test_expressions_are_equivalent_under_commutative_canonicalisation():
    left = parse_restricted_expression("a * b")
    right = parse_restricted_expression("b * a")
    assert expressions_are_equivalent(left, right)


def test_expressions_are_not_equivalent_when_variables_differ():
    left = parse_restricted_expression("a * b")
    right = parse_restricted_expression("a * c")
    assert expressions_are_equivalent(left, right) is False


def test_expressions_are_not_equivalent_when_operator_differs():
    left = parse_restricted_expression("a * b")
    right = parse_restricted_expression("a + b")
    assert expressions_are_equivalent(left, right) is False


def test_expressions_with_distinct_canonical_signatures_compare_on_signature():
    a = parse_restricted_expression("(a + b) * c")
    b = parse_restricted_expression("c * (a + b)")
    assert expressions_are_equivalent(a, b)


def test_expansion_record_can_describe_itself():
    expansion = Expansion(
        expansion_id="exp_0001",
        target_node_id="n_0001",
        relationship_id="rel_0001",
        parent_unit="person/day",
        projected_unit="person*day^-1",
        child_node_ids=("n_0002", "n_0003"),
        rationale="ratio",
    )
    assert "exp_0001" in expansion.describe()
    assert "n_0001" in expansion.describe()


def test_branch_record_covers_named_expansion():
    branch = Branch(
        branch_id="br_0001",
        root_question_id="q-commutes",
        expansion_ids=("exp_0001", "exp_0002"),
    )
    assert branch.covers("exp_0001")
    assert not branch.covers("exp_0003")


def test_redundancy_reason_is_a_string_subclass_for_safe_logging():
    reason = RedundancyReason("matches exp_0001")
    assert str(reason) == "matches exp_0001"


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


def test_compound_unit_canonicalisation_strips_zero_exponents():
    unit = parse_compound_unit("kg*kg^-1")
    assert unit.to_canonical() == ()
    assert unit.is_dimensionless


def test_unit_registry_contains_core_si_and_domain_neutral_symbols():
    for symbol in ("kg", "m", "s", "person", "event", "USD", "tonne", "day"):
        assert symbol in DEFAULT_UNIT_SYMBOLS


def ast_dump(tree):
    return ast.dump(tree)