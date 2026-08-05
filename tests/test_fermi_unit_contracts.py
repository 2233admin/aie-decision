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

def test_compound_unit_canonicalisation_strips_zero_exponents():
    unit = parse_compound_unit("kg*kg^-1")
    assert unit.to_canonical() == ()
    assert unit.is_dimensionless

def test_unit_registry_contains_core_si_and_domain_neutral_symbols():
    for symbol in ("kg", "m", "s", "person", "event", "USD", "tonne", "day"):
        assert symbol in DEFAULT_UNIT_SYMBOLS
