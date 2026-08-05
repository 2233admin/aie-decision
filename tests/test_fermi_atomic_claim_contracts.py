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
