"""Dimensional closure validation for Fermi contract projections."""

from __future__ import annotations

from typing import Mapping, Sequence

from .fermi_contract_core import DimensionalError, RestrictedExpressionError
from .fermi_expressions import RestrictedExpression, _dimension_key, _evaluate_dimensions, parse_restricted_expression
from .fermi_serialization import _stringify_mapping
from .fermi_units import DEFAULT_UNIT_SYMBOLS, CompoundUnit, parse_compound_unit

def check_dimensional_closure(
    parent_unit: str,
    expression: RestrictedExpression,
    child_units: Sequence[str],
    *,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> tuple[bool, str]:
    """Return ``(True, projected_unit)`` when ``expression`` reproduces
    ``parent_unit`` against ``child_units``; otherwise ``(False, reason)``.

    The function rebinds the expression's variables to ``child_units``
    positionally so the projection can be checked even when the expression
    was parsed without explicit unit bindings.
    """

    if len(expression.variables) != len(child_units):
        return (
            False,
            "expression variables and child_units must align one-to-one",
        )
    bound = {
        variable: parse_compound_unit(unit, registry=registry)
        for variable, unit in zip(expression.variables, child_units)
    }
    try:
        projected = _evaluate_dimensions(expression.tree, bound, registry)
    except RestrictedExpressionError as exc:
        return False, str(exc)
    parsed_parent = parse_compound_unit(parent_unit, registry=registry)
    if projected.to_canonical() != parsed_parent.to_canonical():
        return (
            False,
            f"expression {expression.source!r} projects to "
            f"{_dimension_key(projected)} but parent unit is "
            f"{_dimension_key(parsed_parent)}",
        )
    return True, _dimension_key(parsed_parent)


def project_dimensional_closure(
    parent_unit: str,
    expression_source: str,
    child_units: Sequence[str],
    *,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> RestrictedExpression:
    """Parse a restricted expression and verify it reproduces ``parent_unit``.

    The expression's variables bind positionally to ``child_units``.  A
    mismatched variable count, mismatched compound unit, or unknown label
    raises :class:`DimensionalError`.  Returned expression carries
    dimensions and canonical signature so downstream redundancy checks can
    reuse them.
    """

    parsed = parse_restricted_expression(expression_source, registry=registry)
    if len(parsed.variables) != len(child_units):
        raise DimensionalError(
            f"expression {expression_source!r} references "
            f"{len(parsed.variables)} variable(s) but {len(child_units)} child unit(s) "
            "were declared"
        )
    declared = {
        variable: parse_compound_unit(unit, registry=registry)
        for variable, unit in zip(parsed.variables, child_units)
    }
    try:
        rebuilt = _evaluate_dimensions(parsed.tree, declared, registry)
    except RestrictedExpressionError as exc:
        raise DimensionalError(f"expression fails dimensional closure: {exc}") from exc
    parsed_parent = parse_compound_unit(parent_unit, registry=registry)
    if rebuilt.to_canonical() != parsed_parent.to_canonical():
        raise DimensionalError(
            f"relationship projects to {_dimension_key(rebuilt)} but parent "
            f"declares {_dimension_key(parsed_parent)}"
        )
    return RestrictedExpression(
        source=parsed.source,
        tree=parsed.tree,
        variables=parsed.variables,
        signature=parsed.signature,
        dimensions=rebuilt,
    )


__all__ = [
    "check_dimensional_closure",
    "project_dimensional_closure"
]
