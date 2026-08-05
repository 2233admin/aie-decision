"""Dimension registry and unit-resolution functions for joint-wave schemas.

This module is the **single authority** for unit-to-dimension resolution.
Every unit lookup, dimension comparison, conversion, and normalisation flows
through :func:`dimension_of_unit`.  The registry intentionally covers only
the small set of base units needed by the joint-wave MVP.  Adding a new
dimension is explicit and versioned.

The module is private to :mod:`aie_decision.joint_schema`; public callers
import the re-exported symbols from there.
"""

from __future__ import annotations

from .factor_ir import DIMENSIONLESS


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class JointSchemaError(ValueError):
    """Raised when the joint schema, dimensions, or mappings are invalid."""


class UnknownUnitError(JointSchemaError):
    """Raised when a unit is not in the dimension registry."""

    def __init__(self, unit: object) -> None:
        super().__init__(f"unknown unit: {unit!r}")
        self.unit = unit


class DimensionMismatchError(JointSchemaError):
    """Raised when two units do not share a base dimension."""

    def __init__(self, left_unit: str, right_unit: str, *, operation: str) -> None:
        super().__init__(
            f"dimension mismatch in {operation}: {left_unit!r} vs {right_unit!r}"
        )
        self.left_unit = left_unit
        self.right_unit = right_unit
        self.operation = operation


# ---------------------------------------------------------------------------
# Base dimensions and unit table
# ---------------------------------------------------------------------------

_LENGTH = "length"
_TIME = "time"
_MASS = "mass"
_MONEY = "money"
_COUNT = "count"
_VOLUME = "volume"
_DIMENSIONLESS = DIMENSIONLESS

_KNOWN_DIMENSIONS: frozenset[str] = frozenset(
    {_LENGTH, _TIME, _MASS, _MONEY, _COUNT, _VOLUME, _DIMENSIONLESS}
)

# Each unit maps to (dimension, factor_to_canonical_unit).
# Canonical units: length=m, time=s, mass=kg, money=USD, count=count,
# dimensionless=dimensionless.  Each currency is its own dimension so
# that USD + EUR fails explicitly rather than silently assuming an
# exchange rate.  ``_unit_to_dimension_label`` maps a money-unit to its
# currency-specific dimension.
_MONEY_DIMENSION_PREFIX = "money/"
_UNIT_TABLE: dict[str, tuple[str, float]] = {
    # length
    "m": (_LENGTH, 1.0),
    "meter": (_LENGTH, 1.0),
    "meters": (_LENGTH, 1.0),
    "km": (_LENGTH, 1000.0),
    "kilometer": (_LENGTH, 1000.0),
    "kilometers": (_LENGTH, 1000.0),
    "cm": (_LENGTH, 0.01),
    "mm": (_LENGTH, 0.001),
    "ft": (_LENGTH, 0.3048),
    "foot": (_LENGTH, 0.3048),
    "feet": (_LENGTH, 0.3048),
    "mile": (_LENGTH, 1609.344),
    "miles": (_LENGTH, 1609.344),
    # time
    "s": (_TIME, 1.0),
    "sec": (_TIME, 1.0),
    "second": (_TIME, 1.0),
    "seconds": (_TIME, 1.0),
    "min": (_TIME, 60.0),
    "minute": (_TIME, 60.0),
    "minutes": (_TIME, 60.0),
    "h": (_TIME, 3600.0),
    "hr": (_TIME, 3600.0),
    "hour": (_TIME, 3600.0),
    "hours": (_TIME, 3600.0),
    "day": (_TIME, 86400.0),
    "days": (_TIME, 86400.0),
    "week": (_TIME, 7 * 86400.0),
    "weeks": (_TIME, 7 * 86400.0),
    # mass
    "kg": (_MASS, 1.0),
    "kilo-gram": (_MASS, 1.0),
    "kilogram": (_MASS, 1.0),
    "g": (_MASS, 0.001),
    "gram": (_MASS, 0.001),
    "mg": (_MASS, 1e-6),
    "lb": (_MASS, 0.45359237),
    "pound": (_MASS, 0.45359237),
    "pounds": (_MASS, 0.45359237),
    # money (each currency is its own dimension; no exchange-rate inference)
    "USD": (_MONEY_DIMENSION_PREFIX + "USD", 1.0),
    "usd": (_MONEY_DIMENSION_PREFIX + "USD", 1.0),
    "$": (_MONEY_DIMENSION_PREFIX + "USD", 1.0),
    "CNY": (_MONEY_DIMENSION_PREFIX + "CNY", 1.0),
    "cny": (_MONEY_DIMENSION_PREFIX + "CNY", 1.0),
    "EUR": (_MONEY_DIMENSION_PREFIX + "EUR", 1.0),
    "eur": (_MONEY_DIMENSION_PREFIX + "EUR", 1.0),
    "JPY": (_MONEY_DIMENSION_PREFIX + "JPY", 1.0),
    "jpy": (_MONEY_DIMENSION_PREFIX + "JPY", 1.0),
    # volume
    "l": (_VOLUME, 1.0),
    "liter": (_VOLUME, 1.0),
    "liters": (_VOLUME, 1.0),
    # count
    "count": (_COUNT, 1.0),
    "item": (_COUNT, 1.0),
    "items": (_COUNT, 1.0),
    "unit": (_COUNT, 1.0),
    "units": (_COUNT, 1.0),
    # dimensionless
    "dimensionless": (_DIMENSIONLESS, 1.0),
    "ratio": (_DIMENSIONLESS, 1.0),
    "fraction": (_DIMENSIONLESS, 1.0),
    "pct": (_DIMENSIONLESS, 1.0),
    "percent": (_DIMENSIONLESS, 1.0),
    "%": (_DIMENSIONLESS, 1.0),
    "scalar": (_DIMENSIONLESS, 1.0),
}


def dimension_of_unit(unit: object) -> str:
    """Return the base dimension for ``unit`` or raise :class:`UnknownUnitError`.

    Compound units (e.g. ``usd/liter``, ``km/hour``) are decomposed into
    their constituent base dimensions and returned as a serialized composite
    key of the form ``dim:exp;dim:exp;...``.  Callers that only care about
    simple units remain unaffected; callers that perform dimension algebra
    (e.g. :func:`compile_factor_ir`) parse the composite key to recover the
    per-dimension exponents.
    """

    if not isinstance(unit, str) or not unit.strip():
        raise UnknownUnitError(unit)
    key = unit.strip()
    if key in _UNIT_TABLE:
        return _UNIT_TABLE[key][0]
    # Decompose compound unit.
    # Supported grammar:  <simple-unit> ["/" <simple-unit>]
    # Reject malformed patterns before decomposition.
    _reject_malformed_compound(key)
    numerator, sep, denominator = key.partition("/")
    if not sep:
        raise UnknownUnitError(key)
    num_result = _decompose_term(numerator.strip() or "dimensionless", sign=1)
    den_result = _decompose_term(denominator.strip() or "dimensionless", sign=-1)
    merged: dict[str, int] = dict(num_result)
    for dim, exp in den_result.items():
        merged[dim] = merged.get(dim, 0) + exp
    pruned = _prune_dimensions(merged)
    if not pruned:
        return DIMENSIONLESS
    return _serialize_composite(pruned)


def _decompose_term(term: str, *, sign: int) -> dict[str, int]:
    """Decompose a unit term that may contain ``*``-separated sub-units."""
    result: dict[str, int] = {}
    for sub in term.split("*"):
        sub = sub.strip()
        if not sub:
            continue
        base_dim = _UNIT_TABLE[sub][0] if sub in _UNIT_TABLE else _resolve_composite(sub)
        if base_dim == DIMENSIONLESS:
            continue
        result[base_dim] = result.get(base_dim, 0) + sign
    return result


def _resolve_composite(sub_unit: str) -> str:
    """Resolve a sub-unit that may itself be a composite (recursive)."""
    if sub_unit in _UNIT_TABLE:
        return _UNIT_TABLE[sub_unit][0]
    if "/" in sub_unit or "*" in sub_unit:
        return dimension_of_unit(sub_unit)
    raise UnknownUnitError(sub_unit)


def _prune_dimensions(dims: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in dims.items() if v != 0}


def _serialize_composite(dims: dict[str, int]) -> str:
    """Serialize a dimension->exponent dict as ``dim:exp;dim:exp``."""
    return ";".join(f"{k}:{v}" for k, v in sorted(dims.items()))


def _parse_composite(key: str) -> dict[str, int]:
    """Deserialize a composite dimension key back to a dimension dict.

    Simple (non-compound) keys are returned as ``{key: 1}``.
    """
    if ";" not in key:
        return {key: 1} if key != DIMENSIONLESS else {}
    result: dict[str, int] = {}
    for part in key.split(";"):
        dim, _, exp_str = part.partition(":")
        if not dim or not exp_str:
            raise JointSchemaError(f"invalid composite dimension key: {key!r}")
        result[dim] = int(exp_str)
    return result


def _reject_malformed_compound(key: str) -> None:
    """Reject malformed compound unit strings before decomposition.

    The only supported compound grammar is ``simple "/" simple`` where
    each *simple* term is a recognised unit from ``_UNIT_TABLE``.
    Mixed operators, multiple slashes, leading/trailing slashes, and
    empty terms are rejected with :class:`UnknownUnitError`.
    """
    if "//" in key:
        raise UnknownUnitError(key)
    if key.startswith("/") or key.endswith("/"):
        raise UnknownUnitError(key)
    if "*" in key:
        # The narrow reviewed grammar does not support ``*`` in compound
        # units — only simple ``<unit>/<unit>`` is accepted.
        raise UnknownUnitError(key)
    if key.count("/") > 1:
        raise UnknownUnitError(key)
    # Empty numerator or denominator are also rejected.
    parts = key.split("/")
    for part in parts:
        stripped = part.strip()
        if not stripped:
            raise UnknownUnitError(key)
        if stripped not in _UNIT_TABLE:
            raise UnknownUnitError(key)


parse_composite_dimension = _parse_composite
"""Public alias for :func:`_parse_composite` so callers can decode composite dimension keys."""


def known_dimensions() -> frozenset[str]:
    """Return the frozen set of dimensions accepted by the registry."""
    return _KNOWN_DIMENSIONS


def canonical_unit(dimension: str) -> str:
    """Return the canonical unit string used to record a dimension."""

    if dimension == _LENGTH:
        return "m"
    if dimension == _TIME:
        return "s"
    if dimension == _MASS:
        return "kg"
    if dimension == _COUNT:
        return "count"
    if dimension == _VOLUME:
        return "liter"
    if dimension == _DIMENSIONLESS:
        return "dimensionless"
    if dimension.startswith(_MONEY_DIMENSION_PREFIX):
        return dimension[len(_MONEY_DIMENSION_PREFIX):]
    raise JointSchemaError(f"unknown dimension: {dimension!r}")


def conversion_factor(src: str, dst: str) -> float:
    """Return the scalar that converts a value in ``src`` to ``dst``.

    Both units must belong to the same dimension; cross-dimension
    conversion is rejected with :class:`DimensionMismatchError`.
    """

    left_dimension = dimension_of_unit(src)
    right_dimension = dimension_of_unit(dst)
    if left_dimension != right_dimension:
        raise DimensionMismatchError(src, dst, operation="convert")
    return _UNIT_TABLE[src.strip()][1] / _UNIT_TABLE[dst.strip()][1]


def normalize_to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Return ``value`` expressed in the canonical unit of its dimension."""

    dimension = dimension_of_unit(unit)
    factor = _UNIT_TABLE[unit.strip()][1]
    return value * factor, dimension


def compatible_units(*units: str) -> bool:
    """Return True if every provided unit shares a single dimension."""

    if not units:
        return False
    base = dimension_of_unit(units[0])
    return all(dimension_of_unit(unit) == base for unit in units)
