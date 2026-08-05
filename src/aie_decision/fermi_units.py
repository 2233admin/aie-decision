"""Compound-unit parsing and dimensional arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from .fermi_contract_core import DimensionalError, RestrictedExpressionError

DEFAULT_UNIT_SYMBOLS: frozenset[str] = frozenset(
    {
        # SI bases & composites the test suite will exercise.
        "kg",
        "g",
        "t",
        "m",
        "cm",
        "km",
        "mm",
        "s",
        "min",
        "h",
        "day",
        "year",
        "A",
        "K",
        "mol",
        "cd",
        "N",
        "J",
        "W",
        "Pa",
        "Hz",
        "C",
        "V",
        "ohm",
        "F",
        # Domain-neutral counters used by Fermi identities.
        "person",
        "people",
        "household",
        "firm",
        "transaction",
        "event",
        "order",
        "call",
        "visit",
        "trip",
        "tonne",
        "litre",
        "liter",
        "gallon",
        # Money / accounts — domain-neutral.
        "USD",
        "CNY",
        "EUR",
        "JPY",
        "GBP",
        "dollar",
        "dollars",
        "yuan",
        "euro",
        "pound",
        # Generic dimensionless / rate denominators.
        "ratio",
        "share",
        "rate",
        # Conventional dimensionless numerator.
        "1",
    }
)


# ---------------------------------------------------------------------------
# Compound unit arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompoundUnit:
    """A sparse mapping from unit label to integer exponent."""

    exponents: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents or all(value == 0 for value in self.exponents.values())

    def to_canonical(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((label, value) for label, value in self.exponents.items() if value != 0))

    def __str__(self) -> str:  # pragma: no cover - convenience for debugging
        parts = self.to_canonical()
        if not parts:
            return "dimensionless"
        return ", ".join(f"{label}^{exp}" for label, exp in parts)


DIMENSIONLESS = CompoundUnit()


def _normalise_exponents(exponents: Mapping[str, int]) -> dict[str, int]:
    return {label: value for label, value in exponents.items() if value}


def _unit_parser_error(label: str, unit_string: str) -> "RestrictedExpressionError":
    return RestrictedExpressionError(f"{label}: cannot parse {unit_string!r} as a compound unit")


def parse_compound_unit(unit: Any, *, registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS) -> CompoundUnit:
    """Parse a textual compound unit expression into a :class:`CompoundUnit`.

    Accepts ``"/"`` for division, ``"*"`` for multiplication, and ``"^N"``
    or ``"**N"`` for integer powers.  Whitespace is ignored.  Each ``/``
    introduces a single denominator factor; subsequent ``*`` factors are
    added to the numerator again, matching the conventional reading of
    compound units such as ``kg*m/s^2`` (force = mass * acceleration).
    Parens may group sub-expressions.  ``""`` and ``None`` map to
    dimensionless.
    """

    if unit is None or unit == "":
        return DIMENSIONLESS
    if not isinstance(unit, str):
        raise _unit_parser_error("unit", repr(unit))
    cleaned = unit.replace(" ", "")
    if not cleaned:
        return DIMENSIONLESS

    segments = _split_compound_unit(cleaned, "/")
    exponents: dict[str, int] = {}
    for index, segment in enumerate(segments):
        sign = 1 if index == 0 else -1
        for factor_str in _split_compound_unit(segment, "*"):
            factors = _parse_unit_factor(factor_str, registry)
            for sub_label, sub_exp in factors:
                exponents[sub_label] = exponents.get(sub_label, 0) + sign * sub_exp
    return CompoundUnit(_normalise_exponents(exponents))


def _split_compound_unit(text: str, separator: str) -> list[str]:
    if not text:
        return [""]
    pieces: list[str] = []
    depth = 0
    buffer = ""
    for char in text:
        if char == "(":
            depth += 1
            buffer += char
        elif char == ")":
            depth -= 1
            buffer += char
        elif char == separator and depth == 0:
            pieces.append(buffer)
            buffer = ""
        else:
            buffer += char
    pieces.append(buffer)
    return pieces


def _parse_unit_factor(token: str, registry: frozenset[str]) -> list[tuple[str, int]]:
    """Parse a single atomic unit factor (which may itself contain ``*``)."""

    if not token:
        raise _unit_parser_error("unit", token)
    caret = token.find("^")
    stars_index = token.find("**")
    exp_index = -1
    if caret >= 0 and stars_index >= 0:
        exp_index = min(caret, stars_index)
    elif caret >= 0:
        exp_index = caret
    else:
        exp_index = stars_index
    if exp_index >= 0:
        label = token[:exp_index]
        exponent_text = token[exp_index + 1 :] if caret == exp_index else token[exp_index + 2 :]
        try:
            exp = int(exponent_text)
        except ValueError as exc:
            raise _unit_parser_error("unit", token) from exc
    else:
        label = token
        exp = 1
    if not label:
        raise _unit_parser_error("unit", token)
    if "(" in label:
        try:
            inner = label[label.index("(") : label.rindex(")") + 1]
            outer = label.replace(inner, "")
        except ValueError as exc:
            raise _unit_parser_error("unit", token) from exc
        if outer:
            raise _unit_parser_error("unit", token)
        factors = parse_compound_unit(inner[1:-1], registry=registry)
        return [(sub_label, sub_exp * exp) for sub_label, sub_exp in factors.to_canonical()]
    # ``1`` is the conventional dimensionless numerator token; it contributes
    # nothing to the exponent map so rate expressions like ``1/person`` parse
    # cleanly into pure ``person`` dimensions.
    if label == "1":
        return []
    if registry and label not in registry:
        raise _unit_parser_error("unit", token)
    return [(label, exp)]


def _parse_unit_token(token: str, registry: frozenset[str]) -> tuple[str, int]:
    """Backwards-compatible alias used by historical fixtures."""

    factors = _parse_unit_factor(token, registry)
    if len(factors) != 1:
        raise _unit_parser_error("unit", token)
    return factors[0]


def multiply_units(*units: CompoundUnit) -> CompoundUnit:
    combined: dict[str, int] = {}
    for unit in units:
        for label, exp in unit.exponents.items():
            combined[label] = combined.get(label, 0) + exp
    return CompoundUnit(_normalise_exponents(combined))


def divide_units(numerator: CompoundUnit, denominator: CompoundUnit) -> CompoundUnit:
    combined: dict[str, int] = dict(numerator.exponents)
    for label, exp in denominator.exponents.items():
        combined[label] = combined.get(label, 0) - exp
    return CompoundUnit(_normalise_exponents(combined))


def power_units(unit: CompoundUnit, exponent: int) -> CompoundUnit:
    return CompoundUnit(_normalise_exponents({label: exp * exponent for label, exp in unit.exponents.items()}))


def units_close(a: CompoundUnit, b: CompoundUnit) -> bool:
    """Return True iff ``a`` and ``b`` describe the same compound dimension."""

    return a.to_canonical() == b.to_canonical()


__all__ = [
    "CompoundUnit",
    "DIMENSIONLESS",
    "parse_compound_unit",
    "multiply_units",
    "divide_units",
    "power_units",
    "units_close",
    "DEFAULT_UNIT_SYMBOLS"
]
