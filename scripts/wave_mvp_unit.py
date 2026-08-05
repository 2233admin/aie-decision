"""MVP unit system — Dimension, unit table, and conversion helpers."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Minimal dimensional analysis (Pint-style shim).
# ---------------------------------------------------------------------------


class UnitError(ValueError):
    """Raised when a unit string or operation is not representable."""


# Each entry: canonical_name -> (dimension dict, scale to canonical base).
_DIMENSIONLESS: dict[str, int] = {}
_UNIT_TABLE: dict[str, tuple[dict[str, int], float, str]] = {
    "dimensionless": ({}, 1.0, "dimensionless"),
    "1": ({}, 1.0, "dimensionless"),
    "second": ({"time": 1}, 1.0, "second"),
    "s": ({"time": 1}, 1.0, "second"),
    "minute": ({"time": 1}, 60.0, "second"),
    "hour": ({"time": 1}, 3600.0, "second"),
    "h": ({"time": 1}, 3600.0, "second"),
    "day": ({"time": 1}, 86400.0, "second"),
    "d": ({"time": 1}, 86400.0, "second"),
    "meter": ({"length": 1}, 1.0, "meter"),
    "m": ({"length": 1}, 1.0, "meter"),
    "km": ({"length": 1}, 1000.0, "meter"),
    "liter": ({"volume": 1}, 1.0, "liter"),
    "l": ({"volume": 1}, 1.0, "liter"),
    "usd": ({"money": 1}, 1.0, "usd"),
    "$": ({"money": 1}, 1.0, "usd"),
    # Composed units (single source of truth: numerator / denominator).
    "usd/liter": ({"money": 1, "volume": -1}, 1.0, "usd/liter"),
    "usd/hour": ({"money": 1, "time": -1}, 1.0, "usd/hour"),
    "usd/day": ({"money": 1, "time": -1}, 1.0, "usd/day"),
    "km/hour": ({"length": 1, "time": -1}, 1.0 / 3600.0, "meter/second"),
    "liter/hour": ({"volume": 1, "time": -1}, 1.0 / 3600.0, "liter/second"),
    "meter/second": ({"length": 1, "time": -1}, 1.0, "meter/second"),
}


@dataclass(frozen=True, slots=True)
class Dimension:
    """A reduced representation of a Pint-style dimension."""

    exponents: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_unit(cls, unit: str) -> Dimension:
        raw = unit.strip()
        if not raw:
            raise UnitError("unit string is required")
        if raw in _UNIT_TABLE:
            exponents, _scale, _canonical = _UNIT_TABLE[raw]
            return cls(exponents=dict(exponents))
        if "/" in raw:
            num, _, denom = raw.partition("/")
            num_dim = cls.from_unit(num.strip())
            denom_dim = cls.from_unit(denom.strip())
            return num_dim.combine(denom_dim, sign=-1)
        if "*" in raw:
            product = cls()
            for part in raw.split("*"):
                product = product.combine(cls.from_unit(part.strip()), sign=1)
            return product
        raise UnitError(f"unsupported unit: {unit!r}")

    def combine(self, other: Dimension, *, sign: int) -> Dimension:
        exponents: dict[str, int] = dict(self.exponents)
        for key, value in other.exponents.items():
            updated = exponents.get(key, 0) + sign * value
            if updated == 0:
                exponents.pop(key, None)
            else:
                exponents[key] = updated
        return Dimension(exponents=exponents)

    def is_compatible_with(self, other: Dimension) -> bool:
        return self.exponents == other.exponents

    def is_dimensionless(self) -> bool:
        return not self.exponents

    def label(self) -> str:
        if not self.exponents:
            return "dimensionless"
        parts: list[str] = []
        for key in sorted(self.exponents):
            exp = self.exponents[key]
            if exp == 1:
                parts.append(key)
            else:
                parts.append(f"{key}^{exp}")
        return " * ".join(parts)


def _parse_constant(value: Any) -> tuple[float, Dimension]:
    """Convert a JSON constant into ``(scalar, dimension)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnitError("numeric constants must be numbers")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise UnitError("constants must be finite")
    return number, Dimension()


def _convert_to_base(value: float, unit: str) -> tuple[float, Dimension]:
    """Convert ``value`` expressed in ``unit`` to canonical SI/base units."""
    raw = unit.strip()
    if raw not in _UNIT_TABLE:
        raise UnitError(f"unsupported unit: {unit!r}")
    exponents, scale, _canonical = _UNIT_TABLE[raw]
    return value * scale, Dimension(exponents=dict(exponents))


