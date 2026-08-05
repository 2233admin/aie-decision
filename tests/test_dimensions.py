"""Unit tests for the dimension registry in :mod:`aie_decision._dimensions`.

These tests verify dimension resolution, unit conversion, canonical-unit
lookup, and cross-dimension rejection.  The registry is the single authority
for unit-to-dimension mapping; every test that exercises a unit MUST go
through :func:`dimension_of_unit`.
"""

from math import isclose
from unittest import TestCase

from aie_decision.joint_schema import (
    DIMENSIONLESS,
    DimensionMismatchError,
    JointSchemaError,
    UnknownUnitError,
    canonical_unit,
    compatible_units,
    conversion_factor,
    dimension_of_unit,
    known_dimensions,
    normalize_to_canonical,
)


class DimensionRegistryTests(TestCase):
    def test_dimension_of_unit_recognises_base_units(self):
        self.assertEqual(dimension_of_unit("m"), "length")
        self.assertEqual(dimension_of_unit("km"), "length")
        self.assertEqual(dimension_of_unit("hour"), "time")
        self.assertEqual(dimension_of_unit("kg"), "mass")
        self.assertEqual(dimension_of_unit("USD"), "money/USD")
        self.assertEqual(dimension_of_unit("EUR"), "money/EUR")
        self.assertEqual(dimension_of_unit("count"), "count")
        self.assertEqual(dimension_of_unit("dimensionless"), DIMENSIONLESS)

    def test_dimension_of_unit_rejects_unknown(self):
        with self.assertRaises(UnknownUnitError):
            dimension_of_unit("parsec")
        with self.assertRaises(UnknownUnitError):
            dimension_of_unit("")

    def test_known_dimensions_is_stable_frozenset(self):
        self.assertEqual(known_dimensions(), frozenset(
            {"length", "time", "mass", "money", "count", "volume", "dimensionless"}
        ))

    def test_canonical_unit_per_dimension(self):
        self.assertEqual(canonical_unit("length"), "m")
        self.assertEqual(canonical_unit("time"), "s")
        self.assertEqual(canonical_unit("mass"), "kg")
        self.assertEqual(canonical_unit("money/USD"), "USD")
        self.assertEqual(canonical_unit("money/EUR"), "EUR")
        self.assertEqual(canonical_unit("count"), "count")
        self.assertEqual(canonical_unit("dimensionless"), "dimensionless")
        with self.assertRaises(JointSchemaError):
            canonical_unit("force")

    def test_conversion_factor_compatible_units(self):
        self.assertAlmostEqual(conversion_factor("km", "m"), 1000.0)
        self.assertAlmostEqual(conversion_factor("hour", "s"), 3600.0)
        self.assertAlmostEqual(conversion_factor("kg", "g"), 1000.0)
        self.assertAlmostEqual(conversion_factor("USD", "USD"), 1.0)
        self.assertAlmostEqual(conversion_factor("mile", "m"), 1609.344)

    def test_conversion_factor_rejects_cross_dimension(self):
        with self.assertRaises(DimensionMismatchError):
            conversion_factor("hour", "m")
        with self.assertRaises(DimensionMismatchError):
            conversion_factor("USD", "CNY")

    def test_normalize_to_canonical(self):
        self.assertEqual(normalize_to_canonical(3.0, "km"), (3000.0, "length"))
        self.assertEqual(normalize_to_canonical(2.0, "hour"), (7200.0, "time"))

    def test_compatible_units(self):
        self.assertTrue(compatible_units("km", "m"))
        self.assertTrue(compatible_units("USD", "USD"))
        self.assertFalse(compatible_units("hour", "m"))
        self.assertFalse(compatible_units())
        # Money currencies are distinct dimensions for safety.
        self.assertFalse(compatible_units("USD", "EUR"))


class DimensionDirectImportTests(TestCase):
    """Verify that callers can import dimension symbols directly from
    :mod:`aie_decision._dimensions` — the internal module exposes the
    same public API that :mod:`aie_decision.joint_schema` re-exports."""

    def test_direct_import_dimension_of_unit(self):
        from aie_decision._dimensions import dimension_of_unit as direct_du
        from aie_decision.joint_schema import dimension_of_unit as facade_du
        self.assertIs(direct_du, facade_du,
                      "direct and facade dimension_of_unit must be the same function")

    def test_direct_import_errors(self):
        from aie_decision._dimensions import JointSchemaError as direct_js
        from aie_decision.joint_schema import JointSchemaError as facade_js
        self.assertIs(direct_js, facade_js)
        self.assertTrue(issubclass(direct_js, ValueError))

    def test_direct_import_all_public_symbols(self):
        """Every public dimension symbol exported by joint_schema must be
        directly importable from _dimensions."""
        public_symbols = [
            "DIMENSIONLESS",
            "DimensionMismatchError",
            "JointSchemaError",
            "UnknownUnitError",
            "canonical_unit",
            "compatible_units",
            "conversion_factor",
            "dimension_of_unit",
            "known_dimensions",
            "normalize_to_canonical",
            "parse_composite_dimension",
        ]
        import aie_decision._dimensions as dim
        for name in public_symbols:
            self.assertTrue(hasattr(dim, name),
                            f"_dimensions must export {name}")
