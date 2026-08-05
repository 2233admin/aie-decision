from dataclasses import FrozenInstanceError
from math import isclose
from unittest import TestCase

from aie_decision.factor_ir import FactorIRError
from aie_decision.joint_schema import (
    DIMENSIONLESS,
    EVIDENCE_SCHEMA_VERSION,
    JOINT_SCHEMA_VERSION,
    CompiledMapping,
    DimensionMismatchError,
    EpistemicType,
    Evidence,
    JointSchemaError,
    MappingKind,
    MappingSpec,
    OutcomeAxis,
    OutcomeSpace,
    UnknownUnitError,
    VariableSpec,
    VariableStatus,
    canonical_unit,
    compatible_units,
    compile_joint_schema,
    conversion_factor,
    dimension_of_unit,
    known_dimensions,
    normalize_to_canonical,
)


def _axis(axis_id: str = "delivery", unit: str = "hour", **kwargs) -> OutcomeAxis:
    return OutcomeAxis(
        schema_version=JOINT_SCHEMA_VERSION,
        axis_id=axis_id,
        name=kwargs.pop("name", axis_id),
        unit=unit,
        domain=kwargs.pop("domain", (0.0, 48.0)),
        time_semantics=kwargs.pop("time_semantics", "elapsed"),
        absolute_tolerance=kwargs.pop("absolute_tolerance", None),
        reference_value=kwargs.pop("reference_value", None),
        decision_useful=kwargs.pop("decision_useful", True),
    )


def _variable(name: str, unit: str = "hour", **kwargs) -> VariableSpec:
    return VariableSpec(
        schema_version=JOINT_SCHEMA_VERSION,
        name=name,
        unit=unit,
        status=kwargs.pop("status", VariableStatus.OBSERVED),
        lower=kwargs.pop("lower", 1.0),
        upper=kwargs.pop("upper", 5.0),
        method=kwargs.pop("method", "observed"),
        evidence_atom_id=kwargs.pop("evidence_atom_id", None),
        time_semantics=kwargs.pop("time_semantics", None),
    )


def _mapping(mapping_id: str, expression: str, variables: tuple[str, ...], **kwargs) -> MappingSpec:
    return MappingSpec(
        schema_version=JOINT_SCHEMA_VERSION,
        mapping_id=mapping_id,
        kind=kwargs.pop("kind", MappingKind.FORMULA),
        variables=variables,
        result_axis=kwargs.pop("result_axis", "delivery"),
        expression=expression,
        observation=kwargs.pop("observation", None),
        observation_scale=kwargs.pop("observation_scale", None),
        evidence_atom_ids=kwargs.pop("evidence_atom_ids", ()),
        direction=kwargs.pop("direction", "support"),
        applicability=kwargs.pop("applicability", None),
    )


def _evidence(atom_id: str = "atom-1") -> Evidence:
    return Evidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        evidence_atom_id=atom_id,
        claim="source observation",
        epistemic_type=EpistemicType.PRIMARY_RECORD,
        source_id="source-1",
        truth_confidence=0.6,
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
        # Money currencies are recorded with their own labels; the registry
        # now includes volume (added for golden-fixture compound-unit support).
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


class OutcomeAxisTests(TestCase):
    def test_axis_requires_recognised_unit(self):
        with self.assertRaises(UnknownUnitError):
            _axis(unit="parsec")

    def test_axis_requires_unique_ids(self):
        a1 = _axis("delivery")
        a2 = _axis("delivery", unit="day")
        with self.assertRaisesRegex(JointSchemaError, "unique"):
            OutcomeSpace(
                schema_version=JOINT_SCHEMA_VERSION,
                question_id="q1",
                axes=(a1, a2),
            )

    def test_axis_domain_must_be_ordered(self):
        with self.assertRaisesRegex(JointSchemaError, "ordered"):
            _axis(domain=(10.0, 1.0))

    def test_axis_domain_must_be_finite(self):
        with self.assertRaisesRegex(JointSchemaError, "finite"):
            _axis(domain=(float("-inf"), 1.0))

    def test_axis_is_frozen(self):
        axis = _axis()
        with self.assertRaises(FrozenInstanceError):
            axis.name = "changed"  # type: ignore[misc]


class VariableSpecTests(TestCase):
    def test_variable_requires_bounds_when_not_missing(self):
        with self.assertRaisesRegex(JointSchemaError, "lower and upper"):
            _variable("a", lower=None, upper=None)

    def test_variable_missing_cannot_declare_bounds(self):
        with self.assertRaisesRegex(JointSchemaError, "missing"):
            _variable("a", status=VariableStatus.MISSING, lower=1.0, upper=2.0)

    def test_variable_bounds_must_be_ordered(self):
        with self.assertRaisesRegex(JointSchemaError, "must not exceed"):
            _variable("a", lower=5.0, upper=1.0)

    def test_variable_unit_must_be_known(self):
        with self.assertRaises(UnknownUnitError):
            _variable("a", unit="parsec")


class EvidenceTests(TestCase):
    def test_evidence_requires_atom_id(self):
        with self.assertRaisesRegex(JointSchemaError, "evidence_atom_id"):
            Evidence(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                evidence_atom_id="",
                claim="x",
                epistemic_type=EpistemicType.PRIMARY_RECORD,
                source_id="s1",
            )

    def test_evidence_requires_claim(self):
        with self.assertRaisesRegex(JointSchemaError, "claim"):
            Evidence(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                evidence_atom_id="a1",
                claim="   ",
                epistemic_type=EpistemicType.PRIMARY_RECORD,
                source_id="s1",
            )

    def test_evidence_truth_confidence_range(self):
        with self.assertRaisesRegex(JointSchemaError, "truth_confidence"):
            Evidence(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                evidence_atom_id="a1",
                claim="x",
                epistemic_type=EpistemicType.PRIMARY_RECORD,
                source_id="s1",
                truth_confidence=1.5,
            )


class MappingSpecTests(TestCase):
    def test_formula_mapping_requires_expression(self):
        with self.assertRaisesRegex(JointSchemaError, "FORMULA mapping requires expression"):
            MappingSpec(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping_id="m",
                kind=MappingKind.FORMULA,
                variables=("a",),
                result_axis="delivery",
                expression="",
            )

    def test_likelihood_requires_one_variable(self):
        with self.assertRaisesRegex(JointSchemaError, "exactly one variable"):
            MappingSpec(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping_id="m",
                kind=MappingKind.LIKELIHOOD,
                variables=("a", "b"),
                result_axis="delivery",
                observation=(1.0, 2.0),
                observation_scale=1.0,
            )

    def test_likelihood_requires_observation(self):
        with self.assertRaisesRegex(JointSchemaError, "observation"):
            MappingSpec(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping_id="m",
                kind=MappingKind.LIKELIHOOD,
                variables=("a",),
                result_axis="delivery",
                observation=None,
                observation_scale=1.0,
            )

    def test_likelihood_rejects_zero_scale(self):
        with self.assertRaisesRegex(JointSchemaError, "observation_scale"):
            MappingSpec(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping_id="m",
                kind=MappingKind.LIKELIHOOD,
                variables=("a",),
                result_axis="delivery",
                observation=(1.0, 2.0),
                observation_scale=0.0,
            )

    def test_direction_must_be_support_or_penalty(self):
        with self.assertRaisesRegex(JointSchemaError, "direction"):
            _mapping("m", "a + 1", ("a",), direction="unknown")


class CompileTests(TestCase):
    def test_compile_minimal_valid_schema(self):
        axis = _axis("delivery", unit="dimensionless")
        variable = _variable("lane_hours", unit="hour", lower=18.0, upper=22.0)
        # ``lane_hours / lane_hours`` is dimensionless and unit conversion
        # is a no-op for a ratio; it is the canonical happy path.
        mapping = _mapping("leg", "lane_hours / lane_hours", ("lane_hours",))
        compiled = compile_joint_schema(
            question_id="q1",
            axes=(axis,),
            variables=(variable,),
            mappings=(mapping,),
        )
        self.assertEqual(len(compiled), 1)
        entry = compiled[0]
        self.assertEqual(entry.schema_version, JOINT_SCHEMA_VERSION)
        self.assertEqual(entry.variable_dimensions, {"lane_hours": "time"})
        self.assertEqual(entry.factor_ir.mapping_id, "leg")
        self.assertTrue(isclose(entry.log_potential({"lane_hours": 20.0}), 1.0))

    def test_compile_normalises_legal_unit_conversion(self):
        axis = _axis("ratio", unit="dimensionless")
        var_km = _variable("route_km", unit="km", lower=1.0, upper=5.0)
        var_m = _variable("route_m", unit="m", lower=100.0, upper=500.0)
        # 1 km = 1000 m; the result axis is dimensionless because the
        # formula cancels the length dimension.  The evaluator must convert
        # both inputs to metres before evaluating.
        mapping = _mapping(
            "ratio",
            "route_km / route_m",
            ("route_km", "route_m"),
            result_axis="ratio",
        )
        compiled = compile_joint_schema(
            question_id="q1",
            axes=(axis,),
            variables=(var_km, var_m),
            mappings=(mapping,),
        )
        entry = compiled[0]
        self.assertTrue(isclose(entry.log_potential({"route_km": 1.0, "route_m": 100.0}), 10.0))

    def test_compile_dimensional_formula_adapted_via_proxy(self):
        # A dimensionally valid but dimensional formula (e.g. same-dimension
        # addition) is adapted through a dimensionless proxy so that the
        # FactorIR dimensionless-output invariant is preserved.  The proxy
        # ``(expr) / (expr)`` cancels the output dimension while still
        # validating that every internal operation is dimensionally sound.
        axis = _axis("delivery", unit="hour")
        variable = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping("leg", "lane_hours + lane_hours", ("lane_hours",))
        compiled = compile_joint_schema(
            question_id="q1",
            axes=(axis,),
            variables=(variable,),
            mappings=(mapping,),
        )
        self.assertEqual(len(compiled), 1)
        ir = compiled[0].factor_ir
        self.assertIsNotNone(ir)
        # The proxy is dimensionless (FactorIR invariant preserved).
        self.assertEqual(ir.output_dimension, ())

    def test_compile_rejects_cross_dimension_arithmetic(self):
        axis = _axis("delivery", unit="hour")
        var_time = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        var_money = _variable("fee", unit="USD", lower=1.0, upper=5.0)
        mapping = _mapping("bad", "lane_hours + fee", ("lane_hours", "fee"))
        with self.assertRaises(JointSchemaError):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(var_time, var_money),
                mappings=(mapping,),
            )

    def test_compile_rejects_unknown_axis_reference(self):
        axis = _axis("delivery", unit="hour")
        variable = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping("m", "lane_hours / 1", ("lane_hours",), result_axis="unknown")
        with self.assertRaisesRegex(JointSchemaError, "unknown axis"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(variable,),
                mappings=(mapping,),
            )

    def test_compile_rejects_unknown_variable_reference(self):
        axis = _axis("delivery", unit="hour")
        variable = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping("m", "unknown + 1", ("unknown",))
        with self.assertRaisesRegex(JointSchemaError, "unknown variables"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(variable,),
                mappings=(mapping,),
            )

    def test_compile_rejects_duplicate_mapping_ids(self):
        axis = _axis("delivery", unit="dimensionless")
        variable = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping("dup", "lane_hours / lane_hours", ("lane_hours",))
        with self.assertRaisesRegex(JointSchemaError, "unique"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(variable,),
                mappings=(mapping, mapping),
            )

    def test_compile_rejects_duplicate_variable_names(self):
        axis = _axis("delivery", unit="hour")
        v1 = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        v2 = _variable("lane_hours", unit="hour", lower=2.0, upper=8.0)
        with self.assertRaisesRegex(JointSchemaError, "VariableSpec.name"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(v1, v2),
                mappings=(),
            )

    def test_compile_rejects_duplicate_axis_ids(self):
        a1 = _axis("delivery", unit="hour")
        a2 = _axis("delivery", unit="day")
        v1 = _variable("a", unit="hour", lower=1.0, upper=5.0)
        with self.assertRaisesRegex(JointSchemaError, "axis_id"):
            compile_joint_schema(
                question_id="q1",
                axes=(a1, a2),
                variables=(v1,),
                mappings=(),
            )

    def test_compile_requires_axes_and_variables(self):
        with self.assertRaisesRegex(JointSchemaError, "axes"):
            compile_joint_schema(
                question_id="q1",
                axes=(),
                variables=(_variable("a"),),
                mappings=(),
            )
        with self.assertRaisesRegex(JointSchemaError, "variables"):
            compile_joint_schema(
                question_id="q1",
                axes=(_axis("delivery"),),
                variables=(),
                mappings=(),
            )

    def test_compile_likelihood_dimension_must_match_axis(self):
        axis = _axis("magnitude", unit="dimensionless")
        var_time = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        likelihood = MappingSpec(
            schema_version=JOINT_SCHEMA_VERSION,
            mapping_id="lik",
            kind=MappingKind.LIKELIHOOD,
            variables=("lane_hours",),
            result_axis="magnitude",
            observation=(1.0, 2.0),
            observation_scale=1.0,
        )
        with self.assertRaisesRegex(JointSchemaError, "does not share dimension"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(var_time,),
                mappings=(likelihood,),
            )

    def test_compile_likelihood_with_matching_dimension_succeeds(self):
        axis = _axis("delivery", unit="hour")
        var_time = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        likelihood = MappingSpec(
            schema_version=JOINT_SCHEMA_VERSION,
            mapping_id="lik",
            kind=MappingKind.LIKELIHOOD,
            variables=("lane_hours",),
            result_axis="delivery",
            observation=(1.0, 2.0),
            observation_scale=1.0,
        )
        compiled = compile_joint_schema(
            question_id="q1",
            axes=(axis,),
            variables=(var_time,),
            mappings=(likelihood,),
        )
        entry = compiled[0]
        self.assertIsNone(entry.factor_ir)
        # Likelihood mappings have no IR contribution: log_potential is 0.
        self.assertEqual(entry.log_potential({"lane_hours": 1.5}), 0.0)

    def test_compile_rejects_unknown_evidence_reference(self):
        axis = _axis("delivery", unit="dimensionless")
        var = _variable("lane_hours", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping(
            "leg",
            "lane_hours / lane_hours",
            ("lane_hours",),
            evidence_atom_ids=("missing-atom",),
        )
        with self.assertRaisesRegex(JointSchemaError, "unknown evidence atom"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(var,),
                mappings=(mapping,),
                evidence=(_evidence("present-atom"),),
            )

    def test_compile_validates_variable_evidence_reference(self):
        axis = _axis("delivery", unit="dimensionless")
        var = _variable(
            "lane_hours",
            unit="hour",
            lower=1.0,
            upper=5.0,
            evidence_atom_id="missing-atom",
        )
        mapping = _mapping("leg", "lane_hours / lane_hours", ("lane_hours",))
        with self.assertRaisesRegex(JointSchemaError, "unknown evidence atom"):
            compile_joint_schema(
                question_id="q1",
                axes=(axis,),
                variables=(var,),
                mappings=(mapping,),
                evidence=(_evidence("present-atom"),),
            )


class CompiledMappingContractTests(TestCase):
    def test_compiled_mapping_is_frozen(self):
        axis = _axis("delivery", unit="dimensionless")
        var = _variable("a", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping("m", "a / a", ("a",))
        compiled = compile_joint_schema(
            question_id="q1",
            axes=(axis,),
            variables=(var,),
            mappings=(mapping,),
        )
        entry = compiled[0]
        with self.assertRaises(FrozenInstanceError):
            entry.mapping = mapping  # type: ignore[misc]

    def test_compiled_mapping_construct_rejects_mismatched_variables(self):
        axis = _axis("delivery", unit="dimensionless")
        var = _variable("a", unit="hour", lower=1.0, upper=5.0)
        mapping = _mapping("m", "a / a", ("a",))
        # Build a CompiledMapping directly with the wrong variable set.
        with self.assertRaises(JointSchemaError):
            CompiledMapping(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping=mapping,
                axis=axis,
                variables=(),
                variable_dimensions={},
                factor_ir=None,
            )