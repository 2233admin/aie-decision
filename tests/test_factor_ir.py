from unittest import TestCase
import ast

from aie_decision.factor_ir import (
    DIMENSIONLESS,
    FACTOR_IR_VERSION,
    FactorIR,
    FactorIRError,
    compile_factor_ir,
)


def _try_compile(mapping_id: str, formula: str, dims: dict[str, str]) -> FactorIR:
    return compile_factor_ir(mapping_id, formula, dims)


class CompileTests(TestCase):
    def test_compile_simple_dimensionless_formula(self):
        ir = _try_compile("m1", "x + 1", {"x": DIMENSIONLESS})
        self.assertEqual(ir.schema_version, FACTOR_IR_VERSION)
        self.assertEqual(ir.referenced_variables, ("x",))
        self.assertEqual(ir.input_dimensions, (("x", DIMENSIONLESS),))
        self.assertEqual(ir.output_dimension, ())
        self.assertTrue(ir.output_is_dimensionless)

    def test_compile_dimension_cancelling_formula(self):
        ir = _try_compile("m2", "a / a", {"a": "length"})
        self.assertEqual(ir.output_dimension, ())
        self.assertTrue(ir.output_is_dimensionless)

    def test_compile_product_cancels_when_identical(self):
        ir = _try_compile("m3", "(a * b) / (a * b)", {"a": "length", "b": "time"})
        self.assertEqual(ir.output_dimension, ())
        self.assertTrue(ir.output_is_dimensionless)

    def test_compile_dimensionless_constant(self):
        ir = _try_compile("m4", "1.5 * 2", {})
        self.assertEqual(ir.output_dimension, ())
        self.assertEqual(ir.referenced_variables, ())

    def test_compile_rejects_non_dimensionless_output(self):
        with self.assertRaisesRegex(FactorIRError, "dimensionless"):
            _try_compile("m5", "x + y", {"x": "length", "y": "length"})

    def test_compile_rejects_mismatched_addition(self):
        with self.assertRaisesRegex(FactorIRError, "dimension mismatch"):
            _try_compile("m6", "a + b", {"a": "length", "b": "time"})

    def test_compile_rejects_unknown_variable(self):
        with self.assertRaisesRegex(FactorIRError, "unknown variable"):
            _try_compile("m7", "missing + 1", {"present": DIMENSIONLESS})

    def test_compile_rejects_unknown_grammar(self):
        with self.assertRaisesRegex(FactorIRError, "only allows"):
            _try_compile("m8", "x ** 2", {"x": DIMENSIONLESS})
        with self.assertRaisesRegex(FactorIRError, "numeric"):
            _try_compile("m9", "True and False", {})

    def test_compile_rejects_non_numeric_constant(self):
        with self.assertRaisesRegex(FactorIRError, "numeric"):
            _try_compile("m10", "'a' + 'b'", {})

    def test_compile_rejects_non_finite_constant(self):
        with self.assertRaisesRegex(FactorIRError, "finite"):
            _try_compile("m11", "1e1000", {})

    def test_compile_rejects_invalid_identifier(self):
        # Python's parser rejects malformed identifiers before we get a chance;
        # the fallback path is exercised by unicode identifiers that parse but
        # fail the ``isidentifier`` check after normalisation.
        with self.assertRaises(FactorIRError):
            _try_compile("m12", "ö + 1", {})

    def test_compile_rejects_invalid_syntax(self):
        with self.assertRaisesRegex(FactorIRError, "not valid Python"):
            _try_compile("m13", "x +", {"x": DIMENSIONLESS})

    def test_compile_rejects_extra_dimension_keys(self):
        with self.assertRaisesRegex(FactorIRError, "not referenced"):
            _try_compile("m14", "1", {"unused": DIMENSIONLESS})

    def test_compile_rejects_duplicate_input_dimensions(self):
        ir = _try_compile("m15", "x", {"x": DIMENSIONLESS})
        # building a manual FactorIR with duplicates is what we want to forbid
        with self.assertRaises(FactorIRError):
            FactorIR(
                schema_version=FACTOR_IR_VERSION,
                mapping_id="m15",
                formula="x",
                input_dimensions=(("x", DIMENSIONLESS), ("x", DIMENSIONLESS)),
                output_dimension=(),
                referenced_variables=("x",),
                tree=ir.tree,
            )


class EvaluationTests(TestCase):
    def test_log_potential_arithmetic(self):
        ir = _try_compile("eval1", "x * 2 + 1", {"x": DIMENSIONLESS})
        self.assertEqual(ir.log_potential({"x": 3.0}), 7.0)
        self.assertEqual(ir.log_potential({"x": -2.0}), -3.0)

    def test_log_potential_division(self):
        ir = _try_compile("eval2", "a / a", {"a": "length"})
        self.assertEqual(ir.log_potential({"a": 4.0}), 1.0)

    def test_log_potential_unary(self):
        ir = _try_compile("eval3", "-x + 1", {"x": DIMENSIONLESS})
        self.assertEqual(ir.log_potential({"x": 2.0}), -1.0)

    def test_log_potential_missing_variable(self):
        ir = _try_compile("eval4", "x + 1", {"x": DIMENSIONLESS})
        with self.assertRaisesRegex(FactorIRError, "missing variables"):
            ir.log_potential({})

    def test_log_potential_non_finite_input(self):
        ir = _try_compile("eval5", "x + 1", {"x": DIMENSIONLESS})
        with self.assertRaisesRegex(FactorIRError, "non-finite"):
            ir.log_potential({"x": float("nan")})

    def test_log_potential_division_by_zero(self):
        ir = _try_compile("eval6", "x / y", {"x": DIMENSIONLESS, "y": DIMENSIONLESS})
        with self.assertRaisesRegex(FactorIRError, "division by zero"):
            ir.log_potential({"x": 1.0, "y": 0.0})

    def test_log_potential_non_finite_result(self):
        # 1 / 1e-300 = 1e+300; 1e+300 * 1e+300 = 1e+600 -> inf in float
        ir = _try_compile("eval7", "x * x", {"x": DIMENSIONLESS})
        with self.assertRaisesRegex(FactorIRError, "non-finite"):
            ir.log_potential({"x": 1e308})

    def test_log_potential_handles_parentheses(self):
        ir = _try_compile("eval8", "(a + b) / (a + b)", {"a": "length", "b": "length"})
        self.assertEqual(ir.log_potential({"a": 2.0, "b": 3.0}), 1.0)


class DeterminismTests(TestCase):
    def test_compile_is_deterministic(self):
        first = _try_compile("det1", "x + y * 2", {"x": DIMENSIONLESS, "y": DIMENSIONLESS})
        second = _try_compile("det1", "x + y * 2", {"x": DIMENSIONLESS, "y": DIMENSIONLESS})
        # AST objects do not implement value equality, so compare attributes.
        for attr in (
            "schema_version",
            "mapping_id",
            "formula",
            "input_dimensions",
            "output_dimension",
            "referenced_variables",
        ):
            self.assertEqual(getattr(first, attr), getattr(second, attr))
        self.assertEqual(ast.dump(first.tree), ast.dump(second.tree))
        self.assertEqual(first.log_potential({"x": 1.0, "y": 2.0}), 5.0)
        self.assertEqual(second.log_potential({"x": 1.0, "y": 2.0}), 5.0)