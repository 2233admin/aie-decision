"""Restricted factor IR for dimensionless log-potential and deterministic axis transforms.

This module provides two compiled IR contracts that together cover every
``MappingSpec`` formula:

**FactorIR** — dimensionless log-potential for likelihood/support weighting.

* Only names, numeric literals, parentheses and the four arithmetic
  operators are accepted.  No attribute access, calls, comparisons, or
  truthy values can sneak in.
* Every compile performs a symbolic dimension analysis over the formula
  using the declared input dimensions.  ``+`` and ``-`` require identical
  dimensions; ``*`` and ``/`` combine dimensions additively and ``/``
  cancels dimensions when left and right share a key.
* The output dimension MUST be empty (every exponent zero).  A mapping
  that produces a dimensional value is rejected before any particle is
  evaluated.

**DeterministicTransform** — dimensional axis-value computation whose output
dimension equals the target result-axis dimension.

* Uses the same restricted grammar as FactorIR.
* The output dimension MUST match the declared target-axis dimension
  exactly (not just be dimensionless).  A dimensionless output from an
  axis-value formula is rejected because the axis needs dimensional values.

The IRs are consumed by :mod:`aie_decision.joint_schema` and
:mod:`aie_decision.particle_surface` but live in their own module so that
the compile/evaluate contract can be tested without the heavier wave-loop
machinery.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import isfinite
from typing import Mapping


FACTOR_IR_VERSION = "factor-ir/v1"
DIMENSIONLESS = "dimensionless"


class FactorIRError(ValueError):
    """Raised when the formula, dimensions, or evaluation are invalid."""


@dataclass(frozen=True, slots=True)
class FactorIR:
    """Compiled restricted factor IR with a dimensionless output contract."""

    schema_version: str
    mapping_id: str
    formula: str
    input_dimensions: tuple[tuple[str, str], ...]
    output_dimension: tuple[tuple[str, int], ...]
    referenced_variables: tuple[str, ...]
    tree: ast.Expression

    def __post_init__(self) -> None:
        if not self.mapping_id.strip():
            raise FactorIRError("FactorIR.mapping_id is required")
        if not self.formula.strip():
            raise FactorIRError("FactorIR.formula is required")
        if any(exp != 0 for _, exp in self.output_dimension):
            raise FactorIRError(
                f"FactorIR output must be dimensionless; got {dict(self.output_dimension)}"
            )
        names = [name for name, _ in self.input_dimensions]
        if len(set(names)) != len(names):
            raise FactorIRError("FactorIR.input_dimensions must declare each variable once")
        if set(names) != set(self.referenced_variables):
            raise FactorIRError(
                "FactorIR.referenced_variables must match input_dimensions keys"
            )

    @property
    def output_is_dimensionless(self) -> bool:
        return all(exp == 0 for _, exp in self.output_dimension)

    def log_potential(self, values: Mapping[str, float]) -> float:
        """Evaluate the IR over one particle and return a dimensionless value."""

        missing = [name for name in self.referenced_variables if name not in values]
        if missing:
            raise FactorIRError(
                f"missing variables in particle: {', '.join(sorted(missing))}"
            )
        result = _evaluate(self.tree, values)
        if not isfinite(result):
            raise FactorIRError(f"non-finite log_potential: {result!r}")
        return result


# ---------------------------------------------------------------------------
# DeterministicTransform — dimensional axis-value computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeterministicTransform:
    """Compiled axis-transform formula whose output dimension matches the target axis.

    Unlike :class:`FactorIR` which requires a dimensionless output for
    likelihood/support weighting, a ``DeterministicTransform`` computes the
    axis value directly.  The formula's output dimension must equal the
    target axis's dimension (e.g. a formula with output ``time`` can target
    an axis whose unit resolves to ``time``).
    """

    schema_version: str
    mapping_id: str
    formula: str
    input_dimensions: tuple[tuple[str, str], ...]
    output_dimension: tuple[tuple[str, int], ...]
    target_axis_dimension: str
    referenced_variables: tuple[str, ...]
    tree: ast.Expression

    def __post_init__(self) -> None:
        if not self.mapping_id.strip():
            raise FactorIRError("DeterministicTransform.mapping_id is required")
        if not self.formula.strip():
            raise FactorIRError("DeterministicTransform.formula is required")
        if not self.target_axis_dimension:
            raise FactorIRError("DeterministicTransform.target_axis_dimension is required")
        names = [name for name, _ in self.input_dimensions]
        if len(set(names)) != len(names):
            raise FactorIRError(
                "DeterministicTransform.input_dimensions must declare each variable once"
            )
        if set(names) != set(self.referenced_variables):
            raise FactorIRError(
                "DeterministicTransform.referenced_variables must match input_dimensions keys"
            )
        # Output dimension must match target_axis_dimension.  When the target
        # axis is dimensionless, an empty output dimension is valid
        # (dimensionless → dimensionless transform).
        od = dict(self.output_dimension)
        if self.target_axis_dimension == DIMENSIONLESS:
            if od:
                raise FactorIRError(
                    "DeterministicTransform with dimensionless target axis "
                    f"must have empty output_dimension; got {od}"
                )
        else:
            if not od:
                raise FactorIRError(
                    "DeterministicTransform output must be dimensional (matching target axis), "
                    f"not dimensionless; use FactorIR for dimensionless formulas"
                )
            if len(od) != 1:
                raise FactorIRError(
                    f"DeterministicTransform output must resolve to exactly one dimension; "
                    f"got {od}"
                )
            resolved = next(iter(od.keys()))
            if resolved != self.target_axis_dimension:
                raise FactorIRError(
                    f"DeterministicTransform output dimension {resolved!r} does not match "
                    f"target axis dimension {self.target_axis_dimension!r}"
                )

    def evaluate(self, values: Mapping[str, float]) -> float:
        """Evaluate the transform over one particle and return the axis value."""

        missing = [name for name in self.referenced_variables if name not in values]
        if missing:
            raise FactorIRError(
                f"missing variables in particle: {', '.join(sorted(missing))}"
            )
        result = _evaluate(self.tree, values)
        if not isfinite(result):
            raise FactorIRError(f"non-finite axis value: {result!r}")
        return result


def compile_axis_transform(
    mapping_id: str,
    formula: str,
    variable_dimensions: Mapping[str, str],
    target_axis_dimension: str,
) -> DeterministicTransform:
    """Parse the formula, verify the output matches *target_axis_dimension*, and freeze the IR.

    This is the dimensional counterpart of :func:`compile_factor_ir`.  Where
    that function requires a dimensionless output, this one requires the
    output to resolve to exactly the target axis dimension — including the
    special case where both the formula output and the target axis are
    dimensionless (e.g. ``regime_factor * severity_factor`` targeting a
    dimensionless ``magnitude`` axis).
    """

    tree = _parse_restricted(formula)
    signature = _dimension_signature(tree, dict(variable_dimensions))
    if target_axis_dimension == DIMENSIONLESS:
        # Dimensionless axis — the formula must also produce a dimensionless
        # output.  A non-empty signature means the formula is dimensional.
        if signature:
            dimensional = ", ".join(
                f"{dim}^{'+' if exp > 0 else ''}{exp}"
                for dim, exp in sorted(signature.items())
            )
            raise FactorIRError(
                f"axis-transform output is dimensional ({dimensional}); "
                f"expected dimensionless for dimensionless target axis {target_axis_dimension!r}"
            )
        # Empty signature → output is dimensionless → matches dimensionless axis.
        resolved = DIMENSIONLESS
    else:
        if not signature:
            raise FactorIRError(
                f"axis-transform formula output is dimensionless; "
                f"use compile_factor_ir for dimensionless formulas"
            )
        if len(signature) != 1:
            raise FactorIRError(
                "axis-transform output must resolve to exactly one dimension; got "
                + ", ".join(
                    f"{dim}^{'+' if exp > 0 else ''}{exp}"
                    for dim, exp in sorted(signature.items())
                )
            )
        resolved = next(iter(signature.keys()))
        if resolved != target_axis_dimension:
            raise FactorIRError(
                f"axis-transform output dimension {resolved!r} does not match "
                f"target axis dimension {target_axis_dimension!r}"
            )
    references = _collect_references(tree)
    missing = [name for name in references if name not in variable_dimensions]
    if missing:
        raise FactorIRError(
            "formula references variables without a declared dimension: "
            + ", ".join(sorted(missing))
        )
    extra = [name for name in variable_dimensions if name not in references]
    if extra:
        raise FactorIRError(
            "dimension map contains variables not referenced by the formula: "
            + ", ".join(sorted(extra))
        )
    ordered_dimensions = tuple(
        (name, variable_dimensions[name]) for name in references
    )
    return DeterministicTransform(
        schema_version=FACTOR_IR_VERSION,
        mapping_id=mapping_id,
        formula=formula.strip(),
        input_dimensions=ordered_dimensions,
        output_dimension=tuple(sorted(signature.items())),
        target_axis_dimension=target_axis_dimension,
        referenced_variables=references,
        tree=tree,
    )


# ---------------------------------------------------------------------------
# Parser + dimension analysis (shared by FactorIR and DeterministicTransform)
# ---------------------------------------------------------------------------

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
)


def _parse_restricted(formula: str) -> ast.Expression:
    if not isinstance(formula, str) or not formula.strip():
        raise FactorIRError("formula is required")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FactorIRError(f"formula is not valid Python: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FactorIRError(
                "factor IR only allows names, numeric literals, parentheses, +, -, *, /"
            )
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FactorIRError("factor IR constants must be numeric")
            if isinstance(value, float) and (not isfinite(value)):
                raise FactorIRError("factor IR constants must be finite")
        if isinstance(node, ast.Name) and not node.id.isidentifier():
            raise FactorIRError(f"variable name {node.id!r} is not a valid identifier")
    return tree


def _resolve_variable_dimension(dim_key: str) -> dict[str, int]:
    """Resolve a variable dimension key into a ``{dim: exponent}`` dict.

    Simple dimension keys (e.g. ``"time"``, ``"money/USD"``) produce
    ``{key: 1}``.  Composite keys like ``"money/USD:1;volume:-1"`` are
    deserialized into their constituent exponents.
    """
    if dim_key == DIMENSIONLESS:
        return {}
    if ";" not in dim_key:
        return {dim_key: 1}
    result: dict[str, int] = {}
    for part in dim_key.split(";"):
        dim, _, exp_str = part.partition(":")
        if not dim or not exp_str:
            raise FactorIRError(f"invalid composite dimension key: {dim_key!r}")
        result[dim] = int(exp_str)
    return result


def _dimension_signature(
    node: ast.AST,
    variable_dimensions: Mapping[str, str],
) -> dict[str, int]:
    """Return a {dimension: exponent} signature for ``node``.

    ``+`` and ``-`` require both sides to share the same signature; the
    result is that signature.  ``*`` and ``/`` accumulate exponents.  The
    final signature must be empty for the IR to be considered
    dimensionless.
    """

    if isinstance(node, ast.Expression):
        return _dimension_signature(node.body, variable_dimensions)
    if isinstance(node, ast.Constant):
        return {}
    if isinstance(node, ast.Name):
        if node.id not in variable_dimensions:
            raise FactorIRError(f"unknown variable in formula: {node.id!r}")
        return _resolve_variable_dimension(variable_dimensions[node.id])
    if isinstance(node, ast.UnaryOp):
        return _dimension_signature(node.operand, variable_dimensions)
    if isinstance(node, ast.BinOp):
        left = _dimension_signature(node.left, variable_dimensions)
        right = _dimension_signature(node.right, variable_dimensions)
        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
            if left != right:
                raise FactorIRError(
                    "dimension mismatch in "
                    + ("addition" if isinstance(node.op, ast.Add) else "subtraction")
                    + f": {dict(left) or DIMENSIONLESS} vs {dict(right) or DIMENSIONLESS}"
                )
            return left
        if isinstance(node.op, ast.Mult):
            merged = dict(left)
            for dimension, exponent in right.items():
                merged[dimension] = merged.get(dimension, 0) + exponent
            return _prune(merged)
        if isinstance(node.op, ast.Div):
            merged = dict(left)
            for dimension, exponent in right.items():
                merged[dimension] = merged.get(dimension, 0) - exponent
            return _prune(merged)
    raise FactorIRError(f"unsupported AST node: {type(node).__name__}")


def _prune(signature: Mapping[str, int]) -> dict[str, int]:
    return {key: exponent for key, exponent in signature.items() if exponent != 0}


def _collect_references(node: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id not in names:
            names.append(sub.id)
    return tuple(names)


def _evaluate(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Name):
        value = values[node.id]
        if not isfinite(value):
            raise FactorIRError(f"variable {node.id!r} produced non-finite value")
        return float(value)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, values)
        right = _evaluate(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise FactorIRError("division by zero is not allowed")
            return left / right
    raise FactorIRError(f"unsupported AST node: {type(node).__name__}")


def compile_factor_ir(
    mapping_id: str,
    formula: str,
    variable_dimensions: Mapping[str, str],
) -> FactorIR:
    """Parse the formula, verify it produces a dimensionless result, and freeze the IR."""

    tree = _parse_restricted(formula)
    signature = _dimension_signature(tree, dict(variable_dimensions))
    if signature:
        raise FactorIRError(
            "factor IR output must be dimensionless; got "
            + ", ".join(
                f"{dimension}^{'+' if exponent > 0 else ''}{exponent}"
                for dimension, exponent in sorted(signature.items())
            )
        )
    references = _collect_references(tree)
    missing = [name for name in references if name not in variable_dimensions]
    if missing:
        raise FactorIRError(
            "formula references variables without a declared dimension: "
            + ", ".join(sorted(missing))
        )
    extra = [name for name in variable_dimensions if name not in references]
    if extra:
        raise FactorIRError(
            "dimension map contains variables not referenced by the formula: "
            + ", ".join(sorted(extra))
        )
    ordered_dimensions = tuple(
        (name, variable_dimensions[name]) for name in references
    )
    return FactorIR(
        schema_version=FACTOR_IR_VERSION,
        mapping_id=mapping_id,
        formula=formula.strip(),
        input_dimensions=ordered_dimensions,
        output_dimension=tuple(sorted(signature.items())),
        referenced_variables=references,
        tree=tree,
    )


__all__ = [
    "DIMENSIONLESS",
    "FACTOR_IR_VERSION",
    "DeterministicTransform",
    "FactorIR",
    "FactorIRError",
    "compile_axis_transform",
    "compile_factor_ir",
]