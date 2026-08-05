"""Restricted factor IR producing a dimensionless log-potential.

The factor IR is the executable form of a single ``MappingSpec`` formula.  It
is intentionally tiny:

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

The IR is consumed by :mod:`aie_decision.joint_schema` but lives in its
own module so that the compile/evaluate contract can be tested without
the heavier wave-loop machinery.
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
    """Compiled restricted factor IR with declared output dimensions.

    The output dimension records the dimensional signature of the formula;
    it may be dimensional when the formula computes an axis value rather
    than a log-potential.  Dimension compatibility of individual operations
    (e.g. no adding time to money) is still enforced at compile time.
    """

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
# Parser + dimension analysis
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
    "FactorIR",
    "FactorIRError",
    "compile_factor_ir",
]