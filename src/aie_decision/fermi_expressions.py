"""Restricted arithmetic parsing, equivalence, and evaluation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .fermi_contract_core import FermiContractError, RedundancyReason, RestrictedExpressionError
from .fermi_units import (
    DEFAULT_UNIT_SYMBOLS,
    DIMENSIONLESS,
    CompoundUnit,
    divide_units,
    multiply_units,
    parse_compound_unit,
    power_units,
    units_close,
)

_ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
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
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


@dataclass(frozen=True, slots=True)
class RestrictedExpression:
    """A pre-parsed restricted arithmetic expression bound to variable names."""

    source: str
    tree: ast.Expression
    variables: tuple[str, ...]
    signature: str
    dimensions: CompoundUnit

    @property
    def text(self) -> str:
        return self.source


def _dimension_key(unit: CompoundUnit) -> str:
    if unit.is_dimensionless:
        return "1"
    return "*".join(f"{label}^{exp}" for label, exp in unit.to_canonical())


def _collect_expression_variables(tree: ast.AST) -> list[str]:
    """Collect variable names in the order they first appear in the source.

    ``ast.walk`` traverses the tree in breadth-first order, which diverges
    This differs from textual source order whenever a node holds a sub-expression
    that appears before its sibling at the same depth (e.g. ``b * a``
    would otherwise collect ``b`` first and silently rebind the dimension
    of ``a`` to the unit of ``b``).  Walking the tree with a depth-first
    visitor preserves the textual order in which the variables appear,
    so unit bindings line up with the source.
    """

    seen: list[str] = []
    for node in _depth_first(tree):
        if isinstance(node, ast.Name) and node.id not in seen:
            seen.append(node.id)
    return seen


def _depth_first(tree: ast.AST):
    """Yield nodes from ``tree`` in depth-first (source) order."""

    yield tree
    for child in ast.iter_child_nodes(tree):
        yield from _depth_first(child)


def parse_restricted_expression(
    expression: Any,
    *,
    variable_units: Mapping[str, CompoundUnit] | None = None,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> RestrictedExpression:
    """Parse ``expression`` into a restricted arithmetic tree.

    Only the nodes listed in ``_ALLOWED_AST_NODES`` are accepted.  Any other
    syntax — function calls, comprehensions, attribute access, boolean
    operators, comparisons, slicing — triggers a
    :class:`RestrictedExpressionError`.  Numeric constants must be finite,
    non-boolean values.  When ``variable_units`` is provided the runtime
    also computes the projected dimensions of the expression and the
    :class:`RestrictedExpression` carries them with it for downstream
    dimensional closure checks.
    """

    if not isinstance(expression, str) or not expression.strip():
        raise RestrictedExpressionError("expression is required")
    # ``^`` is the conventional mathematical power symbol.  Translate it to
    # ``**`` before parsing because Python's grammar would otherwise treat
    # ``a^2`` as a bitwise xor.
    normalised = expression.replace("^", "**")
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError as exc:
        raise RestrictedExpressionError(
            f"expression must be valid arithmetic: {exc.msg}"
        ) from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise RestrictedExpressionError(
                f"expression supports only numeric constants, names, parentheses, "
                f"+, -, *, /, and ^; rejected {type(node).__name__}"
            )
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RestrictedExpressionError("constants must be numeric and non-boolean")
            if not isfinite(float(value)):
                raise RestrictedExpressionError("constants must be finite")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RestrictedExpressionError(
                f"expression names cannot be dunder identifiers: {node.id}"
            )

    variables = tuple(_collect_expression_variables(tree))
    if variable_units is None:
        dimensions = DIMENSIONLESS
    else:
        missing = [name for name in variables if name not in variable_units]
        if missing:
            raise RestrictedExpressionError(
                "expression references undeclared variables: " + ", ".join(sorted(missing))
            )
        dimensions = _evaluate_dimensions(tree, variable_units, registry)
    signature = _canonical_signature(tree)
    return RestrictedExpression(
        source=normalised.strip(),
        tree=tree,
        variables=variables,
        signature=signature,
        dimensions=dimensions,
    )


def _evaluate_dimensions(
    node: ast.AST,
    variable_units: Mapping[str, CompoundUnit],
    registry: frozenset[str],
) -> CompoundUnit:
    if isinstance(node, ast.Expression):
        return _evaluate_dimensions(node.body, variable_units, registry)
    if isinstance(node, ast.Name):
        try:
            return variable_units[node.id]
        except KeyError as exc:
            raise RestrictedExpressionError(
                f"unknown variable in dimension evaluation: {node.id}"
            ) from exc
    if isinstance(node, ast.Constant):
        return DIMENSIONLESS
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_dimensions(node.operand, variable_units, registry)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_dimensions(node.left, variable_units, registry)
        right = _evaluate_dimensions(node.right, variable_units, registry)
        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
            if not units_close(left, right):
                raise RestrictedExpressionError(
                    "addition/subtraction requires matching dimensions: "
                    f"{_dimension_key(left)} vs {_dimension_key(right)}"
                )
            return left
        if isinstance(node.op, ast.Mult):
            return multiply_units(left, right)
        if isinstance(node.op, ast.Div):
            return divide_units(left, right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise RestrictedExpressionError("exponent must be an integer constant")
            return power_units(left, int(node.right.value))
    raise RestrictedExpressionError(f"unsupported expression node: {type(node).__name__}")


def _canonical_signature(node: ast.AST) -> str:
    """Canonical string form used to detect algebraically redundant rewrites.

    Multiplication and addition become commutative: children are sorted by
    their canonical form before joining.  Subtraction and division preserve
    order because they are not commutative.  Integer exponent constants are
    rendered as ``^N``; everything else inherits Python's ``ast.dump``
    canonicalisation for safety against obscure edge cases.
    """

    if isinstance(node, ast.Expression):
        return _canonical_signature(node.body)
    if isinstance(node, ast.Name):
        return f"name:{node.id}"
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, float) and value.is_integer():
            return f"const:{int(value)}"
        if isinstance(value, int):
            return f"const:{value}"
        return f"const:{value!r}"
    if isinstance(node, ast.UnaryOp):
        inner = _canonical_signature(node.operand)
        op = "+" if isinstance(node.op, ast.UAdd) else "-"
        return f"uop({op},{inner})"
    if isinstance(node, ast.BinOp):
        left = _canonical_signature(node.left)
        right = _canonical_signature(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return "(" + "+".join(sorted([left, right])) + ")"
        if isinstance(op, ast.Mult):
            return "(" + "*".join(sorted([left, right])) + ")"
        if isinstance(op, ast.Sub):
            return f"({left}-{right})"
        if isinstance(op, ast.Div):
            return f"({left}/{right})"
        if isinstance(op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                return f"pow({left},{node.right.value})"
            return f"pow({left},{ast.dump(node.right)})"
    return ast.dump(node)


def _expression_variables_match(left: RestrictedExpression, right: RestrictedExpression) -> bool:
    return sorted(left.variables) == sorted(right.variables)


def expressions_are_equivalent(left: RestrictedExpression, right: RestrictedExpression) -> bool:
    """Return True when ``left`` and ``right`` reference the same variables and
    share a commutative-canonicalised restricted-form signature.
    """

    return _expression_variables_match(left, right) and left.signature == right.signature


# ---------------------------------------------------------------------------
# Restricted arithmetic evaluator (numeric)
# ---------------------------------------------------------------------------


def evaluate_restricted_expression(
    expression: RestrictedExpression,
    values: Mapping[str, float],
) -> float:
    """Evaluate a :class:`RestrictedExpression` against numeric ``values``.

    Every variable referenced by the expression must resolve to a finite
    numeric value.  Constant leaves must already have been admitted by
    :func:`parse_restricted_expression`.  No callables, no attribute
    access, no sequence operations.
    """

    try:
        return _evaluate_node(expression.tree.body, values)
    except KeyError as exc:
        raise RestrictedExpressionError(f"missing value for variable: {exc.args[0]}") from exc


def _evaluate_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, values)
    if isinstance(node, ast.Name):
        value = values[node.id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RestrictedExpressionError(f"non-numeric value for {node.id}")
        result = float(value)
        if not isfinite(result):
            raise RestrictedExpressionError(f"non-finite value for {node.id}")
        return result
    if isinstance(node, ast.Constant):
        result = float(node.value)
        if not isfinite(result):
            raise RestrictedExpressionError("non-finite constant")
        return result
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, values)
        right = _evaluate_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise RestrictedExpressionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise RestrictedExpressionError("exponent must be an integer constant")
            return left ** int(node.right.value)
    raise RestrictedExpressionError(f"unsupported expression node: {type(node).__name__}")


__all__ = [
    "RestrictedExpression",
    "parse_restricted_expression",
    "expressions_are_equivalent",
    "evaluate_restricted_expression"
]
