"""MVP expression parser, compiler, and evaluator."""

from __future__ import annotations
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from wave_mvp_models import (
    CompiledMapping, MappingSpec, MappingFailure, VariableSpec,
)
from wave_mvp_unit import (
    Dimension, UnitError, _UNIT_TABLE, _DIMENSIONLESS,
)

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



# ---------------------------------------------------------------------------
# Formula parser with unit tracking.
# ---------------------------------------------------------------------------


_TOKEN_PATTERN = re.compile(
    r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>[\+\-\*\/\(\),])|(?P<power>\*\*))"
)


def _tokenize(formula: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(formula):
        match = _TOKEN_PATTERN.match(formula, pos)
        if not match:
            raise UnitError(f"unrecognized token at position {pos}: {formula[pos]!r}")
        if match.group("number"):
            tokens.append(("number", match.group("number")))
        elif match.group("ident"):
            tokens.append(("ident", match.group("ident")))
        elif match.group("op"):
            tokens.append(("op", match.group("op")))
        elif match.group("power"):
            tokens.append(("op", "**"))
        pos = match.end()
    return tokens


def _parse_expression(
    tokens: list[tuple[str, str]],
    *,
    value_dimensions: Mapping[str, Dimension] | None = None,
) -> tuple[Any, Dimension]:
    """Recursive descent parser for a restricted arithmetic grammar.

    Grammar (precedence high to low):
        atom := number | ident | '(' expression ')'
        power := atom ( '**' atom )*
        term := power ( ('*' | '/') power )*
        arith := term ( ('+' | '-') term )*

    ``value_dimensions`` resolves identifier names to their declared
    dimensions so that arithmetic operations can be unit-checked.
    """

    value_dimensions = value_dimensions or {}
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected_kind: str | None = None, expected_value: str | None = None) -> tuple[str, str]:
        nonlocal pos
        token = tokens[pos]
        if expected_kind is not None and token[0] != expected_kind:
            raise UnitError(f"expected {expected_kind}, got {token}")
        if expected_value is not None and token[1] != expected_value:
            raise UnitError(f"expected {expected_value}, got {token}")
        pos += 1
        return token

    def parse_atom() -> tuple[Any, Dimension]:
        token = peek()
        if token is None:
            raise UnitError("unexpected end of formula")
        if token[0] == "number":
            consume()
            scalar, dim = _parse_constant(float(token[1]))
            return ("const", scalar), dim
        if token[0] == "ident":
            consume()
            return ("var", token[1]), value_dimensions.get(token[1], Dimension())
        if token[1] == "(":
            consume()
            node, dim = parse_arith()
            closer = peek()
            if closer is None or closer[1] != ")":
                raise UnitError("missing closing parenthesis")
            consume()
            return node, dim
        raise UnitError(f"unexpected token: {token}")

    def parse_power() -> tuple[Any, Dimension]:
        base_node, base_dim = parse_atom()
        while peek() is not None and peek()[1] == "**":
            consume()
            exp_node, exp_dim = parse_atom()
            if not exp_dim.is_dimensionless():
                raise UnitError("exponent must be dimensionless")
            if exp_node[0] != "const":
                raise UnitError("only constant exponents are supported")
            exponent = exp_node[1]
            if exponent != int(exponent):
                raise UnitError("non-integer exponents are not supported")
            int_exp = int(exponent)
            base_dim = Dimension(
                exponents={k: v * int_exp for k, v in base_dim.exponents.items()}
            )
            base_node = ("pow", base_node, exponent)
        return base_node, base_dim

    def parse_term() -> tuple[Any, Dimension]:
        left_node, left_dim = parse_power()
        while peek() is not None and peek()[1] in {"*", "/"}:
            op = consume()[1]
            right_node, right_dim = parse_power()
            if op == "*":
                left_dim = left_dim.combine(right_dim, sign=1)
                left_node = ("mul", left_node, right_node)
            else:
                left_dim = left_dim.combine(right_dim, sign=-1)
                left_node = ("div", left_node, right_node)
        return left_node, left_dim

    def parse_arith() -> tuple[Any, Dimension]:
        left_node, left_dim = parse_term()
        while peek() is not None and peek()[1] in {"+", "-"}:
            op = consume()[1]
            right_node, right_dim = parse_term()
            if not left_dim.is_compatible_with(right_dim):
                raise UnitError(
                    f"unit_mismatch: cannot {'add' if op == '+' else 'subtract'} "
                    f"{left_dim.label()} and {right_dim.label()}"
                )
            left_node = ("add" if op == "+" else "sub", left_node, right_node)
        return left_node, left_dim

    node, dim = parse_arith()
    if pos != len(tokens):
        raise UnitError(f"unexpected trailing tokens: {tokens[pos:]}")
    return node, dim


def _evaluate_compiled(node: Any, values: Mapping[str, float]) -> float:
    kind = node[0]
    if kind == "const":
        return float(node[1])
    if kind == "var":
        return float(values[node[1]])
    if kind == "add":
        return _evaluate_compiled(node[1], values) + _evaluate_compiled(node[2], values)
    if kind == "sub":
        return _evaluate_compiled(node[1], values) - _evaluate_compiled(node[2], values)
    if kind == "mul":
        return _evaluate_compiled(node[1], values) * _evaluate_compiled(node[2], values)
    if kind == "div":
        right = _evaluate_compiled(node[2], values)
        if right == 0:
            raise UnitError("division by zero during evaluation")
        return _evaluate_compiled(node[1], values) / right
    if kind == "pow":
        base = _evaluate_compiled(node[1], values)
        exponent = float(node[2])
        return base ** exponent
    raise UnitError(f"unsupported compiled node: {kind}")


# ---------------------------------------------------------------------------
# Mapping compilation with unit checks.
# ---------------------------------------------------------------------------


def compile_mapping(
    mapping: MappingSpec,
    variable_specs: Mapping[str, VariableSpec],
    *,
    extra_constants: Mapping[str, float] | None = None,
) -> CompiledMapping:
    tokens = _tokenize(mapping.formula)
    extra_constants = extra_constants or {}
    referenced = sorted(
        {token[1] for token in tokens if token[0] == "ident"}
    )
    expected_dimension = Dimension.from_unit(mapping.expected_unit)
    unknown: list[str] = []
    for name in referenced:
        if name in extra_constants:
            continue
        if name not in variable_specs:
            unknown.append(name)
    if unknown:
        compiled, _ = _parse_expression(tokens)
        failure = MappingFailure(
            mapping_id=mapping.mapping_id,
            code="unknown_variable",
            message=f"unknown variables referenced: {', '.join(sorted(unknown))}",
            operand=unknown[0],
            operand_unit="dimensionless",
            expected_unit=mapping.expected_unit,
        )
        return CompiledMapping(
            mapping=mapping,
            compiled=compiled,
            variables=tuple(referenced),
            expected_dimension=expected_dimension,
            failure=failure,
        )

    # Resolve per-variable dimension map for unit propagation.
    value_dimensions: dict[str, Dimension] = {}
    for name in referenced:
        if name in extra_constants:
            value_dimensions[name] = Dimension()
        else:
            value_dimensions[name] = Dimension.from_unit(variable_specs[name].unit)
    try:
        compiled, produced_dimension = _parse_expression(
            tokens, value_dimensions=value_dimensions
        )
    except UnitError as exc:
        failure = MappingFailure(
            mapping_id=mapping.mapping_id,
            code="unit_mismatch",
            message=str(exc),
            operand=_infer_offending_operand(str(exc)),
            operand_unit=_infer_offending_unit(str(exc)),
            expected_unit=mapping.expected_unit,
        )
        return CompiledMapping(
            mapping=mapping,
            compiled=None,
            variables=tuple(referenced),
            expected_dimension=expected_dimension,
            failure=failure,
        )

    if not produced_dimension.is_compatible_with(expected_dimension):
        failure = MappingFailure(
            mapping_id=mapping.mapping_id,
            code="unit_mismatch",
            message=(
                f"unit_mismatch: produced {produced_dimension.label()} "
                f"does not match expected {expected_dimension.label()}"
            ),
            operand="formula",
            operand_unit=produced_dimension.label(),
            expected_unit=mapping.expected_unit,
        )
        return CompiledMapping(
            mapping=mapping,
            compiled=compiled,
            variables=tuple(referenced),
            expected_dimension=expected_dimension,
            failure=failure,
        )

    return CompiledMapping(
        mapping=mapping,
        compiled=compiled,
        variables=tuple(referenced),
        expected_dimension=expected_dimension,
    )


def _infer_offending_operand(message: str) -> str:
    if "cannot add" in message or "cannot subtract" in message:
        return "right"
    return "operand"


def _infer_offending_unit(message: str) -> str:
    match = re.search(
        r"and ([a-z_]+(?:\^[0-9-]+)?(?:\s*\*\s*[a-z_]+(?:\^[0-9-]+)?)*)", message
    )
    if match:
        return match.group(1)
    match = re.search(r"produced ([a-z_]+(?:\^[0-9-]+)?(?:\s*\*\s*[a-z_]+(?:\^[0-9-]+)?)*)", message)
    if match:
        return match.group(1)
    return "dimensionless"


