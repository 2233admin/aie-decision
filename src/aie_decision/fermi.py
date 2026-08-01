"""Minimal Fermi decomposition and 90% interval propagation.

The first product surface deliberately accepts a declared arithmetic decomposition
instead of pretending that a deterministic runtime can invent one from prose.  It
keeps only variables referenced by the formula, propagates their supplied bounds,
and ranks the variable whose resolution would narrow the target interval most.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .measurement import threshold_robustness


@dataclass(frozen=True, slots=True)
class NumericInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isfinite(self.lower) or not isfinite(self.upper) or self.lower > self.upper:
            raise ValueError("interval bounds must be finite and ordered")

    @property
    def width(self) -> float:
        return self.upper - self.lower


class _Names(ast.NodeVisitor):
    def __init__(self) -> None:
        self.items: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.items:
            self.items.append(node.id)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _parse_formula(formula: str) -> tuple[ast.Expression, tuple[str, ...]]:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula is required")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError("formula must be a valid arithmetic expression") from exc

    allowed = (
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
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("formula supports only names, numbers, parentheses, +, -, *, and /")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("formula constants must be numbers")

    collector = _Names()
    collector.visit(tree)
    if not collector.items:
        raise ValueError("formula must reference at least one variable")
    return tree, tuple(collector.items)


def _evaluate(node: ast.AST, values: Mapping[str, NumericInterval]) -> NumericInterval:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Constant):
        value = float(node.value)
        return NumericInterval(value, value)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return NumericInterval(-operand.upper, -operand.lower)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, values)
        right = _evaluate(node.right, values)
        if isinstance(node.op, ast.Add):
            return NumericInterval(left.lower + right.lower, left.upper + right.upper)
        if isinstance(node.op, ast.Sub):
            return NumericInterval(left.lower - right.upper, left.upper - right.lower)
        if isinstance(node.op, ast.Mult):
            products = (
                left.lower * right.lower,
                left.lower * right.upper,
                left.upper * right.lower,
                left.upper * right.upper,
            )
            return NumericInterval(min(products), max(products))
        if isinstance(node.op, ast.Div):
            if right.lower <= 0 <= right.upper:
                raise ValueError("a denominator interval cannot contain zero")
            quotients = (
                left.lower / right.lower,
                left.lower / right.upper,
                left.upper / right.lower,
                left.upper / right.upper,
            )
            return NumericInterval(min(quotients), max(quotients))
    raise ValueError("unsupported formula expression")


def _variables(payload: Mapping[str, Any]) -> tuple[dict[str, NumericInterval], dict[str, dict[str, Any]]]:
    raw = payload.get("variables")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("variables must be a non-empty array")
    intervals: dict[str, NumericInterval] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"variables[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"variables[{index}].name must be a valid identifier")
        if name in intervals:
            raise ValueError(f"duplicate variable: {name}")
        interval = NumericInterval(
            _number(item.get("lower"), f"variables[{index}].lower"),
            _number(item.get("upper"), f"variables[{index}].upper"),
        )
        intervals[name] = interval
        metadata[name] = {
            "name": name,
            "lower": interval.lower,
            "upper": interval.upper,
            "unit": str(item.get("unit", "")),
            "method": str(item.get("method", "user_supplied_90_percent_interval")),
        }
    return intervals, metadata


def estimate_fermi(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the standalone v1 loop from a declared Fermi formula to interval width."""
    if not isinstance(payload, Mapping):
        raise ValueError("Fermi input must be an object")
    question = payload.get("question")
    target = payload.get("target")
    unit = payload.get("unit")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target is required")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit is required")

    coverage = _number(payload.get("coverage", 0.9), "coverage")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    formula = payload.get("formula")
    tree, referenced = _parse_formula(formula)
    intervals, metadata = _variables(payload)
    missing = tuple(name for name in referenced if name not in intervals)
    if missing:
        raise ValueError("formula variables missing bounds: " + ", ".join(missing))

    used = {name: intervals[name] for name in referenced}
    target_interval = _evaluate(tree, used)
    reference_value = payload.get("reference_value")
    normalized_width = None
    if reference_value is not None:
        reference = abs(_number(reference_value, "reference_value"))
        if reference == 0:
            raise ValueError("reference_value must be non-zero")
        normalized_width = target_interval.width / reference

    acceptable_width = payload.get("acceptable_width")
    if acceptable_width is not None:
        acceptable_width = _number(acceptable_width, "acceptable_width")
        if acceptable_width < 0:
            raise ValueError("acceptable_width must be non-negative")

    thresholds_raw = payload.get("thresholds", ())
    if not isinstance(thresholds_raw, Sequence) or isinstance(thresholds_raw, (str, bytes)):
        raise ValueError("thresholds must be an array")
    thresholds = tuple(_number(value, "threshold") for value in thresholds_raw)
    robustness = threshold_robustness(target_interval.lower, target_interval.upper, thresholds)

    sensitivity: list[dict[str, Any]] = []
    for name in referenced:
        interval = used[name]
        midpoint = (interval.lower + interval.upper) / 2
        resolved = dict(used)
        resolved[name] = NumericInterval(midpoint, midpoint)
        narrowed = _evaluate(tree, resolved)
        potential = max(0.0, target_interval.width - narrowed.width)
        sensitivity.append(
            {
                "variable": name,
                "potential_narrowing": potential,
                "narrowing_fraction": potential / target_interval.width if target_interval.width else 0.0,
                "width_if_resolved_to_midpoint": narrowed.width,
            }
        )
    sensitivity.sort(key=lambda item: (-item["potential_narrowing"], item["variable"]))

    if acceptable_width is not None:
        informative = target_interval.width <= acceptable_width
    elif normalized_width is not None:
        informative = normalized_width < 1
    else:
        informative = False
    interval_status = "uncalibrated_informative" if informative else "uncalibrated_uninformative"
    answerability = "answerable" if informative and robustness.robust else "conditionally_answerable"
    next_measurement = next(
        (item["variable"] for item in sensitivity if item["potential_narrowing"] > 0),
        None,
    )

    return {
        "status": interval_status,
        "answerability": answerability,
        "question": question.strip(),
        "target": target.strip(),
        "unit": unit.strip(),
        "formula": formula.strip(),
        "coverage": coverage,
        "coverage_semantics": "subjective_credible_interval",
        "coverage_basis": "declared_joint_input_region",
        "calibration": "unmeasured",
        "minimal_variables": [metadata[name] for name in referenced],
        "minimal_variable_count": len(referenced),
        "minimality_basis": "variables_referenced_by_declared_formula",
        "excluded_variables": [name for name in intervals if name not in referenced],
        "target_interval": {
            "lower": target_interval.lower,
            "upper": target_interval.upper,
        },
        "absolute_width": target_interval.width,
        "interval_method": "deterministic_interval_arithmetic",
        "reference_value": reference_value,
        "normalized_width": normalized_width,
        "acceptable_width": acceptable_width,
        "within_acceptable_width": acceptable_width is not None and target_interval.width <= acceptable_width,
        "decision_robust": robustness.robust,
        "crossed_thresholds": list(robustness.crossed_thresholds),
        "sensitivity": sensitivity,
        "largest_uncertainty_source": next_measurement,
        "next_measurement": next_measurement,
    }
