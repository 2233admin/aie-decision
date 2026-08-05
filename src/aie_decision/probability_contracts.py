"""Immutable probability contracts and restricted expression evaluation."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Optional, Sequence


class MarginalKind(str, Enum):
    """Probability semantics recorded for every leaf."""

    CONSTANT = "constant"
    QUANTILE_FITTED = "quantile_fitted"
    UNKNOWN = "unknown"


class DistributionFamily(str, Enum):
    """Continuous families supported by the quantile fitter."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"


class DependenceCase(str, Enum):
    """Declared joint dependence between uncertain leaves."""

    INDEPENDENT = "independent"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class CalibrationLabel(str, Enum):
    """Honest labelling of how the target interval was produced."""

    UNMEASURED = "unmeasured"
    UNMEASURED_WITH_DEPENDENCY_GAP = "unmeasured_with_dependency_gap"
    NO_PROBABILITY_SEMANTICS = "no_probability_semantics"
    UNMEASURED_UNKNOWN_MARGINAL = "unmeasured_unknown_marginal"


class CoverageSemantics(str, Enum):
    """Explicit label for the probability semantics of a target summary.

    * ``MONTE_CARLO_JOINT_SAMPLING`` — the declared joint model produced a
      reproducible 90 percent probability interval and
      ``probability_interval_valid`` is ``True``.
    * ``SCENARIO_BOUNDS_ONLY`` — at least one required marginal is unknown;
      the numeric range is a sensitivity envelope over the unknown
      marginals' declared domains and must NOT be treated as P05/P95.
    * ``INVALID_UNKNOWN_DEPENDENCE`` — every marginal carries probability
      semantics, but the joint dependence case needed to propagate more
      than one uncertain leaf is ``UNKNOWN``; the system fell back to an
      independent assumption and the resulting interval cannot be claimed
      as a 90 percent coverage.
    """

    MONTE_CARLO_JOINT_SAMPLING = "monte_carlo_joint_sampling"
    SCENARIO_BOUNDS_ONLY = "scenario_bounds_only"
    INVALID_UNKNOWN_DEPENDENCE = "invalid_unknown_dependence"


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstantMarginal:
    """A leaf whose value is known with no residual uncertainty."""

    value: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("constant value must be finite")

    @property
    def kind(self) -> MarginalKind:
        return MarginalKind.CONSTANT


@dataclass(frozen=True, slots=True)
class QuantileFittedMarginal:
    """A continuous leaf fit from a P05/P50/P95 triple."""

    p05: float
    p50: float
    p95: float
    family: DistributionFamily = DistributionFamily.NORMAL
    rationale: str = ""

    def __post_init__(self) -> None:
        for label, value in (("p05", self.p05), ("p50", self.p50), ("p95", self.p95)):
            if not isfinite(value):
                raise ValueError(f"{label} must be finite")
        if not (self.p05 < self.p50 < self.p95):
            raise ValueError("quantiles must satisfy p05 < p50 < p95")

    @property
    def kind(self) -> MarginalKind:
        return MarginalKind.QUANTILE_FITTED


@dataclass(frozen=True, slots=True)
class UnknownMarginal:
    """A leaf whose probability semantics are not yet defensible."""

    reason: str
    domain: tuple[float, float] = (0.0, 1.0)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("unknown marginal requires a reason")
        lower, upper = self.domain
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("unknown marginal domain must be ordered and finite")

    @property
    def kind(self) -> MarginalKind:
        return MarginalKind.UNKNOWN


Marginal = ConstantMarginal | QuantileFittedMarginal | UnknownMarginal


def marginal_kind(marginal: Marginal) -> MarginalKind:
    """Return the probability semantics of a marginal, validated."""

    if isinstance(marginal, ConstantMarginal):
        return MarginalKind.CONSTANT
    if isinstance(marginal, QuantileFittedMarginal):
        return MarginalKind.QUANTILE_FITTED
    if isinstance(marginal, UnknownMarginal):
        return MarginalKind.UNKNOWN
    raise ValueError(f"unsupported marginal type: {type(marginal).__name__}")


# ---------------------------------------------------------------------------
# Leaf and joint model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeafSpec:
    """A leaf identified by ``leaf_id`` together with its marginal."""

    leaf_id: str
    marginal: Marginal
    unit: str = ""
    measurement_procedure: str = ""

    def __post_init__(self) -> None:
        if not self.leaf_id.strip():
            raise ValueError("leaf_id is required")
        # Validate that the marginal type is one of the recognised kinds.
        marginal_kind(self.marginal)


@dataclass(frozen=True, slots=True)
class JointModel:
    """Declared joint dependence and reproducible Monte Carlo controls."""

    dependence: DependenceCase = DependenceCase.INDEPENDENT
    sample_count: int = 4096
    seed: int = 0
    # Positive / negative correlation strength on the Gaussian copula.
    correlation: float = 0.7

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not -1.0 < self.correlation < 1.0:
            raise ValueError("correlation must lie strictly between -1 and 1")
        if self.dependence not in (
            DependenceCase.INDEPENDENT,
            DependenceCase.POSITIVE,
            DependenceCase.NEGATIVE,
            DependenceCase.UNKNOWN,
        ):
            raise ValueError("dependence must be a declared case")


# ---------------------------------------------------------------------------
# Expression evaluator (local, restricted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledExpression:
    """A compiled arithmetic expression and the variables it references."""

    source: str
    tree: ast.Expression
    variables: tuple[str, ...]


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses unsupported syntax."""


_ALLOWED_AST = (
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


def compile_expression(source: str) -> CompiledExpression:
    """Parse ``source`` into a restricted arithmetic expression."""

    if not isinstance(source, str) or not source.strip():
        raise ExpressionError("expression source is required")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ExpressionError(
                "expression supports only names, numbers, parentheses, +, -, *, / and **"
            )
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ExpressionError("constants must be finite numbers")
        if isinstance(node, ast.Name) and not node.id.isidentifier():
            raise ExpressionError(f"invalid identifier in expression: {node.id!r}")
        # Exponentiation must use an integer constant so the restricted
        # expression semantics never silently evaluate ``**`` with a
        # non-integer exponent (which would be undefined behaviour under
        # negative bases, float overflow, etc.).
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
        ):
            exponent = node.right
            if not (
                isinstance(exponent, ast.Constant)
                and isinstance(exponent.value, int)
                and not isinstance(exponent.value, bool)
            ):
                raise ExpressionError(
                    "exponent must be an integer constant"
                )

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in names:
            names.append(node.id)
    if not names:
        raise ExpressionError("expression must reference at least one variable")

    return CompiledExpression(source=source.strip(), tree=tree, variables=tuple(names))


def evaluate_compiled(
    compiled: CompiledExpression, values: Mapping[str, float]
) -> float:
    """Evaluate a compiled expression against a mapping of variable values."""

    try:
        result = _eval_node(compiled.tree.body, values)
    except ZeroDivisionError as exc:
        raise ExpressionError("division by zero during evaluation") from exc
    if not isfinite(result):
        raise ExpressionError("expression did not yield a finite value")
    return float(result)


def _eval_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, values)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ExpressionError(f"missing value for variable: {node.id}")
        return float(values[node.id])
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            # The compile step guarantees integer constant exponents; the
            # try/except is purely a safety net against runtime overflow
            # producing a Python ``OverflowError`` that we surface as an
            # honest ``ExpressionError`` so downstream callers can recover.
            try:
                return left ** right
            except OverflowError as exc:
                raise ExpressionError(
                    f"exponentiation overflow for {left} ** {right}"
                ) from exc
    raise ExpressionError(f"unsupported expression node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Target summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetSummary:
    """Honest summary of a propagated target interval.

    ``probability_interval_valid`` is the single source of truth on whether
    :attr:`p05`/``p50``/``p95``/``width`` describe a 90 percent probability
    interval sampled from the declared joint model.

    When ``probability_interval_valid`` is ``True`` the quantiles are
    floats and ``coverage_semantics`` is
    :attr:`CoverageSemantics.MONTE_CARLO_JOINT_SAMPLING`.  When it is
    ``False`` the quantiles are ``None`` and ``coverage_semantics`` records
    why (``SCENARIO_BOUNDS_ONLY`` for an unknown marginal,
    ``INVALID_UNKNOWN_DEPENDENCE`` for a joint dependence that could not
    be settled).  ``scenario_bounds`` exposes a clearly labelled
    sensitivity envelope over each leaf's declared domain in either case;
    it must never be confused with a probability interval.
    """

    p05: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    width: Optional[float]
    sample_count: int
    seed: int
    method: str
    marginal_summary: str
    dependence: DependenceCase
    correlation: float
    calibration: CalibrationLabel
    probability_interval_valid: bool
    coverage_semantics: str
    scenario_bounds: Optional[tuple[float, float]] = None
    dependency_gaps: tuple[str, ...] = ()
    unknown_leaves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.probability_interval_valid:
            if (
                self.p05 is None
                or self.p50 is None
                or self.p95 is None
                or self.width is None
            ):
                raise ValueError(
                    "probability_interval_valid requires finite p05/p50/p95/width"
                )
            if (
                not isfinite(self.p05)
                or not isfinite(self.p50)
                or not isfinite(self.p95)
                or not isfinite(self.width)
            ):
                raise ValueError(
                    "valid probability interval requires finite quantiles and width"
                )
            if not (self.p05 <= self.p50 <= self.p95):
                raise ValueError("summary quantiles must be ordered")
            if self.width < 0:
                raise ValueError("summary width must be non-negative")
        else:
            # An invalid interval must keep quantiles as None and label
            # how the coverage was produced.
            if (
                self.p05 is not None
                or self.p50 is not None
                or self.p95 is not None
                or self.width is not None
            ):
                raise ValueError(
                    "invalid probability interval must leave quantiles as None"
                )
            if self.coverage_semantics == "monte_carlo_joint_sampling":
                raise ValueError(
                    "invalid probability interval cannot claim "
                    "monte_carlo_joint_sampling coverage"
                )


# ---------------------------------------------------------------------------
# Uncertainty contribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UncertaintyContribution:
    """A leaf's reducible contribution to the target interval width."""

    leaf_id: str
    baseline_width: float
    narrowed_width: float
    expected_narrowing: float
    narrowing_fraction: float
    method: str = "conditional_resolution_to_p50"

    def __post_init__(self) -> None:
        if self.expected_narrowing < 0:
            raise ValueError("expected_narrowing must be non-negative")


@dataclass(frozen=True, slots=True)
class WidthReductionRank:
    """A leaf ranked by expected reduction in target interval width."""

    leaf_id: str
    expected_narrowing: float
    narrowing_fraction: float
    priority: float


__all__ = [
    "MarginalKind",
    "DistributionFamily",
    "DependenceCase",
    "CalibrationLabel",
    "CoverageSemantics",
    "ConstantMarginal",
    "QuantileFittedMarginal",
    "UnknownMarginal",
    "Marginal",
    "marginal_kind",
    "LeafSpec",
    "JointModel",
    "CompiledExpression",
    "ExpressionError",
    "compile_expression",
    "evaluate_compiled",
    "TargetSummary",
    "UncertaintyContribution",
    "WidthReductionRank"
]
