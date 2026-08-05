"""Probability semantics and reproducible joint sampling for the uncertainty frontier.

This module is intentionally self-contained.  It does not import any other
``aie_decision`` submodule so the uncertainty track can be developed in
parallel with the decomposition tree track.  All inputs are plain immutable
dataclasses; the local expression evaluator is a deliberately narrow AST
walker that accepts names, numbers, parentheses, ``+ - * /``, unary signs
and the exponentiation operator.

The module separates three leaf probability semantics and never relabels a
Cartesian endpoint range as a target probability interval.

* :class:`ConstantMarginal` — a fixed value with rationale.  No uncertainty.
* :class:`QuantileFittedMarginal` — a continuous distribution fit from the
  submitted P05/P50/P95 triple together with a declared family.  The
  submitted quantiles are preserved verbatim alongside the fitted family
  so the consumer can audit the approximation.
* :class:`UnknownMarginal` — an explicit gap.  Joint propagation never
  silently substitutes a uniform draw; instead the target summary records
  a clearly labelled scenario envelope over the unknown marginal's domain
  and exposes ``probability_interval_valid=False`` with
  ``coverage_semantics='scenario_bounds_only'``.

Joint propagation uses a deterministic Gaussian copula.  The declared
dependence case controls the correlation level.  ``UNKNOWN`` dependence is
recorded as a dependency gap; whenever joint propagation is actually
required (two or more uncertain leaves) the resulting summary is labelled
``probability_interval_valid=False`` with
``coverage_semantics='invalid_unknown_dependence'`` so downstream
sufficiency gates cannot certify a 90 percent interval.  Negative
equicorrelation is validated against the number of uncertain leaves and
the sampling raises a clear error before attempting Cholesky when the
requested matrix is not positive definite.
"""

from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass
from enum import Enum
from math import erf, isfinite, sqrt
from typing import Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Inverse normal CDF (Peter Acklam's algorithm) — used by the copula.
# ---------------------------------------------------------------------------


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's approximation of the inverse standard normal CDF."""

    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("inverse normal CDF requires a probability in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285401469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368561417e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = sqrt(-2.0 * math.log(1.0 - p))
    return (
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ---------------------------------------------------------------------------
# Sampling kernels
# ---------------------------------------------------------------------------


def _sample_constant(marginal: ConstantMarginal, n: int) -> list[float]:
    return [marginal.value] * n


def _sample_for_family(
    marginal: QuantileFittedMarginal,
    n: int,
    rng: random.Random,
) -> list[float]:
    family = marginal.family
    if family is DistributionFamily.NORMAL:
        mean = marginal.p50
        std = (marginal.p95 - marginal.p05) / (2.0 * 1.6448536269514722)
        if std <= 0.0:
            return [mean] * n
        return [rng.gauss(mean, std) for _ in range(n)]
    if family is DistributionFamily.LOGNORMAL:
        if marginal.p05 <= 0 or marginal.p50 <= 0 or marginal.p95 <= 0:
            raise ValueError("lognormal fit requires strictly positive quantiles")
        median = marginal.p50
        sigma = (math.log(marginal.p95) - math.log(marginal.p05)) / (
            2.0 * 1.6448536269514722
        )
        mu = math.log(median)
        return [rng.lognormvariate(mu, sigma) for _ in range(n)]
    if family is DistributionFamily.TRIANGULAR:
        return [rng.triangular(marginal.p05, marginal.p95, marginal.p50) for _ in range(n)]
    raise ValueError(f"unsupported distribution family: {family}")


def _sample_unknown(marginal: UnknownMarginal, n: int, rng: random.Random) -> list[float]:
    lower, upper = marginal.domain
    return [rng.uniform(lower, upper) for _ in range(n)]


def _sample_marginal(marginal: Marginal, n: int, rng: random.Random) -> list[float]:
    if isinstance(marginal, ConstantMarginal):
        return _sample_constant(marginal, n)
    if isinstance(marginal, QuantileFittedMarginal):
        return _sample_for_family(marginal, n, rng)
    if isinstance(marginal, UnknownMarginal):
        return _sample_unknown(marginal, n, rng)
    raise ValueError(f"unsupported marginal type: {type(marginal).__name__}")


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile from an empty sample")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile probability must lie in (0, 1)")
    n = len(sorted_values)
    pos = q * (n - 1)
    lower_index = int(math.floor(pos))
    upper_index = int(math.ceil(pos))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = pos - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


# ---------------------------------------------------------------------------
# Joint sampling
# ---------------------------------------------------------------------------


def _dependence_correlation(model: JointModel) -> float:
    if model.dependence is DependenceCase.POSITIVE:
        return model.correlation
    if model.dependence is DependenceCase.NEGATIVE:
        return -model.correlation
    return 0.0


def _build_correlation_matrix(dimension: int, off_diagonal: float) -> list[list[float]]:
    matrix = [[0.0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        matrix[i][i] = 1.0
        for j in range(i + 1, dimension):
            matrix[i][j] = off_diagonal
            matrix[j][i] = off_diagonal
    return matrix


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Return the lower-triangular Cholesky factor of a symmetric PD matrix."""

    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j]
            for k in range(j):
                total -= L[i][k] * L[j][k]
            if i == j:
                if total <= 0:
                    raise ValueError("correlation matrix must be positive definite")
                L[i][j] = sqrt(total)
            else:
                L[i][j] = total / L[j][j]
    return L


def _rank_to_uniform(rng: random.Random, n: int) -> list[float]:
    """Independent standard uniforms (reproducible)."""

    return [rng.random() for _ in range(n)]


def _correlated_uniforms(
    rng: random.Random, n: int, dimension: int, off_diagonal: float
) -> list[list[float]]:
    """Gaussian copula uniforms with constant off-diagonal correlation."""

    if dimension == 1 or off_diagonal == 0.0:
        return [_rank_to_uniform(rng, n) for _ in range(dimension)]
    matrix = _build_correlation_matrix(dimension, off_diagonal)
    L = _cholesky(matrix)
    correlated: list[list[float]] = [[0.0] * n for _ in range(dimension)]
    for t in range(n):
        independent = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
        for i in range(dimension):
            total = 0.0
            for j in range(i + 1):
                total += L[i][j] * independent[j]
            correlated[i][t] = _normal_cdf(total)
    return correlated


def _marginal_summary(leaves: Sequence[LeafSpec]) -> str:
    kinds: list[str] = []
    for leaf in leaves:
        kinds.append(f"{leaf.leaf_id}={marginal_kind(leaf.marginal).value}")
    return ",".join(kinds) if kinds else "empty"


def _scenario_value(marginal: Marginal, *, upper: bool) -> float:
    """Return the scenario lower/upper envelope value for a marginal.

    ConstantMarginal collapses to its value; QuantileFittedMarginal uses
    P05 / P95; UnknownMarginal uses its declared admissible domain.  These
    values are used only to compute clearly labelled sensitivity bounds,
    never as a probability quantile.
    """

    if isinstance(marginal, ConstantMarginal):
        return marginal.value
    if isinstance(marginal, QuantileFittedMarginal):
        return marginal.p95 if upper else marginal.p05
    if isinstance(marginal, UnknownMarginal):
        return marginal.domain[1] if upper else marginal.domain[0]
    raise ValueError(f"unsupported marginal: {type(marginal).__name__}")


def _compute_scenario_bounds(
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
) -> tuple[float, float]:
    """Return the (lower, upper) envelope of evaluating ``expression`` with
    every leaf held at the lower, then the upper, end of its declared
    scenario domain.  Clearly labelled and NOT a probability interval.
    """

    lower_values = {
        leaf.leaf_id: _scenario_value(leaf.marginal, upper=False) for leaf in leaves
    }
    upper_values = {
        leaf.leaf_id: _scenario_value(leaf.marginal, upper=True) for leaf in leaves
    }
    low_eval = evaluate_compiled(expression, lower_values)
    high_eval = evaluate_compiled(expression, upper_values)
    return (min(low_eval, high_eval), max(low_eval, high_eval))


def _validate_negative_equicorrelation(correlation: float, n_uncertain: int) -> None:
    """Raise ``ValueError`` when ``-correlation`` is not a valid equicorrelation.

    For ``n`` uncertain leaves, the equicorrelation matrix with off-diagonals
    ``-correlation`` has eigenvalues ``1-correlation`` (multiplicity ``n-1``)
    and ``1-(n-1)*correlation`` (multiplicity 1).  Positive definiteness
    therefore requires both ``correlation<1`` and
    ``correlation<1/(n-1)``.  The first is guaranteed by
    :class:`JointModel` validation; the second is what this helper guards.
    """

    if n_uncertain <= 1:
        return
    limit = 1.0 / (n_uncertain - 1)
    if correlation >= limit:
        raise ValueError(
            f"negative equicorrelation {correlation} for {n_uncertain} "
            f"uncertain leaves is not positive definite; requires "
            f"correlation < {limit:.6g}"
        )


def _collect_dependency_gaps(
    model: JointModel,
    unknown_leaves: tuple[str, ...],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if model.dependence is DependenceCase.UNKNOWN:
        gaps.append("leaf_dependence_undeclared")
    if unknown_leaves:
        gaps.append(
            "unknown_marginal_semantics:" + ",".join(sorted(unknown_leaves))
        )
    return tuple(gaps)


def _monte_carlo_propagate(
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
) -> tuple[float, float, float, float]:
    """Run the deterministic Monte Carlo joint sample and return
    ``(p05, p50, p95, width)``.  Used when probability semantics are valid.
    """

    n = model.sample_count
    rng = random.Random(model.seed)

    uncertain = [
        leaf
        for leaf in leaves
        if not isinstance(leaf.marginal, (ConstantMarginal, UnknownMarginal))
    ]

    marginal_draws: dict[str, list[float]] = {}
    for leaf in leaves:
        marginal_draws[leaf.leaf_id] = _sample_marginal(leaf.marginal, n, rng)

    off_diag = _dependence_correlation(model)
    if off_diag != 0.0 and len(uncertain) > 1:
        correlated_uniforms = _correlated_uniforms(rng, n, len(uncertain), off_diag)
        for index, leaf in enumerate(uncertain):
            uniforms = correlated_uniforms[index]
            sorted_draws = sorted(marginal_draws[leaf.leaf_id])
            mapped = [
                sorted_draws[min(len(sorted_draws) - 1, int(u * len(sorted_draws)))]
                for u in uniforms
            ]
            marginal_draws[leaf.leaf_id] = mapped

    targets: list[float] = []
    for t in range(n):
        values = {
            leaf.leaf_id: marginal_draws[leaf.leaf_id][t] for leaf in leaves
        }
        targets.append(evaluate_compiled(expression, values))
    targets.sort()

    p05 = _quantile(targets, 0.05)
    p50 = _quantile(targets, 0.50)
    p95 = _quantile(targets, 0.95)
    width = p95 - p05
    return p05, p50, p95, width


def joint_sample(
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
) -> TargetSummary:
    """Propagate leaf uncertainty through ``expression`` under ``model``.

    The summary's ``probability_interval_valid`` field is the only place
    downstream consumers should look to decide whether ``p05``/``p50``/
    ``p95`` describe a 90 percent probability interval.  When any required
    marginal carries :class:`UnknownMarginal`, the summary uses
    ``coverage_semantics='scenario_bounds_only'`` and exposes a labelled
    scenario envelope over the unknown leaf domains; the probability
    quantiles are ``None``.  When the declared dependence case is
    ``UNKNOWN`` and joint propagation is required (two or more uncertain
    non-constant leaves), the summary uses
    ``coverage_semantics='invalid_unknown_dependence'``; the probability
    quantiles are ``None`` and the scenario envelope is recorded as a
    sensitivity reference.
    """

    if not leaves:
        raise ValueError("joint_sample requires at least one leaf")

    referenced = set(expression.variables)
    leaf_ids = {leaf.leaf_id for leaf in leaves}
    missing = referenced - leaf_ids
    if missing:
        raise ValueError(
            "expression references leaves without probability specs: "
            + ", ".join(sorted(missing))
        )

    unknown_leaves = tuple(
        leaf.leaf_id for leaf in leaves if isinstance(leaf.marginal, UnknownMarginal)
    )
    # The copula dimension must include only leaves that are actually
    # sampled with correlated uniforms: constant leaves are deterministic
    # and unknown leaves are routed to scenario bounds.  Counting them
    # would either inflate the equicorrelation limit check or mislead the
    # Gaussian copula about the structure of joint uncertainty.
    copula_leaves = [
        leaf
        for leaf in leaves
        if not isinstance(leaf.marginal, (ConstantMarginal, UnknownMarginal))
    ]
    n_copula = len(copula_leaves)
    # ``n_uncertain`` keeps the previous semantic of "any non-constant
    # leaf" for downstream reasoning about whether joint propagation is
    # actually required (an unknown leaf still needs the copula to know
    # whether the joint structure matters).
    n_uncertain = sum(
        0 if isinstance(leaf.marginal, ConstantMarginal) else 1
        for leaf in leaves
    )

    dependency_gaps = _collect_dependency_gaps(model, unknown_leaves)
    marginal_summary = _marginal_summary(leaves)
    off_diag = _dependence_correlation(model)

    # Case A: an unknown marginal is required to produce a target value.
    # We never silently substitute a uniform draw; the scenario envelope
    # is the only honest numeric summary.  This branch must run BEFORE
    # negative-equicorrelation validation: the unknown leaf routes the
    # whole summary into scenario bounds, so we don't want a downstream
    # matrix-PD failure to mask the unknown-marginal gap.
    if unknown_leaves:
        scenario_bounds = _compute_scenario_bounds(leaves, expression)
        return TargetSummary(
            p05=None,
            p50=None,
            p95=None,
            width=None,
            sample_count=model.sample_count,
            seed=model.seed,
            method="scenario_bounds_only",
            marginal_summary=marginal_summary,
            dependence=model.dependence,
            correlation=off_diag,
            calibration=CalibrationLabel.UNMEASURED_UNKNOWN_MARGINAL,
            probability_interval_valid=False,
            coverage_semantics=CoverageSemantics.SCENARIO_BOUNDS_ONLY.value,
            scenario_bounds=scenario_bounds,
            dependency_gaps=dependency_gaps,
            unknown_leaves=unknown_leaves,
        )

    # Validate negative equicorrelation against the copula dimension only.
    # Once the unknown-leaf branch is ruled out, the copula count is the
    # authoritative measure of how many draws share the equicorrelation.
    if (
        model.dependence is DependenceCase.NEGATIVE
        and n_copula > 1
    ):
        _validate_negative_equicorrelation(model.correlation, n_copula)

    # Joint propagation is required only when more than one non-constant
    # leaf carries uncertainty.
    dependence_required = n_uncertain >= 2

    # Case B: joint dependence is unresolved.  We expose the same
    # scenario envelope but explicitly mark probability semantics invalid.
    if dependence_required and model.dependence is DependenceCase.UNKNOWN:
        scenario_bounds = _compute_scenario_bounds(leaves, expression)
        return TargetSummary(
            p05=None,
            p50=None,
            p95=None,
            width=None,
            sample_count=model.sample_count,
            seed=model.seed,
            method="scenario_only_independent_unresolvable",
            marginal_summary=marginal_summary,
            dependence=DependenceCase.UNKNOWN,
            correlation=0.0,
            calibration=CalibrationLabel.UNMEASURED_WITH_DEPENDENCY_GAP,
            probability_interval_valid=False,
            coverage_semantics=CoverageSemantics.INVALID_UNKNOWN_DEPENDENCE.value,
            scenario_bounds=scenario_bounds,
            dependency_gaps=dependency_gaps,
            unknown_leaves=(),
        )

    # Case C: valid probability semantics.  Run the deterministic Monte
    # Carlo propagation and produce a real 90 percent interval.
    p05, p50, p95, width = _monte_carlo_propagate(leaves, expression, model)
    return TargetSummary(
        p05=p05,
        p50=p50,
        p95=p95,
        width=width,
        sample_count=model.sample_count,
        seed=model.seed,
        method="monte_carlo_joint_sampling",
        marginal_summary=marginal_summary,
        dependence=model.dependence,
        correlation=off_diag,
        calibration=CalibrationLabel.UNMEASURED,
        probability_interval_valid=True,
        coverage_semantics=CoverageSemantics.MONTE_CARLO_JOINT_SAMPLING.value,
        scenario_bounds=None,
        dependency_gaps=dependency_gaps,
        unknown_leaves=(),
    )


# ---------------------------------------------------------------------------
# Reducible uncertainty and ranking
# ---------------------------------------------------------------------------


def reducible_uncertainty(
    leaves: Sequence[LeafSpec],
    expression: CompiledExpression,
    model: JointModel,
    baseline: TargetSummary,
) -> tuple[UncertaintyContribution, ...]:
    """Estimate each uncertain leaf's contribution to target interval width.

    The procedure fixes one leaf at its P50 (or, for an
    :class:`UnknownMarginal`, the midpoint of its declared admissible
    domain) and reruns the joint sample while keeping every other leaf's
    marginal draws fixed by re-seeding.  This is an executable
    counterfactual rather than a purely verbal claim.

    Results are explicitly model-conditional: when the baseline target
    interval is not a valid probability interval
    (``baseline.probability_interval_valid`` is ``False``) the reducible
    uncertainty ranking cannot be measured as a fraction of a 90 percent
    width, and the function returns an empty ranking.  Callers should
    rely on the scenario bounds and declared coverage semantics instead.
    """

    if not leaves:
        raise ValueError("reducible_uncertainty requires at least one leaf")
    if not baseline.probability_interval_valid or baseline.width is None:
        return ()
    uncertain_ids = [
        leaf.leaf_id
        for leaf in leaves
        if not isinstance(leaf.marginal, ConstantMarginal)
    ]
    if not uncertain_ids:
        return ()

    contributions: list[UncertaintyContribution] = []
    for leaf_id in uncertain_ids:
        altered = [
            LeafSpec(
                leaf_id=other.leaf_id,
                marginal=_to_constant_for_p50(other),
                unit=other.unit,
                measurement_procedure=other.measurement_procedure,
            )
            if other.leaf_id == leaf_id
            else other
            for other in leaves
        ]
        narrowed = joint_sample(altered, expression, model)
        assert narrowed.width is not None
        expected = max(0.0, baseline.width - narrowed.width)
        fraction = expected / baseline.width if baseline.width else 0.0
        contributions.append(
            UncertaintyContribution(
                leaf_id=leaf_id,
                baseline_width=baseline.width,
                narrowed_width=narrowed.width,
                expected_narrowing=expected,
                narrowing_fraction=fraction,
            )
        )

    return tuple(sorted(contributions, key=lambda item: (-item.expected_narrowing, item.leaf_id)))


def _to_constant_for_p50(leaf: LeafSpec) -> Marginal:
    if isinstance(leaf.marginal, ConstantMarginal):
        return leaf.marginal
    if isinstance(leaf.marginal, QuantileFittedMarginal):
        return ConstantMarginal(value=leaf.marginal.p50, rationale="counterfactual_p50")
    if isinstance(leaf.marginal, UnknownMarginal):
        lower, upper = leaf.marginal.domain
        return ConstantMarginal(value=(lower + upper) / 2.0, rationale="counterfactual_unknown_midpoint")
    raise ValueError(f"unsupported marginal: {type(leaf.marginal).__name__}")


def rank_width_reduction(
    contributions: Sequence[UncertaintyContribution],
) -> tuple[WidthReductionRank, ...]:
    """Rank uncertain leaves by expected target interval reduction."""

    items = list(contributions)
    if not items:
        return ()
    max_narrowing = max((item.expected_narrowing for item in items), default=0.0)
    ranked = [
        WidthReductionRank(
            leaf_id=item.leaf_id,
            expected_narrowing=item.expected_narrowing,
            narrowing_fraction=item.narrowing_fraction,
            priority=(
                (item.expected_narrowing / max_narrowing if max_narrowing else 0.0) * 0.7
                + item.narrowing_fraction * 0.3
            ),
        )
        for item in items
    ]
    return tuple(sorted(ranked, key=lambda item: (-item.priority, item.leaf_id)))


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
    "WidthReductionRank",
    "joint_sample",
    "reducible_uncertainty",
    "rank_width_reduction",
]