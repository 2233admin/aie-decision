"""Deterministic, fail-closed probability propagation operations."""

from __future__ import annotations

import math
import random
from math import erf, sqrt
from typing import Mapping, Optional, Sequence

from .probability_contracts import (
    CalibrationLabel,
    CompiledExpression,
    ConstantMarginal,
    CoverageSemantics,
    DependenceCase,
    DistributionFamily,
    JointModel,
    LeafSpec,
    Marginal,
    MarginalKind,
    QuantileFittedMarginal,
    TargetSummary,
    UncertaintyContribution,
    UnknownMarginal,
    WidthReductionRank,
    evaluate_compiled,
    marginal_kind,
)


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
    "joint_sample",
    "reducible_uncertainty",
    "rank_width_reduction"
]
