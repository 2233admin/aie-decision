"""Tests for the probability and joint-sampling module."""

from __future__ import annotations

import math

import pytest

from aie_decision.probability import (
    CalibrationLabel,
    CompiledExpression,
    ConstantMarginal,
    DependenceCase,
    DistributionFamily,
    ExpressionError,
    JointModel,
    LeafSpec,
    MarginalKind,
    QuantileFittedMarginal,
    UnknownMarginal,
    compile_expression,
    evaluate_compiled,
    joint_sample,
    marginal_kind,
    rank_width_reduction,
    reducible_uncertainty,
)

def test_constant_marginal_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        ConstantMarginal(value=float("inf"))

def test_quantile_fitted_marginal_enforces_order() -> None:
    with pytest.raises(ValueError):
        QuantileFittedMarginal(p05=5, p50=1, p95=10)
    with pytest.raises(ValueError):
        QuantileFittedMarginal(p05=1, p50=2, p95=2)

def test_unknown_marginal_requires_reason_and_ordered_domain() -> None:
    with pytest.raises(ValueError):
        UnknownMarginal(reason="", domain=(0.0, 1.0))
    with pytest.raises(ValueError):
        UnknownMarginal(reason="missing", domain=(1.0, 0.0))
    with pytest.raises(ValueError):
        UnknownMarginal(reason="missing", domain=(0.0, 0.0))

def test_marginal_kind_dispatch() -> None:
    assert marginal_kind(ConstantMarginal(value=1)) is MarginalKind.CONSTANT
    assert (
        marginal_kind(QuantileFittedMarginal(p05=1, p50=2, p95=3))
        is MarginalKind.QUANTILE_FITTED
    )
    assert (
        marginal_kind(UnknownMarginal(reason="gap", domain=(0, 1)))
        is MarginalKind.UNKNOWN
    )
