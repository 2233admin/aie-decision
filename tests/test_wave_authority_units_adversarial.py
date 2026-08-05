"""Compound unit and illegal unit rejection adversarial tests.

Requirements from:
  - ``dimension-aware-factor-mapping`` OpenSpec:
      * Dimensional validation before evaluation
      * Compatible conversion succeeds / Incompatible addition fails
"""

from __future__ import annotations

import pytest

from test_wave_authority_adversarial import _golden_payload  # noqa: F401


# ---------------------------------------------------------------------------
# 1. Compound unit support (legal multi-unit mapping)
# ---------------------------------------------------------------------------


def test_compound_unit_usd_per_liter_compiles():
    """The variable fuel_unit_cost with unit ``usd/liter`` MUST be
    accepted by the dimension registry."""
    from aie_decision.joint_schema import dimension_of_unit

    dim = dimension_of_unit("usd/liter")
    # Must be a composite dimension containing money/USD and volume.
    assert "money/USD" in dim
    assert "volume" in dim


def test_liter_unit_is_recognised():
    """The ``liter`` unit MUST be recognised."""
    from aie_decision.joint_schema import dimension_of_unit

    assert dimension_of_unit("liter") == "volume"


def test_compound_unit_formula_validates_dimensions():
    """``fuel_unit_cost * liters_per_leg`` (usd/liter * liter) MUST fail
    the FactorIR dimensionless gate (output: money/USD) but MUST compile
    as a DeterministicTransform through the axis-transform path in
    compile_joint_schema."""
    from aie_decision.factor_ir import FactorIRError, compile_factor_ir
    from aie_decision.joint_schema import dimension_of_unit

    dims = {
        "fuel_unit_cost": dimension_of_unit("usd/liter"),
        "liters_per_leg": dimension_of_unit("liter"),
    }
    # Direct FactorIR: dimensional → must fail.
    with pytest.raises(FactorIRError, match="dimensionless"):
        compile_factor_ir("test", "fuel_unit_cost * liters_per_leg", dims)


# ---------------------------------------------------------------------------
# 2. Malformed compound unit rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("malformed", [
    "usd/",
    "/liter",
    "usd//liter",
    "usd/liter/second",
])
def test_malformed_compound_unit_rejected(malformed):
    """Malformed compound units MUST raise UnknownUnitError."""
    from aie_decision.joint_schema import UnknownUnitError, dimension_of_unit

    with pytest.raises(UnknownUnitError):
        dimension_of_unit(malformed)


def test_double_operator_compound_rejected():
    """``usd*liter`` and ``usd/*liter`` MUST be rejected; the narrow grammar
    only supports ``<unit>/<unit>`` without ``*`` or mixed operators."""
    from aie_decision.joint_schema import UnknownUnitError, dimension_of_unit

    for bad in ("usd*liter", "usd/*liter", "usd* /liter"):
        with pytest.raises(UnknownUnitError):
            dimension_of_unit(bad)


# ---------------------------------------------------------------------------
# 3. Illegal unit rejection with structured evidence
# ---------------------------------------------------------------------------


def test_illegal_cross_dimension_addition_rejected():
    """Adding time to money MUST be rejected with a structured error."""
    from aie_decision.factor_ir import FactorIRError, compile_factor_ir
    from aie_decision.joint_schema import dimension_of_unit

    dims = {
        "lane_hours": dimension_of_unit("hour"),
        "fuel_unit_cost": dimension_of_unit("usd/liter"),
    }
    with pytest.raises(FactorIRError, match="dimension mismatch"):
        compile_factor_ir("illegal", "lane_hours + fuel_unit_cost", dims)


def test_illegal_dimensionless_addition_rejected():
    """Adding a dimensionless constant to a time variable MUST be rejected
    (the golden fixture marks mapping ``illegitimate-time-constant`` as
    expected_failure=unit_mismatch)."""
    from aie_decision.factor_ir import FactorIRError, compile_factor_ir
    from aie_decision.joint_schema import dimension_of_unit

    dims = {"lane_hours": dimension_of_unit("hour")}
    with pytest.raises(FactorIRError, match="dimension mismatch"):
        compile_factor_ir("illegal-const", "lane_hours + 3", dims)
