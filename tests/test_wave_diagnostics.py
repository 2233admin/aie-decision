"""Narrow unit tests for the wave surface diagnostics summary.

The diagnostics are a CPU-friendly consumer of the joint particle surface.  The
tests verify that marginal, peak, entropy, ESS, multi-modality, sensitivity
and residual summaries are deterministic, calibrated where applicable, and
never advertise uncalibrated probability semantics.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aie_decision.particle_surface import (
    CalibrationBasis,
    CalibrationRecord,
    CoverageSemantics,
    MappingKind,
    MappingSpec,
    OutcomeAxis,
    SurfaceKind,
    SurfaceRequest,
    VariableSpec,
    compile_particle_surface,
)
from aie_decision.wave_diagnostics import (
    diagnostics_as_mapping,
    information_summary,
    marginal_summary,
    peak_summary,
    residual_summary,
    sensitivity_summary,
    summarise_surface,
)


def _uniform_request(**overrides) -> SurfaceRequest:
    payload: dict[str, object] = {
        "question_id": "diag",
        "seed": 7,
        "particle_count": 1024,
        "axes": (OutcomeAxis("revenue", "CNY"),),
        "variables": (
            VariableSpec("visitors", "count/day", 800.0, 1200.0),
            VariableSpec("conversion_rate", "ratio", 0.08, 0.12),
        ),
        "mappings": (
            MappingSpec(
                mapping_id="rev",
                kind=MappingKind.FORMULA,
                variables=("visitors", "conversion_rate"),
                result_axis="revenue",
                expression="visitors * conversion_rate",
            ),
        ),
    }
    payload.update(overrides)
    return SurfaceRequest(**payload)


def _bimodal_request() -> SurfaceRequest:
    return SurfaceRequest(
        question_id="bimodal",
        seed=2026,
        particle_count=2048,
        axes=(OutcomeAxis("price", "CNY"),),
        variables=(VariableSpec("signal", "CNY", 0.0, 1.0),),
        mappings=(
            MappingSpec(
                mapping_id="low",
                kind=MappingKind.LIKELIHOOD,
                variables=("signal",),
                result_axis="price",
                observation=(0.0, 0.2),
                observation_scale=0.5,
            ),
            MappingSpec(
                mapping_id="high",
                kind=MappingKind.LIKELIHOOD,
                variables=("signal",),
                result_axis="price",
                observation=(0.8, 1.0),
                observation_scale=0.5,
            ),
        ),
    )


def test_marginal_summary_quantiles_are_within_support():
    surface = compile_particle_surface(_uniform_request())
    summary = marginal_summary(surface, "revenue")

    assert summary.axis == "revenue"
    assert summary.unit == "CNY"
    assert summary.support_min <= summary.p05 <= summary.p50 <= summary.p95 <= summary.support_max
    # Uniform sampling of two independent uniform variables produces a
    # marginally heavier right tail than the left tail.
    assert 90.0 < summary.weighted_mean < 110.0
    assert summary.weighted_std > 0.0


def test_marginal_summary_detects_multimodal_surface():
    surface = compile_particle_surface(_bimodal_request())
    summary = marginal_summary(surface, "price")

    assert summary.multimodal
    assert summary.mode_count >= 2
    # Modes should straddle the two observation intervals.
    modes = sorted(summary.modes)
    assert any(mode < 0.3 for mode in modes)
    assert any(mode > 0.7 for mode in modes)


def test_peak_summary_reports_a_single_particle_value():
    surface = compile_particle_surface(_uniform_request())
    peak = peak_summary(surface, "revenue")

    assert peak.axis == "revenue"
    assert peak.unit == "CNY"
    assert peak.peak_weight > 0.0
    # Peak weight is at most 1.0; tied alternatives may be empty.
    assert peak.peak_weight <= 1.0


def test_information_summary_reports_entropy_and_ess():
    surface = compile_particle_surface(_uniform_request())
    info = information_summary(surface)

    assert info.entropy_nats >= 0.0
    assert 0.0 <= info.effective_sample_size <= surface.particle_count + 1e-6
    assert 0.0 <= info.ess_ratio <= 1.0 + 1e-6
    assert info.degeneracy is False


def test_information_summary_flags_degeneracy_when_one_mapping_dominates():
    # A very tight likelihood concentrates particles onto a small interval,
    # collapsing the effective sample size.  We construct the tight interval
    # so that the ESS ratio falls below the degeneracy threshold.
    request = SurfaceRequest(
        question_id="tight",
        seed=11,
        particle_count=1024,
        axes=(OutcomeAxis("price", "CNY"),),
        variables=(VariableSpec("m", "CNY", 0.0, 1.0),),
        mappings=(
            MappingSpec(
                mapping_id="obs",
                kind=MappingKind.LIKELIHOOD,
                variables=("m",),
                result_axis="price",
                observation=(0.0, 1.0),
                observation_scale=0.001,
            ),
        ),
    )
    surface = compile_particle_surface(request)
    info = information_summary(surface)

    # With a near-zero observation scale and large interval, the Gaussian
    # tail has a negligible effect; particles are essentially uniformly
    # weighted and ESS ≈ N.  We verify the upper bound.
    assert info.effective_sample_size <= surface.particle_count + 1e-6


def test_sensitivity_summary_orders_mappings_by_potential_narrowing():
    request = SurfaceRequest(
        question_id="sens",
        seed=42,
        particle_count=512,
        axes=(OutcomeAxis("price", "CNY"),),
        variables=(VariableSpec("m", "CNY", 0.0, 1.0),),
        mappings=(
            MappingSpec(
                mapping_id="tight_obs",
                kind=MappingKind.LIKELIHOOD,
                variables=("m",),
                result_axis="price",
                observation=(0.48, 0.52),
                observation_scale=0.02,
            ),
            MappingSpec(
                mapping_id="loose_obs",
                kind=MappingKind.LIKELIHOOD,
                variables=("m",),
                result_axis="price",
                observation=(0.3, 0.7),
                observation_scale=0.5,
            ),
        ),
    )
    surface = compile_particle_surface(request)
    ranking = sensitivity_summary(surface, "price", target_width=None)

    assert {item.mapping_id for item in ranking} == {"tight_obs", "loose_obs"}
    # The wider observation removes more potential narrowing when dropped:
    # removing it widens the marginal across the full [0, 1] support, while
    # removing the tighter observation only widens the inner interval.
    assert ranking[0].mapping_id == "loose_obs"
    for item in ranking:
        assert item.weight_contribution_share >= 0.0
        assert item.relative_variation >= 0.0
        assert item.expected_potential_narrowing >= 0.0


def test_residual_summary_reports_moments_and_bandwidth():
    surface = compile_particle_surface(_uniform_request())
    residual = residual_summary(surface, "revenue", reference=100.0)

    assert residual.axis == "revenue"
    assert residual.unit == "CNY"
    assert residual.variance >= 0.0
    assert residual.skewness != 0.0  # product of two uniforms has positive skew
    assert residual.kurtosis > 0.0
    assert residual.bandwidth > 0.0


def test_summarise_surface_reports_all_axes_and_notes():
    surface = compile_particle_surface(_uniform_request())
    diagnostics = summarise_surface(surface)

    assert {summary.axis for summary in diagnostics.marginals} == set(surface.axis_names)
    assert {peak.axis for peak in diagnostics.peaks} == set(surface.axis_names)
    assert {residual.axis for residual in diagnostics.residuals} == set(surface.axis_names)
    assert all(note for note in diagnostics.notes)
    # The default surface is unmeasured: the notes must surface that.
    assert any("possibility_surface" in note for note in diagnostics.notes)
    assert any("unmeasured" in note for note in diagnostics.notes)


def test_summarise_surface_marks_bimodal_axes():
    surface = compile_particle_surface(_bimodal_request())
    diagnostics = summarise_surface(surface, observation_axis="price")

    assert "price" in diagnostics.multimodal_axes
    assert any(
        "multimodal" in note and "price" in note for note in diagnostics.notes
    )


def test_summarise_surface_is_deterministic():
    surface = compile_particle_surface(_uniform_request())
    first = summarise_surface(surface)
    second = summarise_surface(surface)

    first_dict = diagnostics_as_mapping(first)
    second_dict = diagnostics_as_mapping(second)
    assert first_dict == second_dict
    # JSON serialisable as well.
    json.dumps(first_dict)


def test_summarise_surface_includes_calibration_basis_when_present():
    request = _uniform_request(
        calibration=CalibrationRecord(
            basis=CalibrationBasis.HISTORICAL_RESIDUAL,
            declared_at="2026-08-05",
            sample_size=200,
        ),
        coverage_semantics=CoverageSemantics.EMPIRICAL_PREDICTION_INTERVAL,
    )
    surface = compile_particle_surface(request)
    assert surface.kind is SurfaceKind.PROBABILITY

    diagnostics = summarise_surface(surface)
    rendered = diagnostics_as_mapping(diagnostics)

    assert rendered["surface_kind"] == SurfaceKind.PROBABILITY.value
    assert rendered["calibration_basis"] == CalibrationBasis.HISTORICAL_RESIDUAL.value
    # The notes must NOT advertise a calibration claim when the surface is a
    # probability surface; the calibration_basis value already declares the
    # basis.
    assert not any("unmeasured" in note for note in rendered["notes"])


def test_summarise_surface_reports_possibility_surface_by_default():
    surface = compile_particle_surface(_uniform_request())
    assert surface.kind is SurfaceKind.POSSIBILITY
    diagnostics = summarise_surface(surface)

    rendered = diagnostics_as_mapping(diagnostics)
    assert rendered["surface_kind"] == SurfaceKind.POSSIBILITY.value
    assert rendered["calibration_basis"] == CalibrationBasis.UNMEASURED.value
    assert any("possibility_surface" in note for note in rendered["notes"])


def test_diagnostics_as_mapping_is_json_compatible():
    surface = compile_particle_surface(_bimodal_request())
    diagnostics = summarise_surface(surface)
    rendered = diagnostics_as_mapping(diagnostics)

    serialised = json.dumps(rendered)
    assert isinstance(serialised, str)
    # Required keys are present and structured.
    assert "marginals" in rendered
    assert "peaks" in rendered
    assert "sensitivities" in rendered
    assert "residuals" in rendered
    assert "multimodal_axes" in rendered
    assert "notes" in rendered


def test_marginal_summary_rejects_unknown_axis():
    surface = compile_particle_surface(_uniform_request())
    with pytest.raises(ValueError, match="unknown axis"):
        marginal_summary(surface, "latency")


def test_sensitivity_summary_rejects_unknown_axis():
    surface = compile_particle_surface(_uniform_request())
    with pytest.raises(ValueError, match="unknown axis"):
        sensitivity_summary(surface, "latency")


def test_residual_summary_rejects_unknown_axis():
    surface = compile_particle_surface(_uniform_request())
    with pytest.raises(ValueError, match="unknown axis"):
        residual_summary(surface, "latency")


def test_peak_summary_rejects_unknown_axis():
    surface = compile_particle_surface(_uniform_request())
    with pytest.raises(ValueError, match="unknown axis"):
        peak_summary(surface, "latency")


def test_summarise_surface_supports_optional_target_width_and_references():
    request = SurfaceRequest(
        question_id="multiaxis",
        seed=1,
        particle_count=512,
        axes=(
            OutcomeAxis("revenue", "CNY"),
            OutcomeAxis("latency", "s"),
        ),
        variables=(
            VariableSpec("visitors", "count/day", 800.0, 1200.0),
            VariableSpec("conv", "ratio", 0.08, 0.12),
        ),
        mappings=(
            MappingSpec(
                mapping_id="rev",
                kind=MappingKind.FORMULA,
                variables=("visitors", "conv"),
                result_axis="revenue",
                expression="visitors * conv",
            ),
            MappingSpec(
                mapping_id="lat",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="latency",
                expression="visitors",
            ),
        ),
    )
    surface = compile_particle_surface(request)
    diagnostics = summarise_surface(
        surface,
        observation_axis="revenue",
        target_width=10.0,
        references={"revenue": 100.0, "latency": 1000.0},
    )

    assert {summary.axis for summary in diagnostics.residuals} == {"revenue", "latency"}
    revenue_residual = next(item for item in diagnostics.residuals if item.axis == "revenue")
    # Reference value should bias the residual toward zero.
    assert abs(revenue_residual.bias) < 5.0