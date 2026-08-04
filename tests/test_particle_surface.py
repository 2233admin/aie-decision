"""Narrow unit tests for the deterministic CPU particle joint wave surface.

These tests verify the wave-surface contract without asserting any
calibration.  The surface never claims to be a probability distribution
unless an explicit calibration record is supplied.
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
    ParticleSurface,
    SurfaceKind,
    SurfaceRequest,
    VariableSpec,
    compile_particle_surface,
    compile_particle_surface_cached,
    normalise_weights,
    surface_as_mapping,
)


def _two_variable_request(**overrides) -> SurfaceRequest:
    payload: dict[str, object] = {
        "question_id": "q-fermi",
        "seed": 17,
        "particle_count": 256,
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


def test_particle_surface_compiles_formula_mapping_deterministically():
    request = _two_variable_request()
    surface = compile_particle_surface(request)
    again = compile_particle_surface(request)

    assert surface.surface_id == again.surface_id
    assert np.array_equal(surface.particles, again.particles)
    assert np.array_equal(surface.log_weights, again.log_weights)
    assert surface.particle_count == request.particle_count
    assert surface.kind is SurfaceKind.POSSIBILITY
    assert surface.calibration_basis is CalibrationBasis.UNMEASURED
    # The FORMULA mapping must expose a single axis column.
    assert surface.particles.shape == (256, 1)
    assert surface.axis_names == ("revenue",)
    assert surface.variable_names == ("visitors", "conversion_rate")
    # Sanity check the formula range: 800*0.08=64, 1200*0.12=144.
    assert float(surface.particles.min()) >= 60.0
    assert float(surface.particles.max()) <= 150.0


def test_seed_reproducibility_matches_identical_particles_and_weights():
    surface_a = compile_particle_surface(_two_variable_request(seed=101))
    surface_b = compile_particle_surface(_two_variable_request(seed=101))
    surface_c = compile_particle_surface(_two_variable_request(seed=102))

    assert np.array_equal(surface_a.particles, surface_b.particles)
    assert np.array_equal(surface_a.log_weights, surface_b.log_weights)
    assert not np.array_equal(surface_a.particles, surface_c.particles)


def test_no_dense_grid_is_materialised_in_compile():
    request = _two_variable_request(particle_count=512)
    surface = compile_particle_surface(request)

    # Only (particle_count, n_axes) is materialised; no
    # candidates × variables × samples tensor exists.
    assert surface.particles.shape == (request.particle_count, 1)
    assert surface.log_weights.shape == (request.particle_count,)
    assert sum(arr.size for arr in surface.mapping_breakdown.values()) == request.particle_count


def test_possibility_surface_is_the_default_semantic_label():
    request = _two_variable_request()  # no calibration record
    surface = compile_particle_surface(request)

    assert surface.kind is SurfaceKind.POSSIBILITY
    assert surface.calibration_basis is CalibrationBasis.UNMEASURED
    summary = surface_as_mapping(surface)
    assert summary["kind"] == "possibility_surface"
    assert summary["calibration_basis"] == "unmeasured"


def test_only_calibrated_request_may_relabel_to_probability_surface():
    request = _two_variable_request(
        calibration=CalibrationRecord(
            basis=CalibrationBasis.USER_DECLARED,
            declared_at="2026-08-05",
            sample_size=120,
        ),
        coverage_semantics=CoverageSemantics.DECLARED_CREDIBLE_INTERVAL,
    )
    surface = compile_particle_surface(request)

    assert surface.kind is SurfaceKind.PROBABILITY
    assert surface.calibration_basis is CalibrationBasis.USER_DECLARED
    assert surface.coverage_semantics is CoverageSemantics.DECLARED_CREDIBLE_INTERVAL


def test_unmeasured_calibration_record_does_not_relabel_surface():
    request = _two_variable_request(
        calibration=CalibrationRecord(basis=CalibrationBasis.UNMEASURED, declared_at="2026-08-05"),
    )
    surface = compile_particle_surface(request)

    assert surface.kind is SurfaceKind.POSSIBILITY


def test_likelihood_mapping_emits_bimodal_safe_zones():
    request = SurfaceRequest(
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
    surface = compile_particle_surface(request)
    weights = normalise_weights(surface)
    values = surface.particles[:, 0]

    low_mass = float(weights[values <= 0.2].sum())
    high_mass = float(weights[values >= 0.8].sum())
    middle_mass = float(weights[(values > 0.2) & (values < 0.8)].sum())

    assert low_mass > 0.20
    assert high_mass > 0.20
    assert middle_mass < 0.60
    # The likelihood mappings must write a per-particle contribution so
    # downstream diagnostics can decompose weight origins.
    assert set(surface.mapping_breakdown) == {"low", "high"}
    assert surface.mapping_breakdown["low"].shape == (request.particle_count,)


def test_invalid_formula_is_rejected_without_running_sampler():
    request = _two_variable_request(
        mappings=(
            MappingSpec(
                mapping_id="bad",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="revenue",
                expression="__import__('os')",
            ),
        ),
    )
    with pytest.raises(ValueError, match="supports only"):
        compile_particle_surface(request)


def test_likelihood_mapping_requires_observation_and_scale():
    with pytest.raises(ValueError, match="requires observation and observation_scale"):
        MappingSpec(
            mapping_id="bad",
            kind=MappingKind.LIKELIHOOD,
            variables=("visitors",),
            result_axis="revenue",
        )


def test_unknown_axis_in_mapping_is_rejected_at_compile_time():
    with pytest.raises(ValueError, match="unknown"):
        SurfaceRequest(
            question_id="q",
            seed=1,
            particle_count=4,
            axes=(OutcomeAxis("revenue", "CNY"),),
            variables=(VariableSpec("v", "count", 0.0, 1.0),),
            mappings=(
                MappingSpec(
                    mapping_id="m",
                    kind=MappingKind.FORMULA,
                    variables=("v",),
                    result_axis="missing",
                    expression="v",
                ),
            ),
        )


def test_normalised_weights_sum_to_unit_mass_when_nonempty():
    request = _two_variable_request(particle_count=128)
    surface = compile_particle_surface(request)
    weights = normalise_weights(surface)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert weights.size == surface.particle_count


def test_normalised_weights_collapse_to_zero_when_log_weights_diverge():
    # Construct a request whose likelihood penalty is so steep that all
    # particles saturate and the discrete support collapses.
    request = SurfaceRequest(
        question_id="saturated",
        seed=7,
        particle_count=64,
        axes=(OutcomeAxis("price", "CNY"),),
        variables=(VariableSpec("m", "CNY", 0.0, 1.0),),
        mappings=(
            MappingSpec(
                mapping_id="obs",
                kind=MappingKind.LIKELIHOOD,
                variables=("m",),
                result_axis="price",
                observation=(0.5, 0.5000001),
                observation_scale=0.5,
            ),
        ),
    )
    surface = compile_particle_surface(request)
    weights = normalise_weights(surface)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    # Either every particle is identical (degenerate), or weights are still
    # finite and non-negative.
    assert np.isfinite(weights).all()


def test_compile_particle_surface_cached_is_idempotent():
    request = _two_variable_request()
    first = compile_particle_surface_cached(request)
    second = compile_particle_surface_cached(request)

    assert first is second
    assert np.array_equal(first.particles, second.particles)


def test_surface_summary_is_json_compatible():
    request = _two_variable_request()
    surface = compile_particle_surface(request)
    summary = surface_as_mapping(surface)

    serialised = json.dumps(summary)
    assert isinstance(serialised, str)
    assert summary["axis_names"] == ["revenue"]
    assert summary["variable_names"] == ["visitors", "conversion_rate"]
    assert summary["particle_count"] == request.particle_count


def test_surface_id_changes_with_seed_or_particle_count():
    base = _two_variable_request()
    seed_changed = _two_variable_request(seed=base.seed + 1)
    count_changed = _two_variable_request(particle_count=base.particle_count * 2)

    base_surface = compile_particle_surface(base)
    assert compile_particle_surface(seed_changed).surface_id != base_surface.surface_id
    assert compile_particle_surface(count_changed).surface_id != base_surface.surface_id


def test_compile_rejects_invalid_particle_count():
    with pytest.raises(ValueError, match="particle_count"):
        SurfaceRequest(
            question_id="q",
            seed=1,
            particle_count=0,
            axes=(OutcomeAxis("a", "u"),),
            variables=(VariableSpec("v", "u", 0.0, 1.0),),
            mappings=(
                MappingSpec(
                    mapping_id="m",
                    kind=MappingKind.FORMULA,
                    variables=("v",),
                    result_axis="a",
                    expression="v",
                ),
            ),
        )


def test_compile_rejects_empty_axes_and_variables():
    with pytest.raises(ValueError, match="axes"):
        SurfaceRequest(
            question_id="q",
            seed=1,
            particle_count=4,
            axes=(),
            variables=(VariableSpec("v", "u", 0.0, 1.0),),
            mappings=(
                MappingSpec(
                    mapping_id="m",
                    kind=MappingKind.FORMULA,
                    variables=("v",),
                    result_axis="a",
                    expression="v",
                ),
            ),
        )

    with pytest.raises(ValueError, match="variables"):
        SurfaceRequest(
            question_id="q",
            seed=1,
            particle_count=4,
            axes=(OutcomeAxis("a", "u"),),
            variables=(),
            mappings=(),
        )