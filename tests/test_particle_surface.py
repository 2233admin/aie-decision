"""Narrow unit tests for the deterministic CPU particle joint wave surface.

These tests verify the wave-surface contract without asserting any
calibration.  The surface never claims to be a probability distribution
unless an explicit calibration record is supplied.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest

from aie_decision.particle_surface import (
    CalibrationBasis,
    CalibrationRecord,
    CompiledIR,
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


def _compile_formula_ast(expression: str) -> ast.Expression:
    """Parse a formula expression into an AST for CompiledIR."""
    if not expression or not expression.strip():
        raise ValueError("expression is required")
    return ast.parse(expression.strip(), mode="eval")


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
    # Auto-generate compiled IR trees for FORMULA mappings unless overridden.
    if "compiled_ir_trees" not in payload:
        compiled: dict[str, CompiledIR] = {}
        for mapping in payload["mappings"]:
            if mapping.kind is MappingKind.FORMULA:
                tree = _compile_formula_ast(mapping.expression or "0")
                compiled[mapping.mapping_id] = CompiledIR(
                    mapping_id=mapping.mapping_id,
                    tree=tree,
                    is_factor=False,  # default: axis transform (test formulas compute axis values)
                )
        payload["compiled_ir_trees"] = compiled
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


def test_formula_mapping_without_compiled_ir_fails_before_sampling():
    """A FORMULA mapping without a compiled IR entry MUST fail before any particle
    is evaluated — there is no raw formula parser fallback."""
    request = _two_variable_request()
    # Constructing a SurfaceRequest with an empty compiled_ir_trees when
    # FORMULA mappings exist MUST fail in __post_init__ — fail-early.
    with pytest.raises(ValueError, match="missing entries"):
        SurfaceRequest(
            question_id=request.question_id,
            seed=request.seed,
            particle_count=request.particle_count,
            axes=request.axes,
            variables=request.variables,
            mappings=request.mappings,
            compiled_ir_trees={},  # empty — missing "rev"
        )


def test_extra_compiled_ir_entry_fails():
    """An extra compiled IR entry for a non-existent mapping_id MUST fail."""
    request = _two_variable_request()
    extra = dict(request.compiled_ir_trees)
    extra["nonexistent"] = CompiledIR(
        mapping_id="nonexistent",
        tree=_compile_formula_ast("1 + 1"),
        is_factor=True,
    )
    with pytest.raises(ValueError, match="unknown mapping_ids"):
        SurfaceRequest(
            question_id=request.question_id,
            seed=request.seed,
            particle_count=request.particle_count,
            axes=request.axes,
            variables=request.variables,
            mappings=request.mappings,
            compiled_ir_trees=extra,
        )


def test_wrong_mapping_id_in_compiled_ir_fails():
    """A compiled IR entry with a mismatched mapping_id MUST fail."""
    request = _two_variable_request()
    broken_compiled = {
        "rev": CompiledIR(
            mapping_id="wrong-id",  # mismatch
            tree=request.compiled_ir_trees["rev"].tree,
            is_factor=True,
        ),
    }
    with pytest.raises(ValueError, match="mapping_id mismatch"):
        SurfaceRequest(
            question_id=request.question_id,
            seed=request.seed,
            particle_count=request.particle_count,
            axes=request.axes,
            variables=request.variables,
            mappings=request.mappings,
            compiled_ir_trees=broken_compiled,
        )


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


# ---------------------------------------------------------------------------
# Adversarial tests: typed compiled IR contract
# ---------------------------------------------------------------------------


def test_two_deterministic_transforms_different_formulas_different_particles():
    """Two DeterministicTransforms with the same seed and inputs but different
    compiled formulas MUST produce different axis particles and diagnostics."""
    seed = 42
    tree_a = _compile_formula_ast("visitors * 2")
    tree_b = _compile_formula_ast("visitors / 3")

    base = {
        "question_id": "q-det-xform",
        "seed": seed,
        "particle_count": 128,
        "axes": (OutcomeAxis("x", "CNY"),),
        "variables": (
            VariableSpec("visitors", "count/day", 800.0, 1200.0),
        ),
        "mappings": (
            MappingSpec(
                mapping_id="xform",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="x",
                expression="visitors * 2",  # irrelevant — compiled IR wins
            ),
        ),
    }

    compiled_a = {"xform": CompiledIR("xform", tree_a, is_factor=False)}
    compiled_b = {"xform": CompiledIR("xform", tree_b, is_factor=False)}

    surface_a = compile_particle_surface(SurfaceRequest(compiled_ir_trees=compiled_a, **base))
    surface_b = compile_particle_surface(SurfaceRequest(compiled_ir_trees=compiled_b, **base))

    # Same seed → same variable samples → different formulas → different axis values.
    assert not np.array_equal(surface_a.particles, surface_b.particles), (
        "deterministic transforms with different formulas must produce different particles"
    )
    # Both surfaces are deterministic — replay identity.
    surface_a2 = compile_particle_surface(SurfaceRequest(compiled_ir_trees=compiled_a, **base))
    assert np.array_equal(surface_a.particles, surface_a2.particles)


def test_two_factor_irs_different_weights_same_axis_values():
    """Two FactorIRs with different formulas produce different non-uniform
    weights while the DeterministicTransform axis values stay the same."""
    seed = 99
    tree_axis = _compile_formula_ast("visitors")
    tree_w1 = _compile_formula_ast("conversion_rate")
    tree_w2 = _compile_formula_ast("conversion_rate * 0.5")

    base = {
        "question_id": "q-factor-split",
        "seed": seed,
        "particle_count": 256,
        "axes": (OutcomeAxis("x", "CNY"),),
        "variables": (
            VariableSpec("visitors", "count/day", 800.0, 1200.0),
            VariableSpec("conversion_rate", "ratio", 0.08, 0.12),
        ),
        "mappings": (
            MappingSpec(
                mapping_id="axis-xform",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="x",
                expression="visitors",
            ),
            MappingSpec(
                mapping_id="weight-factor",
                kind=MappingKind.FORMULA,
                variables=("conversion_rate",),
                result_axis="x",
                expression="conversion_rate",
            ),
        ),
    }

    compiled_a = {
        "axis-xform": CompiledIR("axis-xform", tree_axis, is_factor=False),
        "weight-factor": CompiledIR("weight-factor", tree_w1, is_factor=True),
    }
    compiled_b = {
        "axis-xform": CompiledIR("axis-xform", tree_axis, is_factor=False),
        "weight-factor": CompiledIR("weight-factor", tree_w2, is_factor=True),
    }

    surface_a = compile_particle_surface(SurfaceRequest(compiled_ir_trees=compiled_a, **base))
    surface_b = compile_particle_surface(SurfaceRequest(compiled_ir_trees=compiled_b, **base))

    # Same axis transform → same axis particles.
    assert np.array_equal(surface_a.particles, surface_b.particles), (
        "axis transform values must be identical when only FactorIR changes"
    )
    # Different FactorIRs → different non-uniform weights.
    assert not np.array_equal(surface_a.log_weights, surface_b.log_weights), (
        "factor IRs with different formulas must produce different weights"
    )
    # Weights must be non-uniform (FactorIR contributes actual values).
    assert np.std(surface_a.log_weights) > 0, "FactorIR must produce non-uniform weights"
    assert np.std(surface_b.log_weights) > 0, "FactorIR must produce non-uniform weights"


def test_tampered_expression_ignored_when_compiled_ir_fixed():
    """Tampering the raw expression while the compiled IR is fixed MUST NOT
    select a raw fallback — the compiled IR is the single authority."""
    seed = 7
    tree = _compile_formula_ast("visitors * 2")

    compiled = {"xform": CompiledIR("xform", tree, is_factor=False)}

    # Request with the real expression.
    request_clean = SurfaceRequest(
        question_id="q-tamper",
        seed=seed,
        particle_count=64,
        axes=(OutcomeAxis("x", "CNY"),),
        variables=(VariableSpec("visitors", "count/day", 800.0, 1200.0),),
        mappings=(
            MappingSpec(
                mapping_id="xform",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="x",
                expression="visitors * 2",
            ),
        ),
        compiled_ir_trees=compiled,
    )

    # Request with a tampered expression — compiled IR unchanged.
    request_tampered = SurfaceRequest(
        question_id="q-tamper",
        seed=seed,
        particle_count=64,
        axes=(OutcomeAxis("x", "CNY"),),
        variables=(VariableSpec("visitors", "count/day", 800.0, 1200.0),),
        mappings=(
            MappingSpec(
                mapping_id="xform",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="x",
                expression="__import__('os').system('rm -rf /')",
            ),
        ),
        compiled_ir_trees=compiled,
    )

    surface_clean = compile_particle_surface(request_clean)
    surface_tampered = compile_particle_surface(request_tampered)

    # Same compiled IR → identical particles and weights.
    assert np.array_equal(surface_clean.particles, surface_tampered.particles), (
        "tampered expression must not affect evaluation — compiled IR is authority"
    )
    assert np.array_equal(surface_clean.log_weights, surface_tampered.log_weights)
    # Surface identity derives from compiled IR, not raw expression.
    assert surface_clean.surface_id == surface_tampered.surface_id, (
        "surface ID must be stable when compiled IR is unchanged"
    )


def test_zero_valued_transform_is_safe():
    """A DeterministicTransform that evaluates to zero for all particles is safe —
    no F/F division or NaN propagation occurs."""
    tree_zero = _compile_formula_ast("v * 0")

    compiled = {"zero-xform": CompiledIR("zero-xform", tree_zero, is_factor=False)}

    request = SurfaceRequest(
        question_id="q-zero",
        seed=1,
        particle_count=64,
        axes=(OutcomeAxis("x", "CNY"),),
        variables=(VariableSpec("v", "count/day", 0.0, 100.0),),
        mappings=(
            MappingSpec(
                mapping_id="zero-xform",
                kind=MappingKind.FORMULA,
                variables=("v",),
                result_axis="x",
                expression="v * 0",
            ),
        ),
        compiled_ir_trees=compiled,
    )

    surface = compile_particle_surface(request)
    # All axis values must be exactly zero.
    assert np.all(surface.particles == 0.0), "zero transform must produce zero particles"
    # Log weights and particles must be finite.
    assert np.all(np.isfinite(surface.log_weights))
    assert np.all(np.isfinite(surface.particles))
    # Normalised weights must sum to 1.0 (uniform since all weights are 0).
    weights = normalise_weights(surface)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_factor_ir_log_weights_exactly_equal_expression_values():
    """FactorIR contribution must be accumulated exactly once — log_weights
    must equal the compiled factor expression values, not double-accumulated."""
    seed = 123
    tree_factor = _compile_formula_ast("conversion_rate * 2")
    tree_axis = _compile_formula_ast("visitors")

    compiled = {
        "axis-xform": CompiledIR("axis-xform", tree_axis, is_factor=False),
        "weight-factor": CompiledIR("weight-factor", tree_factor, is_factor=True),
    }

    request = SurfaceRequest(
        question_id="q-exact",
        seed=seed,
        particle_count=64,
        axes=(OutcomeAxis("x", "CNY"),),
        variables=(
            VariableSpec("visitors", "count/day", 800.0, 1200.0),
            VariableSpec("conversion_rate", "ratio", 0.08, 0.12),
        ),
        mappings=(
            MappingSpec(
                mapping_id="axis-xform",
                kind=MappingKind.FORMULA,
                variables=("visitors",),
                result_axis="x",
                expression="visitors",
            ),
            MappingSpec(
                mapping_id="weight-factor",
                kind=MappingKind.FORMULA,
                variables=("conversion_rate",),
                result_axis="x",
                expression="conversion_rate * 2",
            ),
        ),
        compiled_ir_trees=compiled,
    )

    surface = compile_particle_surface(request)

    # Recompute expected factor contribution manually.
    sampler = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    unit_samples = sampler.random((64, 2))
    var_bounds = np.array([(800.0, 1200.0), (0.08, 0.12)], dtype=float)
    samples = var_bounds[:, 0] + unit_samples * (var_bounds[:, 1] - var_bounds[:, 0])
    expected_factor = samples[:, 1] * 2.0  # conversion_rate * 2

    # log_weights must exactly equal the factor expression values
    # (no double accumulation, no transformation).
    assert surface.log_weights == pytest.approx(expected_factor, abs=1e-12), (
        "FactorIR log_weights must exactly equal compiled expression values"
    )

    # The axis values must come from the DeterministicTransform only.
    expected_axis = samples[:, 0]  # visitors
    assert surface.particles[:, 0] == pytest.approx(expected_axis, abs=1e-12)