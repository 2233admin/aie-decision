"""MVP deterministic particle surface, evaluation, and diagnostics."""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from wave_mvp_models import (
    CompiledMapping, ParticleSurface, AxisDiagnostic, SurfaceDiagnostics,
    OutcomeAxis, VariableSpec, _sanitize_for_json,
)
from wave_mvp_unit import Dimension
SCHEMA_VERSION = "joint-wave-surface-mvp.v1"

from wave_mvp_expression import _evaluate_compiled

"""MVP deterministic particle surface, evaluation, and diagnostics."""

# Particle plan and evaluation.
# ---------------------------------------------------------------------------


def _latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """Deterministic Latin-hypercube sample on [0, 1)^d with ``n`` points.

    NumPy's Sobol/QMC sampler is not present in the minimal runtime, so the
    MVP ships a seed-deterministic LHS that preserves stratification across
    runs.  The PRD accepts stratified sampling as a Sobol/QMC substitute.
    """
    rng = np.random.default_rng(seed)
    result = np.empty((n, d), dtype=np.float64)
    for axis in range(d):
        perm = rng.permutation(n)
        # Use a sub-stream per axis so the result is independent of the
        # order in which axes are processed.
        axis_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        jitter = axis_rng.random(n)
        result[:, axis] = (perm + jitter) / n
    return result


def _build_particle_plan(
    variables: Sequence[VariableSpec],
    count: int,
    seed: int,
) -> np.ndarray:
    matrix = np.empty((count, len(variables)), dtype=np.float64)
    for index, variable in enumerate(variables):
        lhc = _latin_hypercube(count, 1, seed + index).reshape(-1)
        if variable.bimodal:
            # Mixture of two uniform modes with a clear anti-mode gap so
            # the surface detector can find a stable second mode.  The
            # declared range still bounds both clusters, and the seed
            # governs the deterministic permutation of which particles
            # belong to which mode.
            span = variable.upper - variable.lower
            mode_left_center = variable.lower + 0.15 * span
            mode_right_center = variable.lower + 0.85 * span
            mode_width = 0.1 * span
            half = count // 2
            split = np.concatenate(
                [
                    mode_left_center + (lhc[:half] * 2.0 - 1.0) * mode_width,
                    mode_right_center + (lhc[half:] * 2.0 - 1.0) * mode_width,
                ]
            )
            rng = np.random.default_rng(seed + 7 * index)
            perm = rng.permutation(count)
            matrix[:, index] = split[perm]
        else:
            matrix[:, index] = variable.lower + (variable.upper - variable.lower) * lhc
    return matrix


def _evaluate_surface(
    *,
    compiled_mappings: Sequence[CompiledMapping],
    variables: Sequence[VariableSpec],
    axes: Sequence[OutcomeAxis],
    particle_count: int,
    seed: int,
) -> tuple[ParticleSurface, dict[str, Any]]:
    sample = _build_particle_plan(variables, particle_count, seed)
    log_weights = np.zeros(particle_count, dtype=np.float64)
    axis_index = {axis.name: idx for idx, axis in enumerate(axes)}
    values = np.zeros((particle_count, len(axes)), dtype=np.float64)
    variable_index = {variable.name: idx for idx, variable in enumerate(variables)}
    extra_constants: dict[str, float] = {}
    used_mappings: list[str] = []
    failures: list[str] = []

    for compiled in compiled_mappings:
        if not compiled.is_legal:
            failures.append(compiled.mapping.mapping_id)
            continue
        lookup: dict[str, float] = {}
        for name in compiled.variables:
            if name in extra_constants:
                lookup[name] = extra_constants[name]
            else:
                lookup[name] = float(sample[variable_index[name], 0])
        # Vectorise evaluation across all particles by recursing manually.
        for particle in range(particle_count):
            row: dict[str, float] = {}
            for name in compiled.variables:
                if name in extra_constants:
                    row[name] = extra_constants[name]
                else:
                    row[name] = float(sample[particle, variable_index[name]])
            value = _evaluate_compiled(compiled.compiled, row)
            for axis_name in compiled.mapping.output_axes:
                idx = axis_index[axis_name]
                values[particle, idx] += value
        used_mappings.append(compiled.mapping.mapping_id)

    # Soft log-potential: 1.0 for every non-failing particle, decays with
    # domain violations.  This is intentionally simple so the MVP can be
    # reasoned about without invoking GPU-specific kernels.
    for axis_pos, axis in enumerate(axes):
        column = values[:, axis_pos]
        lower, upper = axis.domain
        mask = (column < lower) | (column > upper)
        log_weights[mask] += -5.0
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = float(weights.sum())
    if total <= 0:
        log_weights = np.full(particle_count, -np.inf)
    else:
        log_weights = np.log(weights / total)

    surface = ParticleSurface(
        axis_order=tuple(axis.name for axis in axes),
        values=values,
        log_weight=log_weights,
        surface_kind="possibility_surface",
        calibration="unmeasured",
        coverage_semantics="declared_joint_input_region",
        evaluator_version=f"{SCHEMA_VERSION}+cpu",
        seed=seed,
    )
    metadata = {
        "used_mappings": used_mappings,
        "failed_mappings": failures,
        "particle_count": particle_count,
    }
    return surface, metadata


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


def _detect_modes(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    min_relative_prominence: float = 0.60,
    min_separation_fraction: float = 0.20,
) -> tuple[int, np.ndarray]:
    """Detect modes via histogram binning with valley-mass scoring.

    Uses the same valley-mass approach as the package evaluator's
    ``_bimodality`` so that the runner oracle and package authority
    produce compatible bimodality classifications.  A mode is detected
    only when the middle bins contain a clear valley between higher
    left and right mass concentrations.
    """
    if values.size < 4:
        if values.size == 0:
            return 0, np.array([], dtype=np.float64)
        return 1, np.array([float(np.median(values))], dtype=np.float64)

    grid_min = float(np.min(values))
    grid_max = float(np.max(values))
    if not np.isfinite(grid_min) or not np.isfinite(grid_max) or grid_min == grid_max:
        return 1, np.array([grid_min], dtype=np.float64)

    span = grid_max - grid_min
    if span <= 0:
        return 1, np.array([grid_min], dtype=np.float64)

    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 1, np.array([grid_min], dtype=np.float64)

    bin_count = min(8, max(3, len(values) // 4))
    bin_width = span / bin_count
    bin_mass = np.zeros(bin_count, dtype=np.float64)
    for idx in range(len(values)):
        bin_idx = int((values[idx] - grid_min) / bin_width)
        if bin_idx >= bin_count:
            bin_idx = bin_count - 1
        bin_mass[bin_idx] += weights[idx]

    max_mass = float(bin_mass.max())
    if max_mass <= 0:
        return 1, np.array([grid_min], dtype=np.float64)

    # Split into three regions: left third, middle third, right third.
    middle_start = bin_count // 3
    middle_end = bin_count - middle_start
    if middle_end - middle_start < 1:
        return 1, np.array([(grid_min + grid_max) * 0.5], dtype=np.float64)

    left_max = float(bin_mass[:middle_start].max()) if middle_start > 0 else 0.0
    right_max = float(bin_mass[middle_end:].max()) if middle_end < bin_count else 0.0
    valley = float(bin_mass[middle_start:middle_end].min())

    left_density = left_max / max_mass
    right_density = right_max / max_mass
    valley_density = valley / max_mass

    # Both edges must have at least 5% mass and the valley must be at most
    # one-third of the weaker edge to count as bimodal.
    if left_density <= 0.05 or right_density <= 0.05:
        return 1, np.array([(grid_min + grid_max) * 0.5], dtype=np.float64)
    if valley_density * 3.0 > min(left_density, right_density):
        # Unimodal: find the bin with max mass as the single mode.
        peak_bin = int(np.argmax(bin_mass))
        centre = grid_min + (peak_bin + 0.5) * bin_width
        return 1, np.array([centre], dtype=np.float64)

    score = min(left_density, right_density) - valley_density
    if score <= 0.15:
        peak_bin = int(np.argmax(bin_mass))
        centre = grid_min + (peak_bin + 0.5) * bin_width
        return 1, np.array([centre], dtype=np.float64)

    # Bimodal: return the two peak locations.
    left_peak_bin = int(np.argmax(bin_mass[:middle_start]))
    right_peak_bin = middle_end + int(np.argmax(bin_mass[middle_end:]))
    left_centre = grid_min + (left_peak_bin + 0.5) * bin_width
    right_centre = grid_min + (right_peak_bin + 0.5) * bin_width
    return 2, np.array([left_centre, right_centre], dtype=np.float64)


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    normalised = weights / total
    sum_sq = float(np.sum(normalised ** 2))
    if sum_sq <= 0:
        return 0.0
    return 1.0 / sum_sq


def _entropy_nats(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    p = weights / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_diagnostics(
    surface: ParticleSurface,
    *,
    regime_split: Mapping[str, Any] | None = None,
) -> SurfaceDiagnostics:
    weights = surface.weight()
    axes: list[AxisDiagnostic] = []
    bimodal: list[str] = []
    failures: list[str] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for axis_pos, axis_name in enumerate(surface.axis_order):
            column = surface.values[:, axis_pos]
            finite_column = column[np.isfinite(column)]
            if finite_column.size == 0:
                width = 0.0
                reference = 0.0
            else:
                width = float(finite_column.max() - finite_column.min())
                reference = float(np.median(finite_column))
            if reference and np.isfinite(reference):
                relative_width = width / abs(reference)
            else:
                relative_width = float("inf") if width > 0 else 0.0
            count, mode_locations = _detect_modes(column, weights)
            if count >= 2:
                bimodal.append(axis_name)
            bins = min(64, max(8, surface.values.shape[0] // 4))
            centers, density = surface.marginal(axis_pos, bins=bins)
            if density.sum() > 0:
                probs = density / density.sum()
                ess_axis = 1.0 / float(np.sum(probs ** 2)) if probs.sum() > 0 else 0.0
                entropy_axis = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
                sharpness_abs = float(width / max(ess_axis, 1.0))
                sharpness_rel = (
                    sharpness_abs / abs(reference) if reference and np.isfinite(reference) else sharpness_abs
                )
            else:
                ess_axis = 0.0
                entropy_axis = 0.0
                sharpness_abs = width
                sharpness_rel = relative_width
            residual = 0.0
            if surface.values.shape[1] > 1:
                for other in range(surface.values.shape[1]):
                    if other == axis_pos:
                        continue
                    other_column = surface.values[:, other]
                    if float(other_column.std()) == 0 or float(column.std()) == 0:
                        continue
                    try:
                        corr = float(np.corrcoef(column, other_column)[0, 1])
                    except (FloatingPointError, ValueError):
                        corr = float("nan")
                    if np.isfinite(corr):
                        residual = max(residual, abs(corr))
            for axis in (regime_split or {}).get("axes", []) if regime_split else []:
                if axis.get("name") == axis_name:
                    lower = axis.get("domain", [None, None])[0]
                    upper = axis.get("domain", [None, None])[1]
                    if lower is not None and float(column.min()) < lower:
                        failures.append(f"{axis_name}_below_domain")
                    if upper is not None and float(column.max()) > upper:
                        failures.append(f"{axis_name}_above_domain")
            axes.append(
                AxisDiagnostic(
                    name=axis_name,
                    unit="",
                    absolute_width=width,
                    relative_width=relative_width,
                    sharpness_absolute=sharpness_abs,
                    sharpness_relative=sharpness_rel,
                    effective_sample_size=ess_axis,
                    entropy_nats=entropy_axis,
                    mode_count=count,
                    mode_locations=tuple(float(loc) for loc in mode_locations),
                    residual_proxy=residual,
                )
            )
    return SurfaceDiagnostics(
        particle_count=int(surface.values.shape[0]),
        axes=tuple(axes),
        bimodal_axes=tuple(bimodal),
        constraint_failures=tuple(failures),
        effective_sample_size=_effective_sample_size(weights),
    )


# ---------------------------------------------------------------------------
