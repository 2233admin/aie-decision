"""Wave surface diagnostics: marginal, peak, entropy, ESS, multi-modality,
sensitivity, residual.

Every diagnostic in this module consumes a :class:`ParticleSurface` and returns
plain Python dictionaries that can be serialised into the existing ledger.
The module never claims the surface is a calibrated probability distribution;
it surfaces the empirical summaries the search controller needs to decide
whether to keep iterating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .particle_surface import (
    CalibrationBasis,
    ParticleSurface,
    SurfaceKind,
    normalise_weights,
)


# ----------------------------------------------------------------------------
# Numeric helpers
# ----------------------------------------------------------------------------


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Weighted empirical quantile with linear interpolation.

    Falls back to the unweighted ``np.quantile`` when the weight mass collapses
    so the caller still receives a finite summary value.
    """

    if values.size == 0:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return float(np.quantile(values, quantile))
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / total
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return float(np.quantile(values, quantile))
    # Linear interpolation between adjacent CDF values.
    upper = np.searchsorted(cumulative, quantile, side="right")
    upper = min(upper, cumulative.size - 1)
    lower = max(upper - 1, 0)
    if upper == lower:
        return float(sorted_values[upper])
    span = cumulative[upper] - cumulative[lower]
    if span <= 0:
        return float(sorted_values[lower])
    fraction = (quantile - cumulative[lower]) / span
    return float(sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower]))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return float(values.mean())
    return float(np.dot(values, weights) / total)


def _weighted_moment(
    values: np.ndarray, weights: np.ndarray, order: int, centre: float | None = None
) -> float:
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return 0.0
    if centre is None:
        centre = _weighted_mean(values, weights)
    centred = values - centre
    return float(np.dot(centred ** order, weights) / total)


def _entropy(weights: np.ndarray) -> float:
    """Discrete Shannon entropy in nats of the unnormalised ``weights``.

    Returns zero when the support collapses to a single particle so the caller
    can interpret the surface as deterministic.
    """

    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return 0.0
    normalised = weights / total
    # Keep the convention that entropy is zero when the support shrinks to a
    # single particle; the underlying arithmetic stays in nats.
    support = normalised[normalised > 0]
    if support.size <= 1:
        return 0.0
    return float(-np.sum(support * np.log(support)))


def _effective_sample_size(weights: np.ndarray) -> float:
    """Effective sample size using ``1 / sum(w_i^2)`` on the discrete weights."""

    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return 0.0
    normalised = weights / total
    denom = float(np.sum(normalised ** 2))
    if denom <= 0 or not math.isfinite(denom):
        return 0.0
    return float(1.0 / denom)


def _gaussian_kde_bandwidth(values: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for a 1-D sample.

    The diagnostic uses it for a coarse residual sanity check; it never affects
    the underlying particle representation.
    """

    n = values.size
    if n < 2:
        return 1.0
    std = float(values.std(ddof=1)) if n > 1 else 1.0
    if std <= 0 or not math.isfinite(std):
        return 1.0
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread <= 0 or not math.isfinite(spread):
        return 1.0
    return float(1.06 * spread * (n ** (-1.0 / 5.0)))


def _detect_modes(values: np.ndarray, weights: np.ndarray, max_modes: int = 5) -> tuple[float, ...]:
    """Detect a small set of weighted modes using a coarse 1-D density.

    The function never returns more than ``max_modes`` peaks and only operates
    on the marginal of a single axis so the cost stays linear in the particle
    count.
    """

    if values.size == 0:
        return ()
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return ()
    normalised = weights / total
    spread = float(values.max() - values.min())
    if spread <= 0 or not math.isfinite(spread):
        # Deterministic support; the single value is the only mode.
        return (float(values[0]),)
    bin_count = max(8, min(64, int(math.sqrt(values.size))))
    edges = np.linspace(values.min(), values.max(), bin_count + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = np.zeros(bin_count, dtype=float)
    bin_indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bin_count - 1)
    np.add.at(density, bin_indices, normalised)
    if density.sum() <= 0:
        return ()
    density = density / density.sum()
    if density.size < 3:
        return (float(centres[int(np.argmax(density))]),)
    # Local maxima: greater than both neighbours.
    interior = density[1:-1]
    left = density[:-2]
    right = density[2:]
    is_peak = (interior > left) & (interior >= right)
    peak_indices = np.where(is_peak)[0] + 1
    # Always include the global maximum so the caller receives at least one mode.
    global_index = int(np.argmax(density))
    if global_index not in peak_indices:
        peak_indices = np.append(peak_indices, global_index)
    peak_density = density[peak_indices]
    order = np.argsort(-peak_density)
    ordered = peak_indices[order][:max_modes]
    ordered.sort()
    return tuple(float(centres[index]) for index in ordered)


# ----------------------------------------------------------------------------
# Public summary records
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarginalSummary:
    axis: str
    unit: str
    weighted_mean: float
    weighted_std: float
    p05: float
    p50: float
    p95: float
    support_min: float
    support_max: float
    modes: tuple[float, ...]
    mode_count: int
    multimodal: bool


@dataclass(frozen=True, slots=True)
class PeakSummary:
    axis: str
    unit: str
    peak_value: float
    peak_weight: float
    alternatives: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class InformationSummary:
    entropy_nats: float
    effective_sample_size: float
    ess_ratio: float
    degeneracy: bool


@dataclass(frozen=True, slots=True)
class SensitivitySummary:
    mapping_id: str
    variable: str | None
    weight_contribution_share: float
    relative_variation: float
    expected_potential_narrowing: float


@dataclass(frozen=True, slots=True)
class ResidualSummary:
    axis: str
    unit: str
    bias: float
    absolute_bias: float
    variance: float
    skewness: float
    kurtosis: float
    bandwidth: float


@dataclass(frozen=True, slots=True)
class WaveDiagnostics:
    surface_id: str
    question_id: str
    surface_kind: str
    calibration_basis: str
    particle_count: int
    entropy: InformationSummary
    marginals: tuple[MarginalSummary, ...]
    peaks: tuple[PeakSummary, ...]
    sensitivities: tuple[SensitivitySummary, ...]
    residuals: tuple[ResidualSummary, ...]
    multimodal_axes: tuple[str, ...]
    notes: tuple[str, ...]


# ----------------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------------


def marginal_summary(surface: ParticleSurface, axis: str) -> MarginalSummary:
    """Weighted marginal summary for a single axis."""

    if axis not in surface.axis_names:
        raise ValueError(f"unknown axis: {axis}")
    axis_index = surface.axis_names.index(axis)
    unit = surface.axis_units[axis_index]
    values = surface.particles[:, axis_index]
    weights = normalise_weights(surface)
    mean = _weighted_mean(values, weights)
    std = math.sqrt(max(0.0, _weighted_moment(values, weights, 2, mean)))
    support_min = float(values.min())
    support_max = float(values.max())
    modes = _detect_modes(values, weights)
    p05 = _weighted_quantile(values, weights, 0.05)
    p50 = _weighted_quantile(values, weights, 0.50)
    p95 = _weighted_quantile(values, weights, 0.95)
    return MarginalSummary(
        axis=axis,
        unit=unit,
        weighted_mean=mean,
        weighted_std=std,
        p05=p05,
        p50=p50,
        p95=p95,
        support_min=support_min,
        support_max=support_max,
        modes=modes,
        mode_count=len(modes),
        multimodal=len(modes) > 1,
    )


def peak_summary(surface: ParticleSurface, axis: str) -> PeakSummary:
    """Highest-weight particles along a single axis.

    The function uses the discrete weights rather than a fitted density so the
    peak report remains auditable; ties are broken by the smaller axis value
    for reproducibility.
    """

    if axis not in surface.axis_names:
        raise ValueError(f"unknown axis: {axis}")
    axis_index = surface.axis_names.index(axis)
    unit = surface.axis_units[axis_index]
    values = surface.particles[:, axis_index]
    weights = normalise_weights(surface)
    order = np.lexsort((values, -weights))
    peak_index = int(order[0])
    peak_value = float(values[peak_index])
    peak_weight = float(weights[peak_index])
    # Identify alternatives by selecting particles within a multiplicative band
    # of the peak weight; the threshold is deliberately small so the report
    # exposes only meaningfully different particles.
    if peak_weight > 0:
        threshold = max(peak_weight * 0.05, 1e-12)
    else:
        threshold = 0.0
    alternatives_set: set[float] = set()
    for index in order:
        if weights[index] >= threshold and index != peak_index:
            alternatives_set.add(float(values[index]))
    alternatives = tuple(sorted(alternatives_set))
    return PeakSummary(
        axis=axis,
        unit=unit,
        peak_value=peak_value,
        peak_weight=peak_weight,
        alternatives=alternatives,
    )


def information_summary(surface: ParticleSurface) -> InformationSummary:
    weights = normalise_weights(surface)
    entropy = _entropy(weights)
    ess = _effective_sample_size(weights)
    ratio = ess / surface.particle_count if surface.particle_count > 0 else 0.0
    degeneracy = ess < max(1.0, 0.01 * surface.particle_count)
    return InformationSummary(
        entropy_nats=entropy,
        effective_sample_size=ess,
        ess_ratio=ratio,
        degeneracy=degeneracy,
    )


def sensitivity_summary(
    surface: ParticleSurface,
    observation_axis: str,
    *,
    target_width: float | None = None,
) -> tuple[SensitivitySummary, ...]:
    """Ranking of per-mapping weight contributions to ``observation_axis``.

    The summary only inspects particles along a single observation axis.  When
    ``target_width`` is provided the function also computes the expected
    potential narrowing under a uniform weight reset for each mapping; this is
    a coarse proxy and the function labels it as such.
    """

    if observation_axis not in surface.axis_names:
        raise ValueError(f"unknown axis: {observation_axis}")
    axis_index = surface.axis_names.index(observation_axis)
    values = surface.particles[:, axis_index]
    weights = normalise_weights(surface)
    full_width = float(_weighted_quantile(values, weights, 0.95) - _weighted_quantile(values, weights, 0.05))
    target_width = float(target_width) if target_width is not None else max(0.0, full_width * 0.1)
    total_mass = float(weights.sum())
    if total_mass <= 0 or not math.isfinite(total_mass):
        return ()
    summaries: list[SensitivitySummary] = []
    for mapping_id, contribution in surface.mapping_breakdown.items():
        share = float(np.abs(contribution).sum() / total_mass) if total_mass > 0 else 0.0
        variation = float(contribution.std()) if contribution.size > 0 else 0.0
        # The variable hint is best-effort: the surface itself does not track
        # the variable name for each mapping contribution, so we leave it as
        # None when not derivable.  Downstream callers may overlay their own
        # mapping->variable lookup if needed.
        related_variable: str | None = None
        # Coarse narrowing estimate: zero the contribution and re-measure the
        # marginal width.  If the change is negative (broader), we report 0.
        if contribution.shape != surface.log_weights.shape:
            narrowing = 0.0
        else:
            reduced_log = surface.log_weights - contribution
            reduced_weights = np.exp(reduced_log - np.max(reduced_log))
            reduced_total = reduced_weights.sum()
            if reduced_total > 0 and math.isfinite(reduced_total):
                reduced_weights = reduced_weights / reduced_total
                reduced_width = float(
                    _weighted_quantile(values, reduced_weights, 0.95)
                    - _weighted_quantile(values, reduced_weights, 0.05)
                )
                narrowing = max(0.0, full_width - reduced_width)
            else:
                narrowing = 0.0
        if target_width > 0:
            ratio = narrowing / target_width
            narrowing = narrowing * min(1.0, ratio)
        summaries.append(
            SensitivitySummary(
                mapping_id=mapping_id,
                variable=related_variable,
                weight_contribution_share=share,
                relative_variation=variation,
                expected_potential_narrowing=narrowing,
            )
        )
    return tuple(sorted(summaries, key=lambda item: (-item.expected_potential_narrowing, item.mapping_id)))


def residual_summary(
    surface: ParticleSurface,
    axis: str,
    *,
    reference: float | None = None,
) -> ResidualSummary:
    """Weighted moments of the marginal residual for a single axis."""

    if axis not in surface.axis_names:
        raise ValueError(f"unknown axis: {axis}")
    axis_index = surface.axis_names.index(axis)
    unit = surface.axis_units[axis_index]
    values = surface.particles[:, axis_index]
    weights = normalise_weights(surface)
    centre = _weighted_mean(values, weights) if reference is None else float(reference)
    bias = _weighted_mean(values, weights) - centre
    variance = _weighted_moment(values, weights, 2, centre)
    third = _weighted_moment(values, weights, 3, centre)
    fourth = _weighted_moment(values, weights, 4, centre)
    std = math.sqrt(max(variance, 0.0))
    skewness = third / (std ** 3) if std > 0 else 0.0
    kurtosis = fourth / (std ** 4) if std > 0 else 0.0
    bandwidth = _gaussian_kde_bandwidth(values)
    return ResidualSummary(
        axis=axis,
        unit=unit,
        bias=bias,
        absolute_bias=abs(bias),
        variance=variance,
        skewness=skewness,
        kurtosis=kurtosis,
        bandwidth=bandwidth,
    )


def summarise_surface(
    surface: ParticleSurface,
    *,
    observation_axis: str | None = None,
    target_width: float | None = None,
    references: Mapping[str, float] | None = None,
) -> WaveDiagnostics:
    """Produce the full diagnostics summary required by the search controller."""

    references = dict(references or {})
    marginals = tuple(marginal_summary(surface, axis) for axis in surface.axis_names)
    peaks = tuple(peak_summary(surface, axis) for axis in surface.axis_names)
    residuals = tuple(
        residual_summary(surface, axis, reference=references.get(axis)) for axis in surface.axis_names
    )
    sensitivity_axis = observation_axis or surface.axis_names[0]
    sensitivities = sensitivity_summary(surface, sensitivity_axis, target_width=target_width)
    multimodal_axes = tuple(summary.axis for summary in marginals if summary.multimodal)
    entropy = information_summary(surface)
    notes = _diagnostic_notes(surface, marginals, sensitivities, entropy)
    return WaveDiagnostics(
        surface_id=surface.surface_id,
        question_id=surface.question_id,
        surface_kind=surface.kind.value,
        calibration_basis=surface.calibration_basis.value,
        particle_count=surface.particle_count,
        entropy=entropy,
        marginals=marginals,
        peaks=peaks,
        sensitivities=sensitivities,
        residuals=residuals,
        multimodal_axes=multimodal_axes,
        notes=notes,
    )


def _diagnostic_notes(
    surface: ParticleSurface,
    marginals: Sequence[MarginalSummary],
    sensitivities: Sequence[SensitivitySummary],
    surface_entropy: InformationSummary,
) -> tuple[str, ...]:
    notes: list[str] = []
    if surface.kind is SurfaceKind.POSSIBILITY:
        notes.append(
            "surface is a possibility_surface; weight summaries are not calibrated probabilities"
        )
    if surface.calibration_basis is CalibrationBasis.UNMEASURED:
        notes.append("calibration_basis is unmeasured; no probability claim is made")
    if surface_entropy.effective_sample_size < max(1.0, 0.01 * surface.particle_count):
        notes.append("effective sample size collapsed; consider adding likelihood evidence")
    for summary in marginals:
        if summary.multimodal:
            notes.append(
                f"axis {summary.axis} is multimodal with {summary.mode_count} peaks; "
                "consider adding an interaction or latent variable"
            )
    for item in sensitivities:
        if item.expected_potential_narrowing <= 0:
            notes.append(
                f"mapping {item.mapping_id} does not narrow {item.mapping_id}'s contribution; "
                "consider revising or ablating"
            )
    return tuple(notes)


def diagnostics_as_mapping(diagnostics: WaveDiagnostics) -> dict[str, Any]:
    """Render the diagnostics summary into a JSON-compatible mapping."""

    return {
        "surface_id": diagnostics.surface_id,
        "question_id": diagnostics.question_id,
        "surface_kind": diagnostics.surface_kind,
        "calibration_basis": diagnostics.calibration_basis,
        "particle_count": diagnostics.particle_count,
        "entropy": {
            "entropy_nats": diagnostics.entropy.entropy_nats,
            "effective_sample_size": diagnostics.entropy.effective_sample_size,
            "ess_ratio": diagnostics.entropy.ess_ratio,
            "degeneracy": diagnostics.entropy.degeneracy,
        },
        "marginals": [
            {
                "axis": summary.axis,
                "unit": summary.unit,
                "weighted_mean": summary.weighted_mean,
                "weighted_std": summary.weighted_std,
                "p05": summary.p05,
                "p50": summary.p50,
                "p95": summary.p95,
                "support_min": summary.support_min,
                "support_max": summary.support_max,
                "modes": list(summary.modes),
                "mode_count": summary.mode_count,
                "multimodal": summary.multimodal,
            }
            for summary in diagnostics.marginals
        ],
        "peaks": [
            {
                "axis": peak.axis,
                "unit": peak.unit,
                "peak_value": peak.peak_value,
                "peak_weight": peak.peak_weight,
                "alternatives": list(peak.alternatives),
            }
            for peak in diagnostics.peaks
        ],
        "sensitivities": [
            {
                "mapping_id": item.mapping_id,
                "variable": item.variable,
                "weight_contribution_share": item.weight_contribution_share,
                "relative_variation": item.relative_variation,
                "expected_potential_narrowing": item.expected_potential_narrowing,
            }
            for item in diagnostics.sensitivities
        ],
        "residuals": [
            {
                "axis": residual.axis,
                "unit": residual.unit,
                "bias": residual.bias,
                "absolute_bias": residual.absolute_bias,
                "variance": residual.variance,
                "skewness": residual.skewness,
                "kurtosis": residual.kurtosis,
                "bandwidth": residual.bandwidth,
            }
            for residual in diagnostics.residuals
        ],
        "multimodal_axes": list(diagnostics.multimodal_axes),
        "notes": list(diagnostics.notes),
    }


__all__ = [
    "InformationSummary",
    "MarginalSummary",
    "PeakSummary",
    "ResidualSummary",
    "SensitivitySummary",
    "WaveDiagnostics",
    "diagnostics_as_mapping",
    "information_summary",
    "marginal_summary",
    "peak_summary",
    "residual_summary",
    "sensitivity_summary",
    "summarise_surface",
]