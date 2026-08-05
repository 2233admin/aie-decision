"""Diagnostic-driven wave-loop with ledger and replay.

This module is the **public facade** for the wave-loop subsystem.  It
re-exports every public symbol from the helper modules below so that
existing callers (``wave_authority.py``, tests, the runner script) are
unaffected by the internal refactoring:

* :mod:`wave_loop_contract`  — shared dataclasses, constants, errors
* :mod:`wave_loop_validation` — JSON schema validation
* :mod:`wave_loop_actions`   — decision-value + typed action mapping
* :mod:`wave_loop_replay`    — ledger replay + checkpoint verification

The only new implementation in this module is :func:`run_wave_loop` (the
deterministic loop orchestrator) and the particle-surface evaluation
helpers that it calls directly.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .ledger import AnalysisLedger
from .models import SCHEMA_VERSION, Revision
from .wave_loop_actions import (
    decision_actions,
    evaluate_decision_value,
)
from .wave_loop_contract import (
    CompiledFactorIR,
    compile_factor_ir,
    FACTOR_IR_VERSION,
    PARTICLE_SURFACE_VERSION,
)
from .wave_loop_replay import (
    _canonical,
    _digest,
    create_wave_checkpoint,
    replay_wave_ledger,
    verify_wave_checkpoint,
)
from .wave_loop_validation import (
    validate_joint_schema,
)

# Re-export every public symbol so callers see the same __all__.
from .wave_loop_contract import (  # noqa: F401 — explicit re-export
    ACTION_KINDS,
    JOINT_SCHEMA_VERSION,
    LoopAction,
    ParticleSurface,
    WAVE_ACTIVATION_STATES,
    WAVE_CHECKPOINT_VERSION,
    WAVE_LEDGER_VERSION,
    WAVE_LOOP_RESULT_VERSION,
    WAVE_TERMINAL_STATES,
    WaveEvent,
    WaveLoopConfig,
    WaveLoopError,
)


# ---------------------------------------------------------------------------
# Internal particle-surface evaluation (only called by run_wave_loop)
# ---------------------------------------------------------------------------


def _deterministic_particles(
    variables: Mapping[str, dict[str, Any]],
    axes: tuple[dict[str, Any], ...],
    rng: Random,
    particle_count: int,
) -> list[dict[str, float]]:
    """Build a deterministic stratified particle plan from the variable bounds."""
    del axes
    var_list = [variables[name] for name in sorted(variables.keys())]
    jitter_rng = Random(rng.random())  # noqa: S311 — seeded from caller
    particles: list[dict[str, float]] = []
    if particle_count <= 0:
        particle_count = 1
    for slot in range(particle_count):
        base = (slot + 0.5) / particle_count
        jitter = jitter_rng.uniform(-0.5 / particle_count, 0.5 / particle_count)
        fraction = max(0.0, min(1.0, base + jitter))
        particle: dict[str, float] = {}
        for var in var_list:
            lower = var["lower"]
            upper = var["upper"]
            if not math.isfinite(lower) or not math.isfinite(upper):
                particle[var["name"]] = 0.0
            else:
                particle[var["name"]] = lower + fraction * (upper - lower)
        particles.append(particle)
    return particles


def _marginal_summary(
    values: Sequence[float], weights: Sequence[float]
) -> dict[str, float]:
    paired = sorted(zip(values, weights), key=lambda item: item[0])
    sorted_values = [item[0] for item in paired]
    sorted_weights = [item[1] for item in paired]
    if not sorted_values:
        return {"mean": 0.0, "width": 0.0, "p05": 0.0, "p95": 0.0}
    weighted_mean = math.fsum(value * weight for value, weight in paired)
    weighted_width = sorted_values[-1] - sorted_values[0]
    p05 = _weighted_quantile(sorted_values, sorted_weights, 0.05)
    p95 = _weighted_quantile(sorted_values, sorted_weights, 0.95)
    return {"mean": weighted_mean, "width": weighted_width, "p05": p05, "p95": p95}


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    if not values:
        return 0.0
    if quantile <= 0:
        return values[0]
    if quantile >= 1:
        return values[-1]
    cumulative = 0.0
    total = math.fsum(weights)
    if total == 0:
        return values[0]
    threshold = quantile * total
    for value, weight in zip(values, weights):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return values[-1]


def _bimodality(
    values: Sequence[float], weights: Sequence[float]
) -> tuple[bool, float]:
    """Histogram-based bimodality detection with valley-mass scoring."""
    if len(values) < 4:
        return False, 0.0
    paired = sorted(zip(values, weights), key=lambda item: item[0])
    sorted_values = [item[0] for item in paired]
    total = math.fsum(weights)
    if total <= 0:
        return False, 0.0
    span = sorted_values[-1] - sorted_values[0]
    if span <= 0:
        return False, 0.0
    bin_count = min(8, max(3, len(sorted_values) // 4))
    bin_width = span / bin_count if bin_count > 0 else 0.0
    if bin_width <= 0:
        return False, 0.0
    bin_mass = [0.0] * bin_count
    for value, weight in paired:
        index = int((value - sorted_values[0]) / bin_width)
        if index >= bin_count:
            index = bin_count - 1
        bin_mass[index] += weight
    max_mass = max(bin_mass)
    if max_mass <= 0:
        return False, 0.0
    middle_start = bin_count // 3
    middle_end = bin_count - middle_start
    if middle_end - middle_start < 1:
        return False, 0.0
    left_max = max(bin_mass[:middle_start]) if middle_start > 0 else 0.0
    right_max = max(bin_mass[middle_end:]) if middle_end < bin_count else 0.0
    valley = min(bin_mass[middle_start:middle_end])
    left_density = left_max / max_mass
    right_density = right_max / max_mass
    valley_density = valley / max_mass
    if left_density <= 0.05 or right_density <= 0.05:
        return False, 0.0
    if valley_density * 3 > min(left_density, right_density):
        return False, 0.0
    score = min(left_density, right_density) - valley_density
    return score > 0.15, score


def _residual_interaction(
    values_by_axis: Mapping[str, Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, tuple[str, str] | None]:
    axis_names = list(values_by_axis.keys())
    if len(axis_names) < 2:
        return 0.0, None
    total = math.fsum(weights)
    if total <= 0:
        return 0.0, None
    means = {
        axis: math.fsum(
            value * weight for value, weight in zip(values_by_axis[axis], weights)
        )
        / total
        for axis in axis_names
    }
    best_score = 0.0
    best_pair: tuple[str, str] | None = None
    for left_index, left_axis in enumerate(axis_names):
        for right_axis in axis_names[left_index + 1 :]:
            covariance = (
                math.fsum(
                    weight
                    * (values_by_axis[left_axis][index] - means[left_axis])
                    * (values_by_axis[right_axis][index] - means[right_axis])
                    for index, weight in enumerate(weights)
                )
                / total
            )
            left_variance = (
                math.fsum(
                    weight * (value - means[left_axis]) ** 2
                    for value, weight in zip(values_by_axis[left_axis], weights)
                )
                / total
            )
            right_variance = (
                math.fsum(
                    weight * (value - means[right_axis]) ** 2
                    for value, weight in zip(values_by_axis[right_axis], weights)
                )
                / total
            )
            if left_variance <= 0 or right_variance <= 0:
                continue
            correlation = covariance / math.sqrt(left_variance * right_variance)
            score = abs(correlation)
            if score > best_score:
                best_score = score
                best_pair = (left_axis, right_axis)
    return best_score, best_pair


def evaluate_particle_surface(
    axes: tuple[dict[str, Any], ...],
    variables: Mapping[str, dict[str, Any]],
    irs: tuple[CompiledFactorIR, ...],
    rng: Random,
    particle_count: int,
    surface_counter: int,
) -> ParticleSurface:
    """Deterministically evaluate the factor IRs and compute surface diagnostics."""

    particles = _deterministic_particles(variables, axes, rng, particle_count)
    if not particles:
        particles = [{name: 0.0 for name in variables} for _ in range(2)]
    raw_weights: list[float] = []
    for particle in particles:
        value = 0.0
        for ir in irs:
            try:
                value += ir.log_potential(particle)
            except WaveLoopError:
                continue
        raw_weights.append(value - _min_particle_value(particles, irs))
    weights = _normalise_weights(raw_weights)
    values_by_axis: dict[str, list[float]] = {}
    for axis in axes:
        axis_id = str(axis["axis_id"])
        per_particle = []
        for particle in particles:
            if not irs:
                per_particle.append(0.0)
                continue
            value = 0.0
            for ir in irs:
                try:
                    value += ir.log_potential(particle)
                except WaveLoopError:
                    continue
            per_particle.append(value)
        values_by_axis[axis_id] = per_particle
    marginals: dict[str, dict[str, float]] = {}
    for axis_id, values in values_by_axis.items():
        marginals[axis_id] = _marginal_summary(values, weights)
    diagnostics: dict[str, Any] = {
        "particle_count": len(particles),
        "effective_sample_size": _effective_sample_size(weights),
        "entropy": _entropy(weights),
        "modes": [],
    }
    for axis_id, values in values_by_axis.items():
        detected, score = _bimodality(values, weights)
        diagnostics["modes"].append(
            {"axis_id": axis_id, "bimodal": detected, "score": score}
        )
    residual, residual_pair = _residual_interaction(values_by_axis, weights)
    diagnostics["residual_interaction"] = residual
    diagnostics["residual_interaction_pair"] = (
        list(residual_pair) if residual_pair else None
    )
    semantics = "possibility_surface"
    if all(marginal["width"] <= 1e-9 for marginal in marginals.values()):
        semantics = "invalid_surface"
    surface_axis = axes[0]["axis_id"] if axes else "surface"
    surface_id = f"{surface_axis}-s{surface_counter:04d}"
    return ParticleSurface(
        surface_id=surface_id,
        surface_version=PARTICLE_SURFACE_VERSION,
        semantics=semantics,
        axes=tuple(str(axis["axis_id"]) for axis in axes),
        values=tuple(tuple(values_by_axis[str(axis["axis_id"])]) for axis in axes),
        log_weights=tuple(raw_weights),
        marginals=marginals,
        diagnostics=diagnostics,
    )


def _min_particle_value(
    particles: Sequence[Mapping[str, float]],
    irs: tuple[CompiledFactorIR, ...],
) -> float:
    minimum = 0.0
    seen = False
    for particle in particles:
        value = 0.0
        for ir in irs:
            try:
                value += ir.log_potential(particle)
            except WaveLoopError:
                continue
        if not seen or value < minimum:
            minimum = value
            seen = True
    return minimum if seen else 0.0


def _normalise_weights(weights: Sequence[float]) -> tuple[float, ...]:
    total = math.fsum(weight for weight in weights if weight > 0)
    if total <= 0:
        if any(weight != 0 for weight in weights):
            equal = 1.0 / len(weights)
            return tuple(equal for _ in weights)
        return tuple(0.0 for _ in weights)
    return tuple(weight / total if weight > 0 else 0.0 for weight in weights)


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = math.fsum(weight ** 2 for weight in weights)
    if total <= 0 or not weights:
        return 0.0
    return 1.0 / (len(weights) * total)


def _entropy(weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        if weight <= 0:
            continue
        normalised = weight / total
        entropy -= normalised * math.log(normalised)
    return entropy


# ---------------------------------------------------------------------------
# Public loop driver
# ---------------------------------------------------------------------------


def _seed_default_ir(axes: tuple[dict[str, Any], ...]) -> tuple[CompiledFactorIR, ...]:
    if not axes:
        return ()
    formula = "0"
    tree = ast.parse(formula, mode="eval")
    return (
        CompiledFactorIR(
            ir_version=FACTOR_IR_VERSION,
            mapping_id="default-zero",
            formula=formula,
            tree=tree,
            referenced_variables=(),
        ),
    )


def _surface_to_dict(surface: ParticleSurface) -> dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "surface_version": surface.surface_version,
        "semantics": surface.semantics,
        "axes": list(surface.axes),
        "marginals": {axis: dict(summary) for axis, summary in surface.marginals.items()},
        "diagnostics": {
            "particle_count": surface.diagnostics["particle_count"],
            "effective_sample_size": surface.diagnostics["effective_sample_size"],
            "entropy": surface.diagnostics["entropy"],
            "modes": list(surface.diagnostics["modes"]),
            "residual_interaction": surface.diagnostics["residual_interaction"],
            "residual_interaction_pair": list(
                surface.diagnostics["residual_interaction_pair"] or ()
            ),
        },
        "log_weights": [float(weight) for weight in surface.log_weights],
    }


def _action_to_dict(action: LoopAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "action_kind": action.action_kind,
        "rationale": action.rationale,
        "affected_entities": list(action.affected_entities),
        "expected_decision_loss_reduction": action.expected_decision_loss_reduction,
        "estimated_cost": action.estimated_cost,
        "evidence": list(action.evidence),
    }


def run_wave_loop(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drive one deterministic round of the wave-loop and return a replayable record."""

    parsed = validate_joint_schema(payload)
    run_id = parsed["run_id"]
    axes = parsed["axes"]
    variables = parsed["variables"]
    mappings = parsed["mappings"]
    policy = parsed["decision_policy"]
    config = parsed["budget"]
    irs = tuple(compile_factor_ir(mapping) for mapping in mappings) or _seed_default_ir(axes)
    rng = Random(config.seed)
    ledger = AnalysisLedger(run_id)
    events: list[WaveEvent] = []

    def record(
        surface_id: str,
        state: str,
        round_index: int,
        reason: str,
        data: Mapping[str, Any],
    ) -> WaveEvent:
        sequence = len(events) + 1
        event_id = f"{run_id}-wave-{sequence:06d}"
        event = WaveEvent(
            schema_version=SCHEMA_VERSION,
            event_id=event_id,
            run_id=run_id,
            surface_id=surface_id,
            state=state,
            round_index=round_index,
            reason=reason,
            data=dict(data),
            revision=Revision(
                revision_id=f"{event_id}-r1",
                sequence=1,
                created_at=config.started_at,
            ),
        )
        events.append(event)
        ledger.append("wave_event", event)
        return event

    round_index = 0
    accepted = False
    stop_reason = "budget-exhausted"
    surface = evaluate_particle_surface(
        tuple(axes), variables, irs, rng, config.particle_count, surface_counter=1
    )
    record(
        surface.surface_id, "DRAFT", round_index, "mapping_compiled",
        {"axes": list(surface.axes)},
    )
    record(
        surface.surface_id, "EVALUATED", round_index, "particle_surface_computed",
        {
            "particle_count": surface.diagnostics["particle_count"],
            "effective_sample_size": surface.diagnostics["effective_sample_size"],
        },
    )
    useful, evidence = evaluate_decision_value(tuple(axes), surface, policy)
    actions = decision_actions(
        tuple(axes), variables, surface,
        len(irs), irs[-1].mapping_id if irs else "none",
        policy, config,
    )
    if useful:
        accepted = True
        stop_reason = "decision-value-met"
        record(
            surface.surface_id, "ACCEPTED", round_index, stop_reason,
            {
                "criteria_evidence": evidence,
                "actions_emitted": [a.action_kind for a in actions],
            },
        )
        stop_action = LoopAction(
            action_id="stop-decision-value-met",
            action_kind="stop",
            rationale="decision-value policy satisfied for every axis",
            affected_entities=tuple(surface.axes),
            expected_decision_loss_reduction=0.0,
            estimated_cost=0.0,
            evidence=tuple(
                str(item["reason"])
                for item in evidence
                if isinstance(item, Mapping) and item.get("reason")
            ),
        )
        if stop_action not in actions:
            actions.append(stop_action)
    else:
        record(
            surface.surface_id, "REFINING", round_index, "diagnostic_actions_emitted",
            {
                "actions_emitted": [a.action_kind for a in actions],
                "criteria_evidence": evidence,
            },
        )
        unresolved_axes = [
            str(item["axis_id"])
            for item in evidence
            if isinstance(item, Mapping) and "axis_id" in item and not item.get("useful")
        ]
        if unresolved_axes:
            stop_reason = "unresolved-decision-criteria"
        else:
            stop_reason = "marginal-value-insufficient"
        record(
            surface.surface_id, "UNRESOLVED", round_index, stop_reason,
            {"unresolved_axes": unresolved_axes, "criteria_evidence": evidence},
        )
        actions.append(
            LoopAction(
                action_id="stop-budget-exhausted" if not actions else "stop-defer-to-action",
                action_kind="stop",
                rationale="loop terminated; further iterations require additional evidence or mappings",
                affected_entities=tuple(unresolved_axes),
                expected_decision_loss_reduction=0.0,
                estimated_cost=0.0,
                evidence=(stop_reason,),
            )
        )

    ledger_export = ledger.export()
    checkpoint = create_wave_checkpoint(ledger_export)
    return {
        "schema_version": WAVE_LOOP_RESULT_VERSION,
        "run_id": run_id,
        "surface": _surface_to_dict(surface),
        "decision_value": {"accepted": accepted, "stop_reason": stop_reason, "criteria_evidence": evidence},
        "actions": [_action_to_dict(action) for action in actions],
        "budget": {
            "max_rounds": config.max_rounds,
            "max_actions": config.max_actions,
            "particle_count": config.particle_count,
            "seed": config.seed,
            "rounds_used": round_index + 1,
            "actions_emitted": len(actions),
        },
        "ledger": ledger_export,
        "checkpoint": checkpoint,
        "ranking_method": {
            "name": "diagnostic_action_policy_v1",
            "formula": "actions ranked by expected_decision_loss_reduction - estimated_cost",
            "calibrated": False,
            "warning": "decision-value is not a calibrated posterior probability",
        },
    }


# ---------------------------------------------------------------------------
# Public API — identical to pre-refactor __all__
# ---------------------------------------------------------------------------

__all__ = [
    "ACTION_KINDS",
    "JOINT_SCHEMA_VERSION",
    "FACTOR_IR_VERSION",
    "PARTICLE_SURFACE_VERSION",
    "WAVE_LEDGER_VERSION",
    "WAVE_CHECKPOINT_VERSION",
    "WAVE_LOOP_RESULT_VERSION",
    "CompiledFactorIR",
    "LoopAction",
    "ParticleSurface",
    "WaveEvent",
    "WaveLoopConfig",
    "WaveLoopError",
    "compile_factor_ir",
    "create_wave_checkpoint",
    "evaluate_particle_surface",
    "replay_wave_ledger",
    "run_wave_loop",
    "validate_joint_schema",
    "verify_wave_checkpoint",
]
