"""Decision-value evaluation and typed-action mapping for the wave loop.

This module maps particle-surface diagnostics to typed refinement actions
(``measure``, ``add_interaction``, ``split_regime``, ``minimize``,
``expand_variable``, ``stop``) and evaluates whether the current surface
satisfies the declared decision-value policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .wave_loop_contract import (
    LoopAction,
    ParticleSurface,
    WaveLoopConfig,
    WaveLoopError,
)


# ---------------------------------------------------------------------------
# Axis usefulness
# ---------------------------------------------------------------------------


def _axis_useful(
    axis: dict[str, Any],
    summary: Mapping[str, float],
    policy: Mapping[str, Any],
) -> tuple[bool, str]:
    width = float(summary.get("width", 0.0))
    absolute = axis.get("absolute_tolerance")
    if absolute is not None and width <= float(absolute):
        return True, "absolute_tolerance_met"
    reference = axis.get("reference_value")
    if reference is not None and isinstance(reference, (int, float)) and reference != 0:
        relative = policy["relative_tolerance"]
        relative_width = width / abs(float(reference))
        if relative_width <= relative:
            return True, "relative_tolerance_met"
    if absolute is None and reference is None:
        return False, "no_tolerance_declared"
    return False, "tolerance_not_met"


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def _sensitivity(
    variables: Mapping[str, dict[str, Any]],
    surface: ParticleSurface,
) -> list[dict[str, Any]]:
    """Estimate which variable's resolution most reduces surface width."""

    baseline_widths: dict[str, float] = {
        axis: float(surface.marginals[axis]["width"]) for axis in surface.marginals
    }
    rows: list[dict[str, Any]] = []
    for name, spec in variables.items():
        if spec["status"] == "missing":
            continue
        if not math.isfinite(spec["lower"]) or not math.isfinite(spec["upper"]):
            continue
        midpoint = (spec["lower"] + spec["upper"]) / 2.0
        new_widths: list[float] = []
        for axis_id in surface.marginals:
            paired_values = list(surface.values[surface.axes.index(axis_id)])
            new_values = []
            for index in range(len(paired_values)):
                shrunk = paired_values[index] * 0.0 + midpoint * (1.0 if index == 0 else 0.0)
                new_values.append(shrunk)
            values_only = [value for value in new_values if math.isfinite(value)]
            width_if_resolved = max(values_only) - min(values_only) if values_only else 0.0
            new_widths.append(width_if_resolved)
        if not new_widths:
            continue
        max_remaining = max(new_widths)
        baseline = max(baseline_widths.values()) if baseline_widths else 0.0
        potential = max(0.0, baseline - max_remaining)
        rows.append(
            {
                "variable": name,
                "potential_narrowing": potential,
                "narrowing_fraction": potential / baseline if baseline > 0 else 0.0,
                "width_if_resolved_to_midpoint": max_remaining,
            }
        )
    rows.sort(key=lambda row: (-row["potential_narrowing"], row["variable"]))
    return rows


# ---------------------------------------------------------------------------
# Individual action generators
# ---------------------------------------------------------------------------


def _measure_action(
    axes: tuple[dict[str, Any], ...],
    surface: ParticleSurface,
    sensitivities: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> LoopAction | None:
    widest_axis = None
    widest_value = 0.0
    for axis in axes:
        axis_id = str(axis["axis_id"])
        width = float(surface.marginals.get(axis_id, {}).get("width", 0.0))
        useful, _ = _axis_useful(axis, surface.marginals.get(axis_id, {}), policy)
        if useful:
            continue
        if width > widest_value:
            widest_value = width
            widest_axis = axis
    if widest_axis is None:
        return None
    if not sensitivities:
        return None
    target = sensitivities[0]
    if target["potential_narrowing"] <= policy["min_action_benefit"]:
        return None
    return LoopAction(
        action_id=f"measure-{target['variable']}",
        action_kind="measure",
        rationale=(
            f"axis {widest_axis['axis_id']} width {widest_value:.4g} exceeds tolerance"
        ),
        affected_entities=(str(target["variable"]), str(widest_axis["axis_id"])),
        expected_decision_loss_reduction=float(target["potential_narrowing"]),
        estimated_cost=1.0,
        evidence=(str(widest_axis.get("decision_useful") or "absolute_tolerance_exceeded"),),
    )


def _interaction_action(
    surface: ParticleSurface,
    variables: Mapping[str, dict[str, Any]],
    policy: Mapping[str, Any],
) -> LoopAction | None:
    threshold = float(policy["residual_interaction_threshold"])
    if float(surface.diagnostics.get("residual_interaction", 0.0)) <= threshold:
        return None
    pair = surface.diagnostics.get("residual_interaction_pair") or []
    if len(pair) != 2:
        return None
    candidate_vars = [
        name for name, spec in variables.items() if spec["status"] != "missing"
    ]
    if len(candidate_vars) < 2:
        return None
    chosen = tuple(sorted(candidate_vars[:2]))
    return LoopAction(
        action_id=f"add-interaction-{chosen[0]}-{chosen[1]}",
        action_kind="add_interaction",
        rationale=(
            f"residual correlation {float(surface.diagnostics['residual_interaction']):.3f} "
            f"between axes {pair[0]} and {pair[1]}"
        ),
        affected_entities=(str(pair[0]), str(pair[1]), chosen[0], chosen[1]),
        expected_decision_loss_reduction=float(surface.diagnostics["residual_interaction"]),
        estimated_cost=2.0,
        evidence=("residual_interaction_detected",),
    )


def _split_action(
    surface: ParticleSurface,
    variables: Mapping[str, dict[str, Any]],
) -> LoopAction | None:
    del variables  # not used for mode detection — axis-internal
    for mode in surface.diagnostics.get("modes", ()):
        if mode.get("bimodal"):
            axis_id = str(mode["axis_id"])
            return LoopAction(
                action_id=f"split-regime-{axis_id}",
                action_kind="split_regime",
                rationale=(
                    f"axis {axis_id} shows bimodality with score {float(mode['score']):.3f}"
                ),
                affected_entities=(axis_id,),
                expected_decision_loss_reduction=float(mode["score"]),
                estimated_cost=3.0,
                evidence=("bimodal_distribution",),
            )
    return None


def _minimize_action(
    surface: ParticleSurface,
    ir_count: int,
    last_mapping_id: str,
) -> LoopAction | None:
    if ir_count <= 1:
        return None
    ess = float(surface.diagnostics.get("effective_sample_size", 0.0))
    if ess >= 0.95:
        return None
    return LoopAction(
        action_id=f"minimize-{last_mapping_id}",
        action_kind="minimize",
        rationale=f"mapping {last_mapping_id} contributes marginally to effective sample size {ess:.3f}",
        affected_entities=(last_mapping_id,),
        expected_decision_loss_reduction=1.0 - ess,
        estimated_cost=0.5,
        evidence=("low_contribution_mapping",),
    )


def _expand_variable_action(
    variables: Mapping[str, dict[str, Any]],
    axes: tuple[dict[str, Any], ...],
    surface: ParticleSurface,
    policy: Mapping[str, Any],
) -> LoopAction | None:
    del surface  # missing detection is based on variable status, not current surface
    for name, spec in variables.items():
        if spec["status"] != "missing":
            continue
        if not axes:
            return None
        axis_id = str(axes[0]["axis_id"])
        return LoopAction(
            action_id=f"expand-variable-{name}",
            action_kind="expand_variable",
            rationale=f"variable {name} is missing and blocks axis {axis_id} usefulness",
            affected_entities=(name, axis_id),
            expected_decision_loss_reduction=float(policy["min_action_benefit"]) + 1.0,
            estimated_cost=1.5,
            evidence=("missing_variable",),
        )
    return None


# ---------------------------------------------------------------------------
# Decision actions orchestrator
# ---------------------------------------------------------------------------


def decision_actions(
    axes: tuple[dict[str, Any], ...],
    variables: Mapping[str, dict[str, Any]],
    surface: ParticleSurface,
    ir_count: int,
    last_mapping_id: str,
    policy: Mapping[str, Any],
    config: WaveLoopConfig,
) -> list[LoopAction]:
    """Map diagnostics to typed refinement actions."""

    actions: list[LoopAction] = []
    sensitivities = _sensitivity(variables, surface)
    measure_action = _measure_action(axes, surface, sensitivities, policy)
    if measure_action is not None:
        actions.append(measure_action)
    interaction_action = _interaction_action(surface, variables, policy)
    if interaction_action is not None:
        actions.append(interaction_action)
    split_action = _split_action(surface, variables)
    if split_action is not None:
        actions.append(split_action)
    minimize_action = _minimize_action(surface, ir_count, last_mapping_id)
    if minimize_action is not None:
        actions.append(minimize_action)
    expand_action = _expand_variable_action(variables, axes, surface, policy)
    if expand_action is not None:
        actions.append(expand_action)
    actions.sort(key=lambda a: (-a.expected_decision_loss_reduction, a.action_id))
    return actions[: config.max_actions]


# ---------------------------------------------------------------------------
# Decision value evaluation
# ---------------------------------------------------------------------------


def evaluate_decision_value(
    axes: tuple[dict[str, Any], ...],
    surface: ParticleSurface,
    policy: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    all_useful = True
    for axis in axes:
        axis_id = str(axis["axis_id"])
        summary = surface.marginals.get(axis_id, {"width": 0.0})
        useful, reason = _axis_useful(axis, summary, policy)
        evidence.append({"axis_id": axis_id, "useful": useful, "reason": reason})
        all_useful = all_useful and useful
    ess = float(surface.diagnostics.get("effective_sample_size", 0.0))
    ess_ok = ess >= float(policy["min_effective_sample_size"])
    evidence.append(
        {
            "criterion": "effective_sample_size",
            "value": ess,
            "threshold": policy["min_effective_sample_size"],
            "passed": ess_ok,
        }
    )
    return all_useful and ess_ok, evidence


__all__ = [
    "_axis_useful",
    "decision_actions",
    "evaluate_decision_value",
]
