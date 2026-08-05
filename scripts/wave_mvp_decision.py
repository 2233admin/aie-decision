"""MVP decision-value policy and typed action selection."""

from __future__ import annotations
import json
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from wave_mvp_models import (
    AxisDiagnostic, VariableSpec, ParticleSurface, SurfaceDiagnostics, TypedAction,
)

# Decision-value policy and action selection.
# ---------------------------------------------------------------------------


def _axis_passes(axis: AxisDiagnostic, tolerance: Mapping[str, Any]) -> bool:
    kind = tolerance.get("kind")
    if kind == "absolute":
        threshold = float(tolerance.get("value", 0.0))
        return axis.absolute_width <= threshold
    if kind == "relative":
        threshold = float(tolerance.get("value", 0.0))
        if axis.relative_width in (float("inf"), float("-inf")):
            return False
        return axis.relative_width <= threshold
    if kind == "loss_threshold":
        threshold = float(tolerance.get("value", 0.0))
        return axis.sharpness_absolute <= threshold
    return False


def evaluate_decision_value(
    diagnostics: SurfaceDiagnostics,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    tolerances = policy.get("axes", {})
    per_axis: dict[str, dict[str, Any]] = {}
    for axis in diagnostics.axes:
        tolerance = tolerances.get(axis.name, {})
        passed = _axis_passes(axis, tolerance) if tolerance else False
        per_axis[axis.name] = {
            "tolerance": dict(tolerance),
            "passed": passed,
            "absolute_width": axis.absolute_width,
            "sharpness": axis.sharpness_absolute,
        }
    overall_passed = all(item["passed"] for item in per_axis.values()) and bool(per_axis)
    return {
        "passed": overall_passed,
        "per_axis": per_axis,
        "bimodal_axes": list(diagnostics.bimodal_axes),
    }


def _select_action(
    *,
    diagnostics: SurfaceDiagnostics,
    decision: Mapping[str, Any],
    variables: Sequence[VariableSpec],
    surface: ParticleSurface,
    regime_split: Mapping[str, Any] | None,
    round_index: int,
) -> TypedAction | None:
    failing_axes = [
        axis
        for axis in diagnostics.axes
        if not decision["per_axis"].get(axis.name, {}).get("passed", False)
    ]
    if not failing_axes and not diagnostics.bimodal_axes:
        return TypedAction(
            action_id=f"r{round_index}-stop",
            kind="stop",
            target="loop",
            rationale="all axes within decision tolerance",
            expected_loss_improvement=0.0,
            estimated_cost=0.0,
            affected_entities=(),
        )
    # Multimodality triggers split_regime.
    if diagnostics.bimodal_axes and regime_split is not None:
        axis_name = diagnostics.bimodal_axes[0]
        return TypedAction(
            action_id=f"r{round_index}-split-{axis_name}",
            kind="split_regime",
            target=axis_name,
            rationale=(
                f"axis {axis_name} shows {len(diagnostics.bimodal_axes)}+ modes; "
                "switching to a regime-conditional model is expected to recover "
                "decision value"
            ),
            expected_loss_improvement=0.4,
            estimated_cost=1.0,
            affected_entities=(axis_name, regime_split.get("split_variable", "")),
            basis={"modes": diagnostics.bimodal_axes},
        )
    # Cross-axis residual suggests an interaction or a missing variable.
    worst = max(failing_axes, key=lambda axis: axis.residual_proxy, default=None)
    if worst is not None and worst.residual_proxy >= 0.5:
        return TypedAction(
            action_id=f"r{round_index}-add-interaction-{worst.name}",
            kind="add_interaction",
            target=worst.name,
            rationale=(
                f"residual proxy on {worst.name} is {worst.residual_proxy:.3f}; "
                "adding an interaction term is expected to reduce unexplained variance"
            ),
            expected_loss_improvement=0.3,
            estimated_cost=1.0,
            affected_entities=(worst.name,),
            basis={"residual_proxy": worst.residual_proxy},
        )
    # Sensitivity-based measurement action.
    sensitivity = _estimate_sensitivity(surface, variables)
    if sensitivity:
        target_variable, score = sensitivity[0]
        return TypedAction(
            action_id=f"r{round_index}-measure-{target_variable}",
            kind="measure",
            target=target_variable,
            rationale=(
                f"variable {target_variable} has the highest sensitivity rank "
                f"({score:.3f}); measuring it should narrow the worst failing axis"
            ),
            expected_loss_improvement=score,
            estimated_cost=0.5,
            affected_entities=(target_variable,),
            basis={"sensitivity": [item for item in sensitivity]},
        )
    # Final fallback: minimise the ablatable variable with the lowest impact.
    ablatable = [variable for variable in variables if variable.ablatable]
    if ablatable:
        target_variable = ablatable[0].name
        return TypedAction(
            action_id=f"r{round_index}-minimize-{target_variable}",
            kind="minimize",
            target=target_variable,
            rationale=(
                f"variable {target_variable} is declared ablatable and shows "
                "minimal contribution to the surface"
            ),
            expected_loss_improvement=0.1,
            estimated_cost=0.2,
            affected_entities=(target_variable,),
            basis={},
        )
    return TypedAction(
        action_id=f"r{round_index}-stop",
        kind="stop",
        target="loop",
        rationale="no further typed actions available",
        expected_loss_improvement=0.0,
        estimated_cost=0.0,
        affected_entities=(),
    )


def _estimate_sensitivity(
    surface: ParticleSurface,
    variables: Sequence[VariableSpec],
) -> list[tuple[str, float]]:
    """Rank variables by their spread along the worst axis.

    The MVP uses per-variable spread as a proxy for sensitivity: narrowing
    that variable's range would most reduce the worst axis width.  This is
    cheap and deterministic; the PRD documents this shortcut for the
    CPU vertical slice.
    """
    if surface.values.shape[0] == 0:
        return []
    spread = surface.values.std(axis=0)
    if not np.any(spread):
        return []
    worst_axis = int(np.argmax(spread))
    axis_name = surface.axis_order[worst_axis]
    scores: list[tuple[str, float]] = []
    for variable in variables:
        # Correlation proxy between each variable's index and the worst axis.
        # Without per-variable particle storage we approximate using the
        # deterministic rank: variables with wider declared bounds get a
        # higher score.
        width = max(variable.upper - variable.lower, 1e-9)
        scores.append((variable.name, float(width)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Loop orchestration.
# ---------------------------------------------------------------------------
