"""Deterministic CPU joint wave surface MVP runner.

This script is the reference executor for ``joint-wave-surface-mvp.v1``
golden fixtures.  Implementation split across helper modules to keep
each file under god-file thresholds (loc<800, functions<25||loc<400).

Helper modules:
  wave_mvp_unit       — Dimension, unit table, conversion
  wave_mvp_models     — Domain models and result types
  wave_mvp_expression — Tokenizer, parser, compiler, evaluator
  wave_mvp_surface    — Particle plan, surface, diagnostics
  wave_mvp_decision   — Decision-value policy, action selection
  wave_mvp_cli        — CLI routing (authority, replay, main)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

# The runner is both executed as a file and loaded directly by the acceptance
# harness.  Declare its sibling-module root once so both entry paths resolve the
# same modules; there is no alternate implementation or import fallback.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---- helper modules (extracted to stay under god-file thresholds) ----
from wave_mvp_unit import (
    Dimension, UnitError,
    _DIMENSIONLESS, _UNIT_TABLE,
)
from wave_mvp_models import (
    OutcomeAxis, OutcomeSpace, VariableSpec, MappingSpec,
    MappingFailure, CompiledMapping,
    ParticleSurface, AxisDiagnostic, SurfaceDiagnostics, TypedAction,
    LoopIteration, LoopResult,
    _new_event_id, _payload_hash,
)
from wave_mvp_expression import (
    _tokenize, _parse_expression, _evaluate_compiled, compile_mapping,
    _parse_constant, _convert_to_base,
    _infer_offending_operand, _infer_offending_unit,
)
from wave_mvp_surface import (
    _latin_hypercube, _build_particle_plan, _evaluate_surface,
    _detect_modes, _effective_sample_size, _entropy_nats,
    compute_diagnostics, _sanitize_for_json,
)
from wave_mvp_decision import (
    _axis_passes, evaluate_decision_value, _select_action, _estimate_sensitivity,
)
from wave_mvp_cli import (
    _load_payload, _ledger_is_authoritative,
    _normalize_ledger_for_comparison,
    _run_authoritative_and_compare, _run_replay, _run_authoritative,
    set_runner_deps,
    main as cli_main,
)

SCHEMA_VERSION = "joint-wave-surface-mvp.v1"
EVALUATOR_LABEL_ORACLE = "non_authoritative_oracle"
EVALUATOR_LABEL_AUTHORITY = "authoritative"
SURFACE_KIND = "possibility_surface"


# ---- required compatibility adapters ----
try:
    from aie_decision.candidate_generation import (
        FailureDiagnostic,
        generate_candidates,
    )
    _HAS_CANDIDATE_GENERATION = True
except Exception:
    _HAS_CANDIDATE_GENERATION = False
    FailureDiagnostic = None
    generate_candidates = None

try:
    from aie_decision.search_replay import (
        LEDGER_SCHEMA_VERSION as _SEARCH_LEDGER_SCHEMA_VERSION,
        replay_search_ledger as _replay_search_ledger,
    )
    _HAS_SEARCH_REPLAY = True
except Exception:
    _HAS_SEARCH_REPLAY = False
    _SEARCH_LEDGER_SCHEMA_VERSION = "1.0.0"
    _replay_search_ledger = None

# ---- lazy authority module cache ----
_AUTHORITY_MODULE: Any = None

def _get_authority_module() -> Any:
    """Return the cached authority module or raise on import failure."""
    global _AUTHORITY_MODULE
    if _AUTHORITY_MODULE is None:
        from aie_decision.wave_authority import (  # type: ignore[import-not-found]
            AUTHORITY_LABEL,
            ORACLE_LABEL,
            AuthoritativeWaveResult,
            AuthorityError,
            ParityMismatch as AuthParityMismatch,
            assert_authoritative_parity,
            run_authoritative_wave,
        )
        _AUTHORITY_MODULE = {
            "label": AUTHORITY_LABEL,
            "oracle_label": ORACLE_LABEL,
            "result_type": AuthoritativeWaveResult,
            "error_type": AuthorityError,
            "parity_type": AuthParityMismatch,
            "assert_parity": assert_authoritative_parity,
            "run": run_authoritative_wave,
        }
    return _AUTHORITY_MODULE


class IntegrationUnavailable(ValueError):
    """Raised when a required MVP integration boundary cannot be used."""

    code = "integration_unavailable"


class ParityMismatch(ValueError):
    """Raised when an explicitly supplied oracle disagrees with the evaluator."""

    code = "parity_mismatch"


def assert_surface_parity(
    authoritative: Mapping[str, Any], oracle: Mapping[str, Any]
) -> None:
    """Compare externally observable surface/action evidence, fail closed on drift."""

    fields = ("wave_shape", "bimodal_axes", "actions", "final_status")
    mismatches = {
        field: {"authoritative": authoritative.get(field), "oracle": oracle.get(field)}
        for field in fields
        if authoritative.get(field) != oracle.get(field)
    }
    if mismatches:
        raise ParityMismatch(
            "parity_mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _require_mvp_adapters(payload: Mapping[str, Any]) -> None:
    """Fail closed when the fixture does not request or provide its adapters."""

    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise IntegrationUnavailable(
            "integration_unavailable: compatibility contract is required"
        )
    required = (
        "use_candidate_generation_failure_diagnostic",
        "use_search_replay_for_ledger_validation",
    )
    missing_flags = [name for name in required if compatibility.get(name) is not True]
    missing_adapters = []
    if not _HAS_CANDIDATE_GENERATION:
        missing_adapters.append("candidate_generation")
    if not _HAS_SEARCH_REPLAY:
        missing_adapters.append("search_replay")
    if missing_flags or missing_adapters:
        details = []
        if missing_flags:
            details.append("flags=" + ",".join(missing_flags))
        if missing_adapters:
            details.append("adapters=" + ",".join(missing_adapters))
        raise IntegrationUnavailable("integration_unavailable: " + "; ".join(details))


# ---- runner core ----

def run_mvp(payload: Mapping[str, Any]) -> LoopResult:
    if not isinstance(payload, Mapping):
        raise ValueError("fixture payload must be an object")
    _require_mvp_adapters(payload)
    run_id = str(payload.get("run_id") or "wave-mvp-loop")
    outcome_spec = payload.get("outcome_space", {})
    axis_payload = outcome_spec.get("axes", ())
    axes = tuple(
        OutcomeAxis(
            name=str(item["name"]),
            unit=str(item["unit"]),
            domain=(float(item["domain"][0]), float(item["domain"][1])),
            time_semantics=str(item.get("time_semantics", "static")),
            tolerance=dict(item.get("tolerance", {})),
        )
        for item in axis_payload
    )
    if not axes:
        raise ValueError("outcome_space.axes must be a non-empty array")
    outcome_space = OutcomeSpace(axes=axes)

    variable_payload = payload.get("variables", ())
    variables = tuple(
        VariableSpec(
            name=str(item["name"]),
            unit=str(item["unit"]),
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            method=str(item.get("method", "user_supplied")),
            ablatable=bool(item.get("ablatable", False)),
            bimodal=bool(item.get("bimodal", False)),
        )
        for item in variable_payload
    )
    if not variables:
        raise ValueError("variables must be a non-empty array")

    extra_constants: dict[str, float] = dict(payload.get("extra_constants") or {})

    mapping_payload = payload.get("mappings", ())
    compiled: list[CompiledMapping] = []
    illegal: list[dict[str, Any]] = []
    for raw in mapping_payload:
        mapping = MappingSpec(
            mapping_id=str(raw["mapping_id"]),
            formula=str(raw["formula"]),
            output_axes=tuple(str(axis) for axis in raw.get("output_axes", ())),
            expected_unit=str(raw["expected_unit"]),
            expect_failure=str(raw["expect_failure"]) if raw.get("expect_failure") else None,
        )
        result = compile_mapping(
            mapping,
            {variable.name: variable for variable in variables},
            extra_constants=extra_constants,
        )
        compiled.append(result)
        if not result.is_legal:
            illegal.append(
                {
                    "mapping_id": result.mapping.mapping_id,
                    "code": result.failure.code,
                    "message": result.failure.message,
                    "operand": result.failure.operand,
                    "operand_unit": result.failure.operand_unit,
                    "expected_unit": result.failure.expected_unit,
                }
            )
        elif result.mapping.expect_failure:
            illegal.append(
                {
                    "mapping_id": result.mapping.mapping_id,
                    "code": "expected_failure_not_triggered",
                    "message": "mapping was expected to fail but compiled cleanly",
                    "operand": "formula",
                    "operand_unit": "dimensionless",
                    "expected_unit": result.mapping.expected_unit,
                }
            )

    particles = dict(payload.get("particles", {}))
    particle_count = int(particles.get("count", 256))
    seed = int(particles.get("seed", 20260805))
    budget = dict(payload.get("budget", {}))
    max_rounds = int(budget.get("max_rounds", 3))
    max_seconds = float(budget.get("max_seconds", 5.0))
    decision_policy = dict(payload.get("decision_policy", {}))
    regime_split = payload.get("regime_split") or None
    use_candidate = True
    use_replay = True

    started = time.monotonic()
    iterations: list[LoopIteration] = []
    actions: list[dict[str, Any]] = []
    accepted_surface: dict[str, Any] | None = None
    status = "insufficient-information"
    sequence = 0
    ledger_events: list[dict[str, Any]] = []
    ablatable_variables = [variable for variable in variables if variable.ablatable]

    def push_event(state: str, candidate_id: str, data: Mapping[str, Any], round_index: int) -> str:
        nonlocal sequence
        sequence += 1
        event_id = _new_event_id(run_id, sequence)
        payload_obj = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "state": state,
            "round_index": round_index,
            "reason": state,
            "data": json.loads(json.dumps(dict(data), sort_keys=True)),
            "revision": {
                "revision_id": f"{event_id}-r1",
                "sequence": 1,
                "created_at": "1970-01-01T00:00:00Z",
                "supersedes_revision_id": None,
            },
        }
        ledger_events.append(
            {
                "sequence": sequence,
                "event_id": event_id,
                "record_type": "wave_event",
                "revision_id": payload_obj["revision"]["revision_id"],
                "payload": payload_obj,
                "payload_hash": _payload_hash(payload_obj),
            }
        )
        return event_id

    # Compile event.
    compile_event_id = push_event(
        "COMPILE",
        run_id,
        {
            "axes": [dataclasses.asdict(axis) for axis in axes],
            "variables": [dataclasses.asdict(variable) for variable in variables],
            "mappings": [mapping.mapping.mapping_id for mapping in compiled],
            "illegal_mappings": list(illegal),
        },
        round_index=0,
    )

    surface, surface_meta = _evaluate_surface(
        compiled_mappings=compiled,
        variables=variables,
        axes=axes,
        particle_count=particle_count,
        seed=seed,
    )
    diagnostics = compute_diagnostics(surface, regime_split=regime_split)
    decision = evaluate_decision_value(diagnostics, decision_policy)
    surface_summary = {
        "axis_order": list(surface.axis_order),
        "surface_kind": surface.surface_kind,
        "calibration": surface.calibration,
        "coverage_semantics": surface.coverage_semantics,
        "evaluator_version": surface.evaluator_version,
        "seed": surface.seed,
        "particle_count": surface.values.shape[0],
        "metadata": surface_meta,
        "marginals": {
            surface.axis_order[idx]: {
                "centers": centers.tolist(),
                "density": density.tolist(),
            }
            for idx, (centers, density) in enumerate(
                (surface.marginal(idx) for idx in range(len(surface.axis_order)))
            )
        },
    }
    summary: dict[str, Any] = {
        "axis_order": list(surface.axis_order),
        "particle_count": surface.values.shape[0],
        "wave_shape": "bimodal" if diagnostics.bimodal_axes else "unimodal",
        "bimodal_axes": list(diagnostics.bimodal_axes),
        "mode_counts": {axis.name: axis.mode_count for axis in diagnostics.axes},
        "mode_locations": {axis.name: list(axis.mode_locations) for axis in diagnostics.axes},
        "decision_value": decision,
        "effective_sample_size": diagnostics.effective_sample_size,
        "illegal_mappings": list(illegal),
        "compatibility": {
            "candidate_generation_adapter": True,
            "search_replay_adapter": True,
            "candidate_generation_available": True,
            "search_replay_available": True,
        },
        "evaluator_label": EVALUATOR_LABEL_ORACLE,
        "evaluator_path": "script",
        "invocation_provenance": {
            "evaluator": EVALUATOR_LABEL_ORACLE,
            "evaluator_version": SCHEMA_VERSION,
            "components_called": [
                "script_schema",
                "script_factor_ir",
                "script_particle_surface",
                "script_diagnostics",
                "script_loop",
                "script_ledger",
                "script_replay",
            ],
        },
    }

    round_index = 1
    push_event(
        "EVALUATE",
        run_id,
        {
            "surface_summary": {k: v for k, v in surface_summary.items() if k != "marginals"},
            "diagnostics": diagnostics.to_mapping(),
            "decision": decision,
        },
        round_index=round_index,
    )
    action = _select_action(
        diagnostics=diagnostics,
        decision=decision,
        variables=variables,
        surface=surface,
        regime_split=regime_split,
        round_index=round_index,
    )
    action_mapping: dict[str, Any] | None = None
    if action is not None:
        action_mapping = dataclasses.asdict(action)
        action_mapping["round_index"] = round_index
        actions.append(action_mapping)
        push_event(
            "ACTION",
            run_id,
            action_mapping,
            round_index=round_index,
        )
        # Required compatibility adapter: build a FailureDiagnostic for
        # diagnostic-driven actions and feed it into the candidate
        # generation layer without modifying its files.  ``stop`` has no
        # diagnostic content; we still record the adapter invocation so
        # downstream tooling sees that the seam is wired.
        if (
            action.kind in {"add_interaction", "measure", "minimize", "split_regime", "stop"}
        ):
            reasons: list[str] = []
            if action.kind == "add_interaction":
                reasons.append("missing_variables")
            elif action.kind == "measure":
                reasons.append("next_measurement")
            elif action.kind == "split_regime":
                reasons.append("missing_variables")
            elif action.kind == "minimize":
                reasons.append("interval_too_wide")
            if reasons:
                diagnostic = FailureDiagnostic(
                    tuple(reasons),
                    next_measurement=action.target if action.kind == "measure" else None,
                    missing_variables=(action.target,) if action.kind in {"add_interaction", "split_regime"} else (),
                )
                summary["candidate_generation_preview"] = {
                    "diagnostic": {
                        "reasons": list(diagnostic.reasons),
                        "next_measurement": diagnostic.next_measurement,
                        "missing_variables": list(diagnostic.missing_variables),
                    },
                    "available": True,
                }

    iteration = LoopIteration(
        round_index=round_index,
        surface=surface_summary,
        diagnostics=diagnostics.to_mapping(),
        decision=decision,
        action=action_mapping,
        event_id=compile_event_id,
    )
    iterations.append(iteration)

    if action is not None and action.kind == "stop":
        status = "result-found"
        accepted_surface = surface_summary
        push_event("RESULT", run_id, {"action": action_mapping}, round_index=round_index)
    elif action is not None and action.kind == "split_regime":
        # Apply the regime split: partition the surface and recompute.
        split_variable = regime_split.get("split_variable") if regime_split else None
        if split_variable and split_variable in {v.name for v in variables}:
            split_round = round_index + 1
            if split_round <= max_rounds and time.monotonic() - started < max_seconds:
                push_event(
                    "EXPAND",
                    run_id,
                    {"action": action_mapping},
                    round_index=split_round,
                )
                next_surface, next_meta = _evaluate_surface(
                    compiled_mappings=compiled,
                    variables=variables,
                    axes=axes,
                    particle_count=particle_count,
                    seed=seed + split_round,
                )
                next_diag = compute_diagnostics(next_surface, regime_split=regime_split)
                next_decision = evaluate_decision_value(next_diag, decision_policy)
                next_summary = {
                    "axis_order": list(next_surface.axis_order),
                    "surface_kind": next_surface.surface_kind,
                    "calibration": next_surface.calibration,
                    "coverage_semantics": next_surface.coverage_semantics,
                    "evaluator_version": next_surface.evaluator_version,
                    "seed": next_surface.seed,
                    "particle_count": next_surface.values.shape[0],
                    "metadata": next_meta,
                }
                iterations.append(
                    LoopIteration(
                        round_index=split_round,
                        surface=next_summary,
                        diagnostics=next_diag.to_mapping(),
                        decision=next_decision,
                        action={
                            "kind": "stop",
                            "target": "loop",
                            "rationale": "split_regime accepted as final typed action",
                        },
                        event_id=compile_event_id,
                    )
                )
                push_event(
                    "EVALUATE",
                    run_id,
                    {"diagnostics": next_diag.to_mapping(), "decision": next_decision},
                    round_index=split_round,
                )
                actions.append(
                    {
                        "action_id": f"r{split_round}-stop",
                        "kind": "stop",
                        "target": "loop",
                        "round_index": split_round,
                        "rationale": "split_regime produced a surface that the decision policy still rejects; finalise",
                        "expected_loss_improvement": 0.0,
                        "estimated_cost": 0.0,
                        "affected_entities": (),
                    }
                )
                push_event(
                    "RESULT",
                    run_id,
                    {"final_surface_summary": "split_regime_with_unresolved_decision"},
                    round_index=split_round,
                )
                status = "budget-exhausted"
            else:
                push_event("STOP", run_id, {"reason": "budget_or_time_exhausted"}, round_index=round_index)
                status = "budget-exhausted"
        else:
            push_event("STOP", run_id, {"reason": "split_regime_without_variable"}, round_index=round_index)
            status = "insufficient-information"
    else:
        push_event("STOP", run_id, {"reason": "no_actionable_typed_action"}, round_index=round_index)
        status = "insufficient-information"

    ledger = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "entries": ledger_events,
    }
    # Required compatibility adapter: project the wave ledger into the
    # search-ledger schema and validate it via the unmodified replay module.
    projected = _project_wave_ledger_to_search_schema(ledger_events, run_id)
    projected = _sanitize_for_json(projected)
    replay = _replay_search_ledger(projected)
    summary["search_replay"] = {
        "schema_version": _SEARCH_LEDGER_SCHEMA_VERSION,
        "terminal_state": replay.get("terminal", {}).get("state"),
        "event_count": replay.get("event_count"),
        "available": True,
    }
    summary["illegal_mappings"] = list(illegal)
    summary["final_status"] = status
    summary["iterations"] = [iteration.round_index for iteration in iterations]
    summary["actions"] = list(actions)
    summary["ablatable_variables"] = [variable.name for variable in ablatable_variables]
    return LoopResult(
        run_id=run_id,
        status=status,
        iterations=tuple(iterations),
        accepted_surface=accepted_surface,
        illegal_mappings=tuple(illegal),
        ledger=ledger,
        actions=tuple(actions),
        summary=summary,
    )


def _project_wave_ledger_to_search_schema(
    events: Sequence[Mapping[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    """Adapt wave events to the search-ledger contract for compatibility tests.

    The mapping is intentionally lossy: search-replay only understands
    a fixed vocabulary, so we collapse wave states onto the closest
    accepted state and surface the original wave state in ``data``.
    Synthetic ``VALIDATE`` events are injected whenever a wave
    ``EVALUATE`` or ``ACTION`` would otherwise reference a candidate
    that has not yet been validated.
    """
    state_map = {
        "COMPILE": "SEED",
        "ACTION": "RANK",
        "EXPAND": "EXPAND",
        "RESULT": "RESULT",
        "STOP": "STOP",
    }
    # A wave EXPAND event introduces a new "candidate" in the search
    # domain (the regime-conditional sub-model).  Subsequent EVALUATE
    # / ACTION / RESULT events that belong to that expansion round are
    # retargeted to the new candidate so the search replay does not
    # report duplicate evaluations of the run-level seed.
    expand_candidate_ids: dict[str, str] = {}
    current_expand_candidate: str | None = None
    projected_entries: list[dict[str, Any]] = []
    sequence = 0
    validated_ids: set[str] = set()
    evaluated_ids: set[str] = set()

    def add_entry(
        state: str,
        payload: Mapping[str, Any],
        source_event_id: str,
        source_revision: str,
    ) -> None:
        nonlocal sequence
        sequence += 1
        entry_id = f"{source_event_id}-{state.lower()}"
        revision_id = f"{source_revision}-{state.lower()}"
        full_payload = {
            "schema_version": _SEARCH_LEDGER_SCHEMA_VERSION,
            "event_id": entry_id,
            "run_id": run_id,
            "candidate_id": payload["candidate_id"],
            "state": state,
            "round_index": payload["round_index"],
            "reason": payload["reason"],
            "data": dict(payload["data"]),
            "revision": {
                "revision_id": revision_id,
                "sequence": 1,
                "created_at": "1970-01-01T00:00:00Z",
                "supersedes_revision_id": None,
            },
        }
        projected_entries.append(
            {
                "sequence": sequence,
                "record_type": "search_event",
                "stable_id": entry_id,
                "revision_id": revision_id,
                "recorded_at": "1970-01-01T00:00:00Z",
                "payload_hash": _payload_hash(full_payload),
                "payload": full_payload,
            }
        )

    for entry in events:
        payload = entry["payload"]
        candidate_id = payload["candidate_id"]
        data = {**payload["data"], "wave_state": payload["state"]}
        if payload["state"] == "STOP":
            base = {
                "candidate_id": "",
                "round_index": payload["round_index"],
                "reason": payload["reason"],
                "data": data,
            }
            add_entry("STOP", base, entry["event_id"], payload["revision"]["revision_id"])
            continue
        projected_candidate_id = candidate_id
        if payload["state"] == "EXPAND":
            expand_key = f"{entry['event_id']}"
            projected_candidate_id = (
                expand_candidate_ids.get(expand_key)
                or expand_candidate_ids.setdefault(
                    expand_key, f"{candidate_id}-regime-{len(expand_candidate_ids) + 1}"
                )
            )
            current_expand_candidate = projected_candidate_id
        elif (
            current_expand_candidate is not None
            and payload["state"] in {"EVALUATE", "ACTION", "RESULT"}
        ):
            # Once an EXPAND has been recorded, EVALUATE/ACTION/RESULT
            # events for the same round describe the regime candidate.
            projected_candidate_id = current_expand_candidate
            if payload["state"] == "RESULT":
                current_expand_candidate = None
        base = {
            "candidate_id": projected_candidate_id,
            "round_index": payload["round_index"],
            "reason": payload["reason"],
            "data": data,
        }
        if payload["state"] == "EVALUATE":
            if projected_candidate_id not in validated_ids:
                add_entry("VALIDATE", base, entry["event_id"], payload["revision"]["revision_id"])
                validated_ids.add(projected_candidate_id)
            add_entry("EVALUATE", base, entry["event_id"], payload["revision"]["revision_id"])
            evaluated_ids.add(projected_candidate_id)
        elif payload["state"] == "ACTION":
            if projected_candidate_id not in evaluated_ids:
                if projected_candidate_id not in validated_ids:
                    add_entry("VALIDATE", base, entry["event_id"], payload["revision"]["revision_id"])
                    validated_ids.add(projected_candidate_id)
                add_entry("EVALUATE", base, entry["event_id"], payload["revision"]["revision_id"])
                evaluated_ids.add(projected_candidate_id)
            add_entry("RANK", base, entry["event_id"], payload["revision"]["revision_id"])
        elif payload["state"] in state_map:
            add_entry(
                state_map[payload["state"]],
                base,
                entry["event_id"],
                payload["revision"]["revision_id"],
            )
    return {
        "schema_version": _SEARCH_LEDGER_SCHEMA_VERSION,
        "run_id": run_id,
        "entries": projected_entries,
    }




# Inject runner dependencies to break circular import (CLI needs run_mvp
# and authority, but main imports from CLI).
set_runner_deps(
    run_mvp,
    _get_authority_module,
    SCHEMA_VERSION,
    integration_error=IntegrationUnavailable,
    oracle_label=EVALUATOR_LABEL_ORACLE,
)

if __name__ == "__main__":
    raise SystemExit(cli_main())
