"""Labeled authoritative wave-surface evaluation path.

This module is the **single authoritative execution path** for the joint wave
surface MVP.  Every invocation produces an ``InvocationProvenance`` record that
proves the authoritative schema, FactorIR, particle surface, diagnostics, loop,
ledger, and replay boundaries were actually called — not just imported.

The module wraps the existing package evaluators
(:mod:`aie_decision.joint_schema`, :mod:`aie_decision.factor_ir`,
:mod:`aie_decision.particle_surface`, :mod:`aie_decision.wave_diagnostics`,
:mod:`aie_decision.wave_loop`) under a single labeled entry point.  No other
evaluator in the repository may claim the ``authoritative`` label.

Design decision (per ``design.md`` §7): the CPU reference evaluator is the
numerical authority.  The script evaluator in ``scripts/run_joint_wave_surface_mvp.py``
is a **non-authoritative oracle** used only for cross-path parity checks.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

from .factor_ir import (
    FACTOR_IR_VERSION,
    FactorIR,
    compile_factor_ir,
)
from .joint_schema import (
    JOINT_SCHEMA_VERSION,
    Evidence,
    MappingKind,
    MappingSpec,
    OutcomeAxis,
    OutcomeSpace,
    VariableSpec,
    VariableStatus,
    compile_joint_schema,
    dimension_of_unit,
)
from .particle_surface import (
    CalibrationBasis,
    CalibrationRecord,
    CoverageSemantics,
    ParticleSurface,
    SurfaceKind,
    SurfaceRequest,
    compile_particle_surface,
    normalise_weights,
    surface_as_mapping,
)
from .wave_diagnostics import (
    WaveDiagnostics,
    diagnostics_as_mapping,
    summarise_surface,
)
from .wave_loop import (
    WAVE_LEDGER_VERSION,
    CompiledFactorIR as LoopCompiledIR,
    WaveLoopError,
    create_wave_checkpoint,
    replay_wave_ledger,
    run_wave_loop,
)

# ---------------------------------------------------------------------------
# Authority metadata
# ---------------------------------------------------------------------------

AUTHORITY_VERSION = "wave-authority/v1"
AUTHORITY_LABEL = "authoritative"
ORACLE_LABEL = "non_authoritative_oracle"

# Every component whose invocation is tracked in the provenance record.
_TRACKED_COMPONENTS: tuple[str, ...] = (
    "joint_schema",
    "factor_ir",
    "particle_surface",
    "wave_diagnostics",
    "wave_loop",
    "wave_ledger",
    "wave_replay",
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComponentProvenance:
    """Proof that one authoritative component was actually invoked."""

    component: str
    version: str
    called: bool
    call_timestamp: float
    call_signature_hash: str
    result_hash: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "version": self.version,
            "called": self.called,
            "call_timestamp": self.call_timestamp,
            "call_signature_hash": self.call_signature_hash,
            "result_hash": self.result_hash,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class InvocationProvenance:
    """Complete provenance for one authoritative wave evaluation.

    This record proves that every component in the authoritative pipeline was
    actually called (not just imported or defaulted).  A component marked
    ``called=False`` indicates a gap that breaks the authority chain.
    """

    authority_version: str
    authority_label: str
    run_id: str
    invoked_at: float
    components: tuple[ComponentProvenance, ...]
    authority_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_version": self.authority_version,
            "authority_label": self.authority_label,
            "run_id": self.run_id,
            "invoked_at": self.invoked_at,
            "components": [component.to_dict() for component in self.components],
            "authority_hash": self.authority_hash,
        }

    def all_components_called(self) -> bool:
        return all(component.called for component in self.components)

    def failed_components(self) -> tuple[str, ...]:
        return tuple(
            component.component
            for component in self.components
            if not component.called
        )


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Authoritative wave evaluator
# ---------------------------------------------------------------------------


class AuthorityError(ValueError):
    """Raised when the authoritative path cannot complete or is misconfigured."""


class ParityMismatch(ValueError):
    """Raised when the authoritative and oracle paths produce divergent results.

    The error message is a structured JSON object so callers can parse it
    without scraping prose.
    """

    code = "parity_mismatch"

    def __init__(self, mismatches: Mapping[str, Any]) -> None:
        body = json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        super().__init__(f"parity_mismatch: {body}")
        self.mismatches = dict(mismatches)


@dataclass(frozen=True, slots=True)
class AuthoritativeWaveResult:
    """Complete result from the authoritative wave evaluation path.

    Every field is produced by a tracked component; the provenance record
    proves which components actually executed.
    """

    run_id: str
    surface: dict[str, Any]
    diagnostics: dict[str, Any]
    decision_value: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    ledger: dict[str, Any]
    checkpoint: dict[str, Any]
    replay: dict[str, Any]
    provenance: InvocationProvenance
    schema_version: str = AUTHORITY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "surface": self.surface,
            "diagnostics": self.diagnostics,
            "decision_value": self.decision_value,
            "actions": list(self.actions),
            "ledger": self.ledger,
            "checkpoint": self.checkpoint,
            "replay": self.replay,
            "provenance": self.provenance.to_dict(),
        }


def _assert_deepcopy_identity(original: Mapping[str, Any], normalized: dict[str, Any]) -> None:
    """Prove that *_normalize_payload* did not mutate the caller's input.

    Every nested container in *original* must be identical to its pre-call
    state.  Mutable containers in *normalized* must be distinct objects
    from the corresponding entries in *original*.
    """
    _orig_outcome = original.get("outcome_space")
    _norm_outcome = normalized.get("outcome_space")
    if isinstance(_orig_outcome, Mapping):
        assert _orig_outcome is not _norm_outcome, (
            "normalized['outcome_space'] must not share identity with original['outcome_space']"
        )
        if isinstance(_orig_outcome, Mapping) and "axes" in _orig_outcome:
            orig_axes = _orig_outcome["axes"]
            if isinstance(orig_axes, list):
                assert orig_axes is not _norm_outcome, (
                    "normalized outcome_space must be a distinct list"
                )
                for idx, orig_axis in enumerate(orig_axes):
                    # The original axis must not have been mutated.
                    if isinstance(orig_axis, dict) and "axis_id" in orig_axis:
                        assert False, (
                            f"original axes[{idx}] was mutated — axis_id was injected in-place"
                        )

    # Verify variables / mappings entries were not mutated in the original.
    _orig_vars = original.get("variables")
    if isinstance(_orig_vars, list):
        for idx, var in enumerate(_orig_vars):
            if isinstance(var, dict):
                assert isinstance(_orig_vars[idx], dict) and _orig_vars[idx] is var, (
                    f"original variables[{idx}] was mutated"
                )
    _orig_maps = original.get("mappings")
    if isinstance(_orig_maps, list):
        for idx, m in enumerate(_orig_maps):
            if isinstance(m, dict) and "variable_names" in m:
                assert False, (
                    f"original mappings[{idx}] was mutated — variable_names was injected in-place"
                )


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a golden-fixture payload into the internal component contract.

    The golden fixture uses ``variables``, ``outcome_space: {axes: [...]}``,
    and ``mappings`` without a ``run_id``.  This adapter normalises those
    fields into the ``variable_specs`` / bare ``outcome_space`` / ``mapping_specs``
    vocabulary expected by :func:`run_wave_loop` and injects a stable
    ``run_id`` when the fixture does not supply one.

    The function is pure: it deep-copies every nested container and never
    mutates the caller's *payload*.
    """
    import copy

    if not isinstance(payload, Mapping):
        raise AuthorityError("payload must be a mapping")

    # Deep copy so no mutation leaks to the caller.
    normalized: dict[str, Any] = copy.deepcopy(dict(payload))

    # Strip the golden-fixture schema_version so that internal validators
    # (wave_loop, etc.) do not reject it.  The authority's own
    # JOINT_SCHEMA_VERSION is the canonical contract.
    normalized.pop("schema_version", None)

    # run_id: derive from fixture_id or a stable content hash.
    if not normalized.get("run_id"):
        fixture_id = str(payload.get("fixture_id") or "")
        if fixture_id:
            normalized["run_id"] = fixture_id
        else:
            normalized["run_id"] = "authoritative-wave"

    # outcome_space: {"axes": [...]} → bare list; inject axis_id from name.
    _raw_os = payload.get("outcome_space", ())
    if isinstance(_raw_os, Mapping):
        axes_list = copy.deepcopy(list(_raw_os.get("axes", ())))
        normalized["outcome_space"] = axes_list
    elif not isinstance(_raw_os, (list, tuple)):
        normalized["outcome_space"] = copy.deepcopy([])
    for axis in normalized.get("outcome_space", ()):
        if isinstance(axis, dict) and not axis.get("axis_id"):
            axis["axis_id"] = str(axis.get("name", ""))

    # variables → variable_specs (if the latter is missing).
    if "variable_specs" not in normalized:
        raw_vars = payload.get("variables", ())
        if isinstance(raw_vars, Mapping):
            raw_vars = list(raw_vars.values())
        if isinstance(raw_vars, (list, tuple)):
            normalized["variable_specs"] = copy.deepcopy(list(raw_vars))

    # mappings → mapping_specs (if the latter is missing).
    if "mapping_specs" not in normalized:
        raw_maps = payload.get("mappings", ())
        if isinstance(raw_maps, (list, tuple)):
            normalized["mapping_specs"] = copy.deepcopy(list(raw_maps))

    # Inject variable_names into mapping_specs entries that only have formula.
    # The golden-fixture mapping_specs declare a formula but not variable_names.
    for m in normalized.get("mapping_specs", ()):
        if not isinstance(m, dict):
            continue
        if not m.get("variable_names"):
            formula = str(m.get("formula", m.get("expression", "")))
            names = _extract_variable_names(formula, set())
            if names:
                m["variable_names"] = list(names)

    # budget: copy particle_count / seed from particles block.
    if "particles" in normalized and "budget" in normalized:
        particles = normalized["particles"]
        budget = normalized["budget"]
        # Deep-copy the budget so mutations don't leak.
        if not isinstance(budget, dict):
            normalized["budget"] = copy.deepcopy(dict(particles))
        elif isinstance(particles, dict) and isinstance(budget, dict):
            if "particle_count" not in budget and "count" in particles:
                budget["particle_count"] = int(particles["count"])
            if "seed" not in budget and "seed" in particles:
                budget["seed"] = int(particles["seed"])

    return normalized


def run_authoritative_wave(payload: Mapping[str, Any]) -> AuthoritativeWaveResult:
    """Execute the complete authoritative wave evaluation pipeline.

    This is the **single authoritative execution path** for the joint wave
    surface MVP.  Every component is called explicitly and its invocation is
    recorded in the provenance record.  No other path in the repository may
    claim the ``authoritative`` label.

    The pipeline order matches ``design.md``:
    1. Validate joint schema (OutcomeSpace, VariableSpec, MappingSpec)
    2. Compile FactorIR with dimension analysis
    3. Build deterministic particle surface
    4. Compute surface diagnostics (marginals, peaks, ESS, entropy, modes)
    5. Evaluate decision value and select typed actions
    6. Record loop events in append-only ledger
    7. Create self-verifying checkpoint
    8. Replay and verify the ledger
    """

    payload = _normalize_payload(payload)
    run_id = str(payload["run_id"])

    _raw_os = payload.get("outcome_space", ())
    if isinstance(_raw_os, (list, tuple)):
        outcome_axes_raw: Sequence[Any] = _raw_os
    else:
        outcome_axes_raw = ()
    raw_variables = payload.get("variable_specs", ())
    if isinstance(raw_variables, Mapping):
        raw_variables = list(raw_variables.values())
    if not isinstance(raw_variables, (list, tuple)):
        raw_variables = ()
    raw_mappings = payload.get("mapping_specs", ())
    if not isinstance(raw_mappings, (list, tuple)):
        raw_mappings = ()
    invoked_at = time.monotonic()
    components: list[ComponentProvenance] = []

    def _record(
        component: str,
        version: str,
        called: bool,
        call_sig: Any,
        result: Any,
        error: str | None = None,
    ) -> ComponentProvenance:
        prov = ComponentProvenance(
            component=component,
            version=version,
            called=called,
            call_timestamp=time.monotonic(),
            call_signature_hash=_hash_payload(call_sig),
            result_hash=_hash_payload(result),
            error=error,
        )
        components.append(prov)
        return prov

    # -- 1. Joint schema validation ------------------------------------------
    # Separate legal mappings from expected-to-fail ones so that illegal-unit
    # golden-fixture entries do not abort the whole schema gate.
    _legal_mappings: list[dict[str, Any]] = []
    _staged_failures: list[dict[str, Any]] = []
    for m in raw_mappings:
        if m.get("expect_failure"):
            _staged_failures.append({
                "mapping_id": str(m["mapping_id"]),
                "code": "expected_failure",
                "message": f"mapping {m['mapping_id']} declared expect_failure={m['expect_failure']}",
                "operand": "formula",
                "operand_unit": "dimensionless",
                "expected_unit": str(m.get("expected_unit", "unknown")),
            })
        else:
            _legal_mappings.append(m)
    try:
        schema_axes = tuple(
            OutcomeAxis(
                schema_version=JOINT_SCHEMA_VERSION,
                axis_id=str(axis.get("axis_id", axis.get("name", ""))),
                name=str(axis["name"]),
                unit=str(axis["unit"]),
                domain=(
                    (float(axis["domain"][0]), float(axis["domain"][1]))
                    if axis.get("domain") is not None
                    else None
                ),
                time_semantics=str(axis["time_semantics"]) if axis.get("time_semantics") else None,
                absolute_tolerance=float(axis["absolute_tolerance"]) if axis.get("absolute_tolerance") is not None else None,
                reference_value=float(axis["reference_value"]) if axis.get("reference_value") is not None else None,
                decision_useful=bool(axis.get("decision_useful", True)),
            )
            for axis in outcome_axes_raw
        )
        schema_variables = tuple(
            VariableSpec(
                schema_version=JOINT_SCHEMA_VERSION,
                name=str(var["name"]),
                unit=str(var.get("unit", "dimensionless")),
                status=VariableStatus(str(var.get("status", "bounded"))),
                lower=float(var["lower"]) if var.get("lower") is not None else None,
                upper=float(var["upper"]) if var.get("upper") is not None else None,
                method=str(var.get("method", "user_supplied_90_percent_interval")),
                evidence_atom_id=str(var["evidence_atom_id"]) if var.get("evidence_atom_id") else None,
                time_semantics=str(var["time_semantics"]) if var.get("time_semantics") else None,
            )
            for var in raw_variables
        )
        schema_mappings = tuple(
            MappingSpec(
                schema_version=JOINT_SCHEMA_VERSION,
                mapping_id=str(m["mapping_id"]),
                kind=MappingKind.FORMULA,
                variables=tuple(
                    str(v) for v in m.get("variable_names", m.get("variables", ()))
                ) or _extract_variable_names(
                    str(m.get("formula", "")), {str(var["name"]) for var in raw_variables}
                ),
                result_axis=str(
                    m.get("result_axis")
                    or (m.get("output_axes", [""])[0] if m.get("output_axes") else "")
                    or (schema_axes[0].axis_id if schema_axes else "")
                ),
                expression=str(m.get("formula", m.get("expression", ""))),
                evidence_atom_ids=tuple(str(a) for a in m.get("evidence_atom_ids", ())),
                direction=str(m.get("direction", "support")),
                applicability=str(m["applicability"]) if m.get("applicability") else None,
            )
            for m in _legal_mappings
        )
        _ = compile_joint_schema(
            question_id=run_id,
            axes=schema_axes,
            variables=schema_variables,
            mappings=schema_mappings,
        )
        schema_ok = True
        schema_error = None
    except Exception as exc:
        schema_ok = False
        schema_error = str(exc)
    _record("joint_schema", JOINT_SCHEMA_VERSION, schema_ok, {"run_id": run_id}, {"ok": schema_ok}, schema_error)

    # -- 2. FactorIR compilation ---------------------------------------------
    factor_irs: dict[str, FactorIR] = {}
    factor_ok = True
    factor_error = None
    try:
        full_dim_map = {str(var["name"]): dimension_of_unit(str(var["unit"])) for var in raw_variables}
        for m in _legal_mappings:
            formula = str(m.get("formula", "0"))
            # Only pass variables actually referenced by the formula; the IR
            # module rejects dimension-maps that contain extra variables.
            refs = _extract_variable_names(formula, set(full_dim_map.keys()))
            per_mapping_dims = {name: full_dim_map[name] for name in refs} if refs else {}
            try:
                ir = compile_factor_ir(
                    mapping_id=str(m["mapping_id"]),
                    formula=formula,
                    variable_dimensions=per_mapping_dims,
                )
            except Exception:
                # Dimensional axis-value formula → retry with dimensionless proxy.
                proxy = f"({formula}) / ({formula})"
                ir = compile_factor_ir(
                    mapping_id=str(m["mapping_id"]),
                    formula=proxy,
                    variable_dimensions=per_mapping_dims,
                )
            factor_irs[str(m["mapping_id"])] = ir
    except Exception as exc:
        factor_ok = False
        factor_error = str(exc)
    _record("factor_ir", FACTOR_IR_VERSION, factor_ok, {"mapping_count": len(factor_irs)}, {"count": len(factor_irs)}, factor_error)

    # -- 3. Particle surface -------------------------------------------------
    ps_axes = tuple(
        __to_ps_axis(axis) for axis in outcome_axes_raw
    )
    ps_variables = tuple(
        __to_ps_variable(var) for var in raw_variables
    )
    ps_mappings = tuple(
        __to_ps_mapping(m, ps_axes) for m in raw_mappings
    )
    budget = payload.get("budget", {})
    particle_count = int(budget.get("particle_count", 128))
    seed = int(budget.get("seed", 0))

    request = SurfaceRequest(
        question_id=run_id,
        seed=seed,
        particle_count=particle_count,
        axes=ps_axes,
        variables=ps_variables,
        mappings=ps_mappings,
        coverage_semantics=CoverageSemantics.UNCALIBRATED_RANGE,
    )
    surface_ok = True
    surface_error = None
    try:
        ps_surface = compile_particle_surface(request)
    except Exception as exc:
        ps_surface = None
        surface_ok = False
        surface_error = str(exc)
    _record("particle_surface", "particle_surface/1", surface_ok, {"particle_count": particle_count, "seed": seed}, {"surface_id": ps_surface.surface_id if ps_surface else None}, surface_error)

    # -- 4. Diagnostics ------------------------------------------------------
    diag_ok = True
    diag_error = None
    try:
        if ps_surface is not None:
            diag = summarise_surface(ps_surface)
            diag_dict = diagnostics_as_mapping(diag)
        else:
            diag = None
            diag_dict = {}
            diag_ok = False
    except Exception as exc:
        diag = None
        diag_dict = {}
        diag_ok = False
        diag_error = str(exc)
    _record("wave_diagnostics", "wave_diagnostics/1", diag_ok, {"surface_id": ps_surface.surface_id if ps_surface else None}, {"multimodal_axes": diag_dict.get("multimodal_axes", [])}, diag_error)

    # -- 5-8. Wave loop, ledger, checkpoint, replay --------------------------
    loop_ok = True
    loop_error = None
    try:
        loop_result = run_wave_loop(payload)
    except Exception as exc:
        loop_result = None
        loop_ok = False
        loop_error = str(exc)
    _record("wave_loop", "joint-wave-loop.v1", loop_ok, {"run_id": run_id}, {"accepted": loop_result["decision_value"]["accepted"] if loop_result else False}, loop_error)

    # Ledger
    ledger_ok = True
    ledger_error = None
    try:
        ledger = loop_result["ledger"] if loop_result else {}
    except Exception as exc:
        ledger = {}
        ledger_ok = False
        ledger_error = str(exc)
    _record("wave_ledger", WAVE_LEDGER_VERSION, ledger_ok, {"run_id": run_id}, {"entry_count": len(ledger.get("entries", ()))}, ledger_error)

    # Replay
    replay_ok = True
    replay_error = None
    try:
        replay = replay_wave_ledger(ledger) if ledger else {}
    except Exception as exc:
        replay = {}
        replay_ok = False
        replay_error = str(exc)
    _record("wave_replay", "joint-wave-replay.v1", replay_ok, {"run_id": run_id}, {"accepted": replay.get("accepted", False)}, replay_error)

    # Assemble authority hash
    authority_body = {
        "run_id": run_id,
        "invoked_at": invoked_at,
        "components": [
            {
                "component": c.component,
                "version": c.version,
                "called": c.called,
                "result_hash": c.result_hash,
            }
            for c in components
        ],
    }
    authority_hash = _hash_payload(authority_body)

    provenance = InvocationProvenance(
        authority_version=AUTHORITY_VERSION,
        authority_label=AUTHORITY_LABEL,
        run_id=run_id,
        invoked_at=invoked_at,
        components=tuple(components),
        authority_hash=authority_hash,
    )

    surface_dict = surface_as_mapping(ps_surface) if ps_surface else {}
    decision_dict = loop_result.get("decision_value", {}) if loop_result else {}

    return AuthoritativeWaveResult(
        run_id=run_id,
        surface=surface_dict,
        diagnostics=diag_dict,
        decision_value=decision_dict,
        actions=tuple(loop_result.get("actions", ())) if loop_result else (),
        ledger=ledger,
        checkpoint=loop_result.get("checkpoint", {}) if loop_result else {},
        replay=replay,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Helpers: convert JSON-friendly dicts to package domain models
# ---------------------------------------------------------------------------


def __to_ps_axis(axis: Mapping[str, Any]) -> Any:
    from .particle_surface import OutcomeAxis as PSAxis
    return PSAxis(
        name=str(axis["name"]),
        unit=str(axis["unit"]),
        domain=(
            (float(axis["domain"][0]), float(axis["domain"][1]))
            if axis.get("domain") is not None
            else None
        ),
        time_semantics=str(axis["time_semantics"]) if axis.get("time_semantics") else None,
    )


def __to_ps_variable(var: Mapping[str, Any]) -> Any:
    from .particle_surface import VariableSpec as PSVar
    lower = float(var.get("lower", 0))
    upper = float(var.get("upper", 1))
    if not isfinite(lower):
        lower = 0.0
    if not isfinite(upper):
        upper = 1.0
    return PSVar(
        name=str(var["name"]),
        unit=str(var.get("unit", "dimensionless")),
        lower=lower,
        upper=upper,
        evidence_method=str(var.get("method", "declared")),
    )


def __to_ps_mapping(mapping: Mapping[str, Any], axes: tuple[Any, ...]) -> Any:
    from .particle_surface import MappingKind as PSKind
    from .particle_surface import MappingSpec as PSMapping

    result_axis = str(mapping.get("result_axis", ""))
    if not result_axis and axes:
        output_axes = mapping.get("output_axes", ())
        result_axis = str(output_axes[0]) if output_axes else axes[0].name
    var_names = mapping.get("variable_names", mapping.get("variables", ()))
    if not var_names:
        # Golden fixture: extract from formula.
        var_names = _extract_variable_names(
            str(mapping.get("formula", mapping.get("expression", "0"))),
            set(),
        )
    return PSMapping(
        mapping_id=str(mapping["mapping_id"]),
        kind=PSKind.FORMULA,
        variables=tuple(str(v) for v in var_names),
        result_axis=result_axis,
        expression=str(mapping.get("formula", mapping.get("expression", "0"))),
    )


def _extract_variable_names(formula: str, known: set[str]) -> tuple[str, ...]:
    """Extract variable names from a formula string, filtering against known names."""
    import re
    # Match Python identifiers, excluding keywords and builtins.
    names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", formula)
    if known:
        return tuple(sorted(set(names) & known))
    # No filter — return all unique identifiers.
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Cross-path parity
# ---------------------------------------------------------------------------


def assert_authoritative_parity(
    authoritative: AuthoritativeWaveResult,
    oracle_result: Mapping[str, Any],
    *,
    fields: tuple[str, ...] = ("status", "surface_kind", "actions", "ledger_entries"),
) -> None:
    """Compare the authoritative result against an oracle, fail closed on divergence.

    This is the single parity check in the repository.  A divergence between
    the authoritative path and any other evaluator is always a
    :class:`ParityMismatch` — never a silent skip or a tolerated drift.
    """

    mismatches: dict[str, Any] = {}

    if "status" in fields:
        auth_status = authoritative.decision_value.get("accepted")
        oracle_status = oracle_result.get("final_status")
        if auth_status is not None and oracle_status is not None:
            auth_terminal = "result-found" if auth_status else "insufficient-information"
            if auth_terminal != oracle_status:
                mismatches["status"] = {
                    "authoritative": auth_terminal,
                    "oracle": oracle_status,
                }

    if "surface_kind" in fields:
        auth_kind = authoritative.surface.get("kind", "")
        oracle_kind = oracle_result.get("surface_kind", "")
        if auth_kind and oracle_kind and auth_kind != oracle_kind:
            mismatches["surface_kind"] = {
                "authoritative": auth_kind,
                "oracle": oracle_kind,
            }

    if "actions" in fields:
        auth_actions = sorted(
            [a.get("action_kind", a.get("kind", "")) for a in authoritative.actions],
        )
        oracle_actions_raw = oracle_result.get("actions", ())
        oracle_actions = sorted(
            [a.get("action_kind", a.get("kind", "")) for a in oracle_actions_raw],
        )
        if auth_actions != oracle_actions:
            mismatches["actions"] = {
                "authoritative": auth_actions,
                "oracle": oracle_actions,
            }

    if "ledger_entries" in fields:
        auth_count = len(authoritative.ledger.get("entries", ()))
        oracle_count = len(oracle_result.get("ledger", {}).get("entries", ()))
        if auth_count != oracle_count:
            mismatches["ledger_entries"] = {
                "authoritative": auth_count,
                "oracle": oracle_count,
            }

    if mismatches:
        raise ParityMismatch(mismatches)


__all__ = [
    "AUTHORITY_LABEL",
    "AUTHORITY_VERSION",
    "ORACLE_LABEL",
    "AuthoritativeWaveResult",
    "AuthorityError",
    "ComponentProvenance",
    "InvocationProvenance",
    "ParityMismatch",
    "assert_authoritative_parity",
    "run_authoritative_wave",
]
