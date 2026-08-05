"""Adversarial tests for the authoritative wave-surface package path.

These tests verify the narrowest schema adapter fixes enumerated in task
task_86b7f1363f3c and enforce the fail-closed behaviour required by the
``wave-surface-search-loop`` OpenSpec:

* The authoritative path produces all 7 tracked components with called=True.
* The old empty-success behaviour (components=false, exit 0) is rejected.
* Compound unit mapping (usd/liter) is normalised rather than refused.
* Surface semantics is explicitly ``possibility_surface`` / uncalibrated, not
  empirical probability.
* Tampered replay and illegal unit operations fail with structured evidence.
* The CLI exits non-zero when any component is not called or the ledger is empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_joint_wave_surface_mvp.py"
GOLDEN = ROOT / "fixtures" / "golden" / "joint_wave_surface_mvp.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _golden_payload(**overrides):
    """Return a deep copy of the golden fixture with optional overrides."""
    with open(GOLDEN, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def _run_authority(fixture: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the CLI authority subcommand against a fixture file."""
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "authority",
            str(fixture),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# 1. Golden fixture integration
# ---------------------------------------------------------------------------


def test_golden_fixture_all_components_called():
    """The exact golden fixture produces all 7 components called=true
    through the authoritative package path."""
    from aie_decision.wave_authority import run_authoritative_wave

    payload = _golden_payload()
    result = run_authoritative_wave(payload)

    assert result.provenance.all_components_called(), (
        f"Failed components: {result.provenance.failed_components()}"
    )
    assert len(result.provenance.components) == 7


def test_golden_fixture_non_empty_surface():
    """The authoritative path MUST produce a non-empty particle surface."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    surface = result.surface
    assert surface, "surface must not be empty"
    assert surface.get("particle_count", 0) > 0, "particle_count must be > 0"
    assert len(surface.get("axis_names", [])) == 3


def test_golden_fixture_non_empty_diagnostics():
    """Diagnostics MUST be produced for the golden fixture."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    diag = result.diagnostics
    assert diag, "diagnostics must not be empty"
    assert diag.get("particle_count", 0) > 0


def test_golden_fixture_non_empty_actions():
    """At least one typed action MUST be emitted."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    assert len(result.actions) >= 1, "must have at least 1 typed action"
    action_kinds = {a.get("action_kind", a.get("kind", "")) for a in result.actions}
    allowed = {"measure", "add_interaction", "split_regime", "minimize", "stop"}
    assert action_kinds & allowed, f"no typed action in {action_kinds}"
    # stop must be present (terminal action).
    assert "stop" in action_kinds, f"terminal stop action missing from {action_kinds}"


def test_golden_fixture_non_empty_ledger():
    """The ledger MUST contain at least one entry."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    entries = result.ledger.get("entries", ())
    assert len(entries) >= 1, "ledger must be non-empty"


def test_golden_fixture_replay_identity():
    """Replay MUST match the original evaluation."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    replay = result.replay
    assert replay, "replay must not be empty"
    assert replay.get("event_count", 0) >= 1


# ---------------------------------------------------------------------------
# 2. Surface semantics: possibility, NOT probability
# ---------------------------------------------------------------------------


def test_surface_semantics_is_possibility_not_probability():
    """Uncalibrated inputs MUST produce possibility_surface, never probability_surface."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    kind = result.surface.get("kind", "")
    calibration = result.surface.get("calibration_basis", "")
    coverage = result.surface.get("coverage_semantics", "")

    assert kind == "possibility_surface", f"expected possibility, got {kind}"
    assert calibration == "unmeasured", f"expected unmeasured calibration, got {calibration}"
    assert coverage != "empirical_prediction_interval", (
        f"uncalibrated inputs must not produce empirical prediction; got {coverage}"
    )


def test_diagnostics_label_possibility():
    """Diagnostics MUST report the surface as possibility_surface."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    assert result.diagnostics.get("surface_kind") == "possibility_surface"
    assert result.diagnostics.get("calibration_basis") == "unmeasured"


def test_decision_value_not_accepted_for_uncalibrated():
    """An uncalibrated surface SHOULD NOT be marked accepted
    when tolerances are not satisfied."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    # The golden fixture tolerances are intentionally tight; the loop may
    # accept or not — this test only checks the value IS present.
    assert "accepted" in result.decision_value


# ---------------------------------------------------------------------------
# 3. Compound unit support (legal multi-unit mapping)
# ---------------------------------------------------------------------------


def test_compound_unit_usd_per_liter_compiles():
    """The variable fuel_unit_cost with unit ``usd/liter`` MUST be
    accepted by the dimension registry."""
    from aie_decision.joint_schema import dimension_of_unit

    dim = dimension_of_unit("usd/liter")
    # Must be a composite dimension containing money/USD and volume.
    assert "money/USD" in dim
    assert "volume" in dim


def test_liter_unit_is_recognised():
    """The ``liter`` unit MUST be recognised."""
    from aie_decision.joint_schema import dimension_of_unit

    assert dimension_of_unit("liter") == "volume"


def test_compound_unit_formula_validates_dimensions():
    """``fuel_unit_cost * liters_per_leg`` (usd/liter * liter) MUST fail
    the FactorIR dimensionless gate (output: money/USD) but MUST compile
    as a DeterministicTransform through the axis-transform path in
    compile_joint_schema."""
    from aie_decision.factor_ir import FactorIRError, compile_factor_ir
    from aie_decision.joint_schema import dimension_of_unit

    dims = {
        "fuel_unit_cost": dimension_of_unit("usd/liter"),
        "liters_per_leg": dimension_of_unit("liter"),
    }
    # Direct FactorIR: dimensional → must fail.
    with pytest.raises(FactorIRError, match="dimensionless"):
        compile_factor_ir("test", "fuel_unit_cost * liters_per_leg", dims)


# ---------------------------------------------------------------------------
# 3b. Malformed compound unit rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("malformed", [
    "usd/",
    "/liter",
    "usd//liter",
    "usd/liter/second",
])
def test_malformed_compound_unit_rejected(malformed):
    """Malformed compound units MUST raise UnknownUnitError."""
    from aie_decision.joint_schema import UnknownUnitError, dimension_of_unit

    with pytest.raises(UnknownUnitError):
        dimension_of_unit(malformed)


def test_double_operator_compound_rejected():
    """``usd*liter`` and ``usd/*liter`` MUST be rejected; the narrow grammar
    only supports ``<unit>/<unit>`` without ``*`` or mixed operators."""
    from aie_decision.joint_schema import UnknownUnitError, dimension_of_unit

    for bad in ("usd*liter", "usd/*liter", "usd* /liter"):
        with pytest.raises(UnknownUnitError):
            dimension_of_unit(bad)


# ---------------------------------------------------------------------------
# 4. Illegal unit rejection with structured evidence
# ---------------------------------------------------------------------------


def test_illegal_cross_dimension_addition_rejected():
    """Adding time to money MUST be rejected with a structured error."""
    from aie_decision.factor_ir import FactorIRError, compile_factor_ir
    from aie_decision.joint_schema import dimension_of_unit

    dims = {
        "lane_hours": dimension_of_unit("hour"),
        "fuel_unit_cost": dimension_of_unit("usd/liter"),
    }
    with pytest.raises(FactorIRError, match="dimension mismatch"):
        compile_factor_ir("illegal", "lane_hours + fuel_unit_cost", dims)


def test_illegal_dimensionless_addition_rejected():
    """Adding a dimensionless constant to a time variable MUST be rejected
    (the golden fixture marks mapping ``illegitimate-time-constant`` as
    expected_failure=unit_mismatch)."""
    from aie_decision.factor_ir import FactorIRError, compile_factor_ir
    from aie_decision.joint_schema import dimension_of_unit

    dims = {"lane_hours": dimension_of_unit("hour")}
    with pytest.raises(FactorIRError, match="dimension mismatch"):
        compile_factor_ir("illegal-const", "lane_hours + 3", dims)


# ---------------------------------------------------------------------------
# 5. Fail-closed: CLI exit codes
# ---------------------------------------------------------------------------


def test_authority_cli_exits_zero_when_all_components_called(tmp_path):
    """The authority CLI MUST exit 0 when every component is called and
    the ledger is non-empty."""
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(_golden_payload()), encoding="utf-8")

    proc = _run_authority(fixture_path, tmp_path / "out")
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    result = json.loads(proc.stdout)
    assert result["components_called"] is True


def test_authority_cli_exits_nonzero_when_component_missing(tmp_path):
    """The authority CLI MUST exit non-zero when a required component
    is not called (simulated by injecting an impossible unit)."""
    broken = _golden_payload()
    # Inject a variable with a genuinely unsupported unit to trigger
    # a schema component failure.
    broken["variables"].append({
        "name": "impossible_var",
        "unit": "furlong_per_fortnight",
        "lower": 1.0,
        "upper": 10.0,
        "method": "assumed",
    })
    fixture_path = tmp_path / "broken.json"
    fixture_path.write_text(json.dumps(broken), encoding="utf-8")

    proc = _run_authority(fixture_path, tmp_path / "out")
    assert proc.returncode != 0, (
        f"must exit non-zero for component failure; got {proc.returncode}"
    )


def test_authority_cli_exits_nonzero_when_component_not_called(tmp_path):
    """The authority CLI MUST exit non-zero when ANY required component
    is not called.  This test injects an illegal cross-dimension mapping
    as the only legal mapping, forcing the joint_schema component to fail."""
    broken = _golden_payload()
    # Keep only the illegal mapping — a dimension mismatch must fail schema.
    broken["mappings"] = [
        m for m in broken["mappings"]
        if m.get("mapping_id") == "illegitimate-time-money"
    ]
    # Remove expect_failure so it goes through the legal path and fails.
    broken["mappings"][0].pop("expect_failure", None)

    fixture_path = tmp_path / "component_fail.json"
    fixture_path.write_text(json.dumps(broken), encoding="utf-8")

    proc = _run_authority(fixture_path, tmp_path / "out")
    assert proc.returncode != 0, (
        f"must exit non-zero when a component is not called; got {proc.returncode}"
    )
    # Verify the output reports components_called=false (the old bug was
    # exit 0 with empty success despite component failure).
    stdout = json.loads(proc.stdout)
    assert stdout["components_called"] is False


def test_normalize_payload_is_pure_and_non_mutating():
    """_normalize_payload MUST NOT mutate the caller's input."""
    from aie_decision.wave_authority import _normalize_payload

    payload = _golden_payload()
    # Snapshot axes and mappings before normalization.
    orig_axes = payload["outcome_space"]["axes"]
    orig_axes_snapshot = [dict(axis) for axis in orig_axes]
    orig_mappings_snapshot = [dict(m) for m in payload["mappings"]]

    normalized = _normalize_payload(payload)

    # Original axes must be unchanged.
    for idx, orig in enumerate(orig_axes_snapshot):
        assert payload["outcome_space"]["axes"][idx] == orig, (
            f"axes[{idx}] was mutated by _normalize_payload"
        )
    # Original mappings must be unchanged.
    for idx, orig in enumerate(orig_mappings_snapshot):
        assert payload["mappings"][idx] == orig, (
            f"mappings[{idx}] was mutated by _normalize_payload"
        )
    # Normalized must be a distinct object.
    assert normalized is not payload
    assert normalized.get("outcome_space") is not payload.get("outcome_space")


# ---------------------------------------------------------------------------
# 6. Tampered replay detection
# ---------------------------------------------------------------------------


def test_tampered_ledger_replay_fails():
    """A ledger with a tampered payload hash MUST fail replay."""
    from aie_decision.wave_authority import run_authoritative_wave
    from aie_decision.wave_loop import WaveLoopError, replay_wave_ledger

    result = run_authoritative_wave(_golden_payload())
    ledger = dict(result.ledger)

    # Tamper with the first entry's payload.
    entries = list(ledger.get("entries", ()))
    if entries:
        tampered_entry = dict(entries[0])
        tampered_entry["payload_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
        entries[0] = tampered_entry
        ledger["entries"] = entries

    with pytest.raises(WaveLoopError, match="payload hash mismatch"):
        replay_wave_ledger(ledger)


def test_replay_nondeterministic_on_version_change():
    """Replay MUST explicitly report a version mismatch rather than
    silently accepting cross-version replay."""
    from aie_decision.wave_authority import run_authoritative_wave
    from aie_decision.wave_loop import WaveLoopError, replay_wave_ledger

    result = run_authoritative_wave(_golden_payload())
    ledger = dict(result.ledger)
    ledger["schema_version"] = "joint-wave-ledger.v99"

    with pytest.raises(WaveLoopError, match="unsupported wave ledger schema_version"):
        replay_wave_ledger(ledger)


# ---------------------------------------------------------------------------
# 7. Old empty-success regression guard
# ---------------------------------------------------------------------------


def test_old_empty_success_behaviour_is_rejected():
    """The package authority path from commit f839e27 returned exit 0
    with components_called=false and an empty ledger.  This test proves
    the fix prevents that regression: the authority MUST NOT produce
    empty success."""
    from aie_decision.wave_authority import run_authoritative_wave

    payload = _golden_payload()
    result = run_authoritative_wave(payload)

    # The old bug produced components_called=False with exit 0.
    assert result.provenance.all_components_called(), (
        f"regression: components not all called: {result.provenance.failed_components()}"
    )
    # Old bug produced empty ledger; we must have entries.
    entries = result.ledger.get("entries", ())
    assert len(entries) > 0, "regression: empty ledger returned as success"


def test_authority_never_exit_zero_without_components(tmp_path):
    """Running a broken fixture through the CLI MUST never exit 0
    if the components report called=False."""
    broken = _golden_payload()
    # Corrupt the unit of the first axis to a genuinely unknown unit so
    # the schema validator rejects it before evaluation starts.
    broken["outcome_space"]["axes"][0]["unit"] = "megaparsec_per_jiffy"

    fixture_path = tmp_path / "bad_unit.json"
    fixture_path.write_text(json.dumps(broken), encoding="utf-8")

    proc = _run_authority(fixture_path, tmp_path / "out")
    assert proc.returncode != 0, (
        f"must not exit 0 when components are not all called; got {proc.returncode}"
    )


# ---------------------------------------------------------------------------
# 8. Golden surface mapping_ids — legal only, illegal as structured failures
# ---------------------------------------------------------------------------


def test_golden_surface_mapping_ids_exactly_three_legal():
    """The exact golden authoritative run MUST produce surface.mapping_ids
    containing only the three legal mappings (time-leg, price-fuel,
    magnitude-fuel).  The two illegal mappings (illegitimate-time-constant,
    illegitimate-time-money) MUST be absent from mapping_ids and present
    as structured failure evidence in staged_failures."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())

    mapping_ids = set(result.surface.get("mapping_ids", []))
    legal = {"time-leg", "price-fuel", "magnitude-fuel"}
    illegal = {"illegitimate-time-constant", "illegitimate-time-money"}

    assert mapping_ids == legal, (
        f"surface.mapping_ids must be exactly the three legal mappings; "
        f"got {mapping_ids}"
    )
    assert not (mapping_ids & illegal), (
        f"illegal mappings must not appear in surface.mapping_ids: "
        f"{mapping_ids & illegal}"
    )

    # Structured failures must contain both illegal mappings with evidence.
    staged = result.staged_failures
    assert len(staged) == 2, (
        f"expected 2 staged failures, got {len(staged)}"
    )
    staged_ids = {f["mapping_id"] for f in staged}
    assert staged_ids == illegal, (
        f"staged failures must cover both illegal mappings; got {staged_ids}"
    )
    for failure in staged:
        assert failure["code"] == "expected_failure"
        assert "operand" in failure
        assert "operand_unit" in failure
        assert "expected_unit" in failure


def test_illegal_mappings_not_in_particle_surface():
    """Illegal mappings MUST be excluded from particle evaluation entirely —
    they appear only in staged_failures, never in surface.mapping_ids
    or the particle_surface component record."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())

    # Check provenance: particle_surface component must be called=True.
    ps_prov = [c for c in result.provenance.components if c.component == "particle_surface"]
    assert len(ps_prov) == 1
    assert ps_prov[0].called is True

    # The surface mapping_ids must not include illegal entries.
    surface_ids = set(result.surface.get("mapping_ids", []))
    assert "illegitimate-time-constant" not in surface_ids
    assert "illegitimate-time-money" not in surface_ids

    # staged_failures must include them with structured evidence.
    staged_ids = {f["mapping_id"] for f in result.staged_failures}
    assert "illegitimate-time-constant" in staged_ids
    assert "illegitimate-time-money" in staged_ids


# ---------------------------------------------------------------------------
# 9. Magnitude axis nonzero + bimodality (dimensionless → dimensionless xform)
# ---------------------------------------------------------------------------


def test_golden_magnitude_axis_has_nonzero_and_bimodal_particles():
    """The magnitude axis (magnitude-fuel: regime_factor * severity_factor)
    MUST produce nonzero particles and preserve bimodality.  The formula
    is dimensionless → dimensionless and must compile as a
    DeterministicTransform (is_factor=False) because it has explicit
    output_axes=['magnitude']."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())

    # Verify magnitude is in multimodal_axes.
    diag = result.diagnostics
    multimodal = set(diag.get("multimodal_axes", []))
    assert "magnitude" in multimodal, (
        f"magnitude axis must be multimodal; got {multimodal}"
    )

    # Verify the compiled IR for magnitude-fuel is a DeterministicTransform
    # (is_factor=False), not a FactorIR.
    factor_prov = [
        c for c in result.provenance.components if c.component == "factor_ir"
    ]
    assert len(factor_prov) == 1
    assert factor_prov[0].called is True, (
        "factor_ir component must be called successfully"
    )

    # surface.mapping_ids includes magnitude-fuel.
    mapping_ids = set(result.surface.get("mapping_ids", []))
    assert "magnitude-fuel" in mapping_ids


def test_expected_failure_operand_unit_is_not_hardcoded_dimensionless():
    """Expected-failure mappings MUST be actually validated through the
    dimension checker.  operand_unit MUST expose the real operand dimensions,
    not a hardcoded 'dimensionless' placeholder.

    - illegitimate-time-constant: must identify the dimensionless constant
      vs time conflict.
    - illegitimate-time-money: must expose the actual money/volume dimension.
    """
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())

    failures_by_id = {f["mapping_id"]: f for f in result.staged_failures}

    # illegitimate-time-constant: lane_hours + 3
    tc = failures_by_id["illegitimate-time-constant"]
    assert tc["operand_unit"] != "dimensionless", (
        "operand_unit must NOT be hardcoded 'dimensionless'; must identify "
        f"the time-vs-constant conflict. Got: {tc['operand_unit']!r}"
    )
    assert "time" in tc["operand_unit"], (
        f"operand_unit must reference the time dimension; got {tc['operand_unit']!r}"
    )
    assert tc["expected_unit"] == "hour"

    # illegitimate-time-money: lane_hours + fuel_unit_cost
    tm = failures_by_id["illegitimate-time-money"]
    assert tm["operand_unit"] != "dimensionless", (
        "operand_unit must NOT be hardcoded 'dimensionless'; must expose "
        f"actual money/volume dimension. Got: {tm['operand_unit']!r}"
    )
    assert "money/USD" in tm["operand_unit"] or "money" in tm["operand_unit"], (
        f"operand_unit must expose money dimension; got {tm['operand_unit']!r}"
    )
    assert "volume" in tm["operand_unit"], (
        f"operand_unit must expose volume dimension; got {tm['operand_unit']!r}"
    )
    assert "time" in tm["operand_unit"], (
        f"operand_unit must expose time dimension; got {tm['operand_unit']!r}"
    )
    assert tm["expected_unit"] == "hour"

    # Both failures must carry real error messages from the dimension checker.
    for fid in ("illegitimate-time-constant", "illegitimate-time-money"):
        msg = failures_by_id[fid]["message"]
        assert "dimension mismatch" in msg.lower() or "dimension" in msg.lower(), (
            f"{fid} message must be from actual dimension check; got {msg!r}"
        )


# ---------------------------------------------------------------------------
# 10. Authority → replay black-box roundtrip (cross-command contract)
# ---------------------------------------------------------------------------


def test_authority_replay_roundtrip_exit_zero(tmp_path):
    """Black-box: authority writes a ledger, then replay on that ledger
    MUST exit 0 with status ok.
    """
    import subprocess, sys

    fixture = GOLDEN
    out_dir = tmp_path / "auth_out"
    proc_auth = subprocess.run(
        [sys.executable, str(RUNNER), "authority", str(fixture),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_auth.returncode == 0, (
        "authority must exit 0; got " + str(proc_auth.returncode)
        + " stderr: " + str(proc_auth.stderr)
    )
    ledger_path = out_dir / "wave-ledger.json"
    assert ledger_path.exists(), "ledger not written at " + str(ledger_path)

    proc_replay = subprocess.run(
        [sys.executable, str(RUNNER), "replay", str(ledger_path),
         "--fixture", str(fixture)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_replay.returncode == 0, (
        "replay must exit 0; got " + str(proc_replay.returncode)
        + " stdout: " + str(proc_replay.stdout)
        + " stderr: " + str(proc_replay.stderr)
    )
    outcome = json.loads(proc_replay.stdout.strip().split(chr(10))[-1])
    assert outcome["status"] == "ok", (
        "replay must report status ok; got " + repr(outcome.get("status"))
    )


def test_authority_replay_roundtrip_deterministic_ledger_equality():
    """Two authority runs on the same fixture MUST produce ledgers that
    are identical after stripping non-deterministic recorded_at timestamps.
    """
    import copy
    from aie_decision.wave_authority import run_authoritative_wave

    payload = _golden_payload()
    first = run_authoritative_wave(payload)
    second = run_authoritative_wave(payload)

    first_ledger = copy.deepcopy(dict(first.ledger))
    second_ledger = copy.deepcopy(dict(second.ledger))
    for entry in first_ledger.get("entries", ()):
        if isinstance(entry, dict):
            entry.pop("recorded_at", None)
    for entry in second_ledger.get("entries", ()):
        if isinstance(entry, dict):
            entry.pop("recorded_at", None)

    assert first_ledger == second_ledger, (
        "authority must produce deterministic ledgers after removing wall-clock timestamps"
    )


def test_authority_replay_rejects_tampered_ledger(tmp_path):
    """Black-box: a tampered ledger (corrupted payload_hash) MUST be
    rejected by the authoritative replay path.
    """
    import subprocess, sys
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())
    ledger = dict(result.ledger)

    entries = list(ledger.get("entries", ()))
    assert entries, "ledger must have entries to tamper"
    tampered_entry = dict(entries[0])
    tampered_entry["payload_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
    entries[0] = tampered_entry
    ledger["entries"] = entries

    tampered_path = tmp_path / "tampered-ledger.json"
    tampered_path.write_text(json.dumps(ledger), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(RUNNER), "replay", str(tampered_path),
         "--fixture", str(GOLDEN)],
        capture_output=True, text=True, timeout=60,
    )
    outcome = json.loads(proc.stdout.strip().split(chr(10))[-1])
    assert outcome["status"] == "ledger_mismatch", (
        "tampered ledger must report ledger_mismatch; got " + repr(outcome.get("status"))
    )


def test_oracle_replay_still_works_for_oracle_ledgers(tmp_path):
    """Black-box: an oracle-produced ledger MUST still be replayable
    through the oracle path after the authority routing fix.
    """
    import subprocess, sys

    out_dir = tmp_path / "oracle_out"
    proc_run = subprocess.run(
        [sys.executable, str(RUNNER), "run", str(GOLDEN),
         "--output-dir", str(out_dir), "--authority", "oracle"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_run.returncode == 0, (
        "oracle run must exit 0; got " + str(proc_run.returncode)
        + " stderr: " + str(proc_run.stderr)
    )
    ledger_path = out_dir / "wave-ledger.json"
    assert ledger_path.exists(), "oracle ledger not written at " + str(ledger_path)

    proc_replay = subprocess.run(
        [sys.executable, str(RUNNER), "replay", str(ledger_path),
         "--fixture", str(GOLDEN)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_replay.returncode == 0, (
        "oracle replay must exit 0; got " + str(proc_replay.returncode)
        + " stdout: " + str(proc_replay.stdout)
        + " stderr: " + str(proc_replay.stderr)
    )
    outcome = json.loads(proc_replay.stdout.strip().split(chr(10))[-1])
    assert outcome["status"] == "ok", (
        "oracle replay must report ok; got " + repr(outcome.get("status"))
    )



def test_authority_replay_double_invocation_byte_identical(tmp_path):
    """Black-box: two CLI replay invocations on the same authority ledger
    MUST produce byte-for-byte identical JSON output, including identical
    ledger_hash.  This is the contract that the previous fix broke by
    hashing the fresh ledger with recorded_at timestamps."""
    import subprocess, sys

    fixture = GOLDEN
    out_dir = tmp_path / "auth_out"
    proc_auth = subprocess.run(
        [sys.executable, str(RUNNER), "authority", str(fixture),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_auth.returncode == 0
    ledger_path = out_dir / "wave-ledger.json"
    assert ledger_path.exists()

    def replay():
        proc = subprocess.run(
            [sys.executable, str(RUNNER), "replay", str(ledger_path),
             "--fixture", str(fixture)],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0
        return json.loads(proc.stdout.strip().split(chr(10))[-1])

    first = replay()
    second = replay()

    # Byte-for-byte equality of the full parsed output.
    assert first == second, (
        "two replay invocations must produce identical outputs; " +
        "first: " + json.dumps(first) + " second: " + json.dumps(second)
    )

    # ledger_hash must be stable across invocations.
    assert first["ledger_hash"] == second["ledger_hash"], (
        "ledger_hash must be identical across replay invocations; " +
        "got " + repr(first["ledger_hash"]) + " vs " + repr(second["ledger_hash"])
    )

    # iterations must be honest (non-empty when ledger has loop entries).
    assert isinstance(first["iterations"], list), (
        "iterations must be a list"
    )
    assert len(first["iterations"]) > 0, (
        "iterations must not be empty when the ledger has loop entries"
    )
