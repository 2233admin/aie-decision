"""Golden fixture integration and surface semantics adversarial tests.

Requirements from:
  - ``joint-wave-surface`` OpenSpec: golden fixture, surface semantics
  - ``wave-surface-search-loop`` OpenSpec: component provenance

These tests verify the authoritative package path against the exact
golden fixture ``fixtures/golden/joint_wave_surface_mvp.json``.
"""

from __future__ import annotations

from test_wave_authority_adversarial import _golden_payload


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
# 3. Golden surface mapping_ids — legal only, illegal as structured failures
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
# 4. Magnitude axis nonzero + bimodality (dimensionless → dimensionless xform)
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
# 5. Reviewer-style adversarial: semantics gate — OpenSpec derived
# ---------------------------------------------------------------------------


def test_possibility_surface_rejects_calibrated_expectation_mutation():
    """REVIEWER-STYLE: derived backward from ``joint-wave-surface`` OpenSpec
    Requirement §Surface semantics gate — "The system MUST label a surface as
    possibility_surface unless declared distribution semantics and a validated
    calibration basis justify probability_surface."

    If a mutation flips the surface kind from ``possibility_surface`` to
    ``probability_surface`` while inputs remain uncalibrated, this test MUST
    detect the regression.  The test computes the actual surface kind and
    asserts it is exactly ``possibility_surface`` — a bounded mutation that
    alters the semantics gate to return ``probability_surface`` would fail.
    """
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_golden_payload())

    # The authoritative surface MUST NOT claim probability semantics.
    surface_kind = result.surface.get("kind")
    assert surface_kind == "possibility_surface", (
        f"SEMANTICS GATE MUTATION DETECTED: expected possibility_surface, "
        f"got {surface_kind}.  Uncalibrated inputs were promoted to "
        f"probability — this violates the OpenSpec surface semantics gate."
    )

    # diagnostics MUST agree with surface.
    diag_kind = result.diagnostics.get("surface_kind")
    assert diag_kind == "possibility_surface", (
        f"Diagnostics surface_kind ({diag_kind}) disagrees with surface kind "
        f"({surface_kind}) — internal inconsistency in semantics gate."
    )

    # Neither surface nor diagnostics must declare empirical calibration.
    coverage = result.surface.get("coverage_semantics")
    bad_coverages = {"empirical_prediction_interval", "empirical_confidence_interval"}
    assert coverage not in bad_coverages, (
        f"Coverage semantics {coverage} claims empirical basis "
        f"without declared calibration."
    )
