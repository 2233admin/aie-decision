"""CLI fail-closed and regression guard adversarial tests.

Requirements from:
  - ``wave-surface-search-loop`` OpenSpec: loop termination and replay
  - Fail-closed: authoritative CLI exits nonzero on component failure
"""

from __future__ import annotations

import json

from test_wave_authority_adversarial import (
    _golden_payload,
    _run_authority,
)


# ---------------------------------------------------------------------------
# 1. Fail-closed: CLI exit codes
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
# 2. Old empty-success regression guard
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
