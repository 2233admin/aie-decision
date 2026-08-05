"""Replay integrity and black-box roundtrip adversarial tests.

Requirements from:
  - ``wave-surface-search-loop`` OpenSpec: deterministic action replay
  - Loop termination: ledger immutability and replay identity
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_wave_authority_adversarial import (
    GOLDEN,
    RUNNER,
    _golden_payload,
)


# ---------------------------------------------------------------------------
# 1. Tampered replay detection
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
# 2. Authority → replay black-box roundtrip (cross-command contract)
# ---------------------------------------------------------------------------


def test_authority_replay_roundtrip_exit_zero(tmp_path):
    """Black-box: authority writes a ledger, then replay on that ledger
    MUST exit 0 with status ok.
    """
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
