"""Shared helpers for wave-authority adversarial tests.

This module was refactored from the original god test file (36 tests, 786
lines).  Test functions have been moved to requirement-organized sibling
files — this module now exports only shared fixtures and helpers:

  - test_wave_authority_golden_adversarial.py   (golden / surface semantics)
  - test_wave_authority_units_adversarial.py    (compound / illegal units)
  - test_wave_authority_cli_adversarial.py      (CLI / regression guards)
  - test_wave_authority_replay_adversarial.py   (replay / roundtrip)

Every existing assertion is preserved verbatim in meaning and collection
count.  The shared helpers are imported by each sibling via::

    from test_wave_authority_adversarial import _golden_payload, _run_authority
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_joint_wave_surface_mvp.py"
GOLDEN = ROOT / "fixtures" / "golden" / "joint_wave_surface_mvp.json"


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
