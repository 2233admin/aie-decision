"""Black-box tests for lite session cache-only contract (OpenSpec 1.6).

These tests verify that ``session_start`` / ``session_end`` write ONLY to
``.sentrux/cache/lite-baseline.json`` and session evidence, and NEVER
alter the native baseline bytes or SHA-256.

Requirements derived from:
  openspec/changes/add-joint-wave-surface-mapping/tasks.md (task 1.6)
  openspec/changes/add-joint-wave-surface-mapping/design.md
  docs/CODE-INTEL-TOOLCHAIN-MIGRATION.md
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTRUX_DIR = REPO_ROOT / ".sentrux"
NATIVE_BASELINE = SENTRUX_DIR / "baseline.json"
CACHE_DIR = SENTRUX_DIR / "cache"
LITE_BASELINE = CACHE_DIR / "lite-baseline.json"
SESSION_DIR = SENTRUX_DIR / "agent-sessions"

_BETA5_LEGACY = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "code-intel"
    / "releases"
    / "v0.7.0-beta.5"
    / "legacy"
)
_AGENT_TOOL = _BETA5_LEGACY / "Invoke-SentruxAgentTool.ps1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_session_start(session_id: str) -> dict:
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_start", str(REPO_ROOT),
            "-SessionId", session_id,
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
    )
    assert proc.stdout, f"session_start produced no stdout (exit {proc.returncode})"
    result = json.loads(proc.stdout)
    result["_exit_code"] = proc.returncode
    return result


def _run_session_end(session_id: str) -> dict:
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_end", str(REPO_ROOT),
            "-SessionId", session_id,
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
    )
    assert proc.stdout, f"session_end produced no stdout (exit {proc.returncode})"
    result = json.loads(proc.stdout)
    result["_exit_code"] = proc.returncode
    return result


def _cleanup_session(session_id: str) -> None:
    for pattern in [f"{session_id}.start.json", f"{session_id}.end.json"]:
        for path in SESSION_DIR.glob(pattern):
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def test_prerequisites():
    """Verify the canonical toolchain files exist before any session test."""
    assert _AGENT_TOOL.is_file(), (
        f"Invoke-SentruxAgentTool.ps1 not found at {_AGENT_TOOL}"
    )
    assert NATIVE_BASELINE.is_file(), (
        f"Native baseline not found at {NATIVE_BASELINE}"
    )


# ---------------------------------------------------------------------------
# Core contract: native baseline MUST NOT be altered
# ---------------------------------------------------------------------------


def test_session_start_does_not_alter_native_baseline_hash():
    """session_start MUST leave the native baseline bytes unchanged.

    Requirement (OpenSpec 1.6):
      The lite session cache-only contract requires session_start to write
      only to ``.sentrux/cache/lite-baseline.json`` and session evidence,
      and NEVER alter the native baseline bytes or SHA-256.
    """
    baseline_data = json.loads(NATIVE_BASELINE.read_text())
    assert baseline_data.get("schema") == "code-intel-sentrux-baseline.v4", (
        f"native baseline pre-condition: expected v4 schema, "
        f"got {baseline_data.get('schema')}"
    )
    assert baseline_data.get("engine", {}).get("id") == "sentrux-native", (
        "native baseline pre-condition: expected sentrux-native engine"
    )

    hash_before = _sha256(NATIVE_BASELINE)
    bytes_before = NATIVE_BASELINE.read_bytes()

    for pattern in ["test-lite-*.start.json", "test-lite-*.end.json"]:
        for stale in SESSION_DIR.glob(pattern):
            stale.unlink()

    session_id = f"test-lite-{hash_before[:12]}"
    _run_session_start(session_id)

    hash_after = _sha256(NATIVE_BASELINE)
    bytes_after = NATIVE_BASELINE.read_bytes()

    assert hash_after == hash_before, (
        f"session_start ALTERED the native baseline!\n"
        f"  SHA-256 before: {hash_before}\n"
        f"  SHA-256 after:  {hash_after}\n"
        f"  Byte length before: {len(bytes_before)}\n"
        f"  Byte length after:  {len(bytes_after)}"
    )

    assert LITE_BASELINE.is_file(), (
        f"session_start did not write lite baseline to {LITE_BASELINE}"
    )
    lite_data = json.loads(LITE_BASELINE.read_text())
    assert "tool" in lite_data or "engine" in lite_data, (
        f"lite cache baseline has no tool/engine identifier"
    )

    start_files = list(SESSION_DIR.glob(f"{session_id}.start.json"))
    assert len(start_files) == 1, "expected one session start evidence file"

    _cleanup_session(session_id)


def test_session_end_does_not_alter_native_baseline_hash():
    """session_end MUST leave the native baseline bytes unchanged."""
    hash_before = _sha256(NATIVE_BASELINE)
    session_id = f"test-lite-end-{hash_before[:12]}"
    _run_session_start(session_id)

    hash_after_start = _sha256(NATIVE_BASELINE)
    bytes_after_start = NATIVE_BASELINE.read_bytes()

    _run_session_end(session_id)

    hash_after_end = _sha256(NATIVE_BASELINE)
    bytes_after_end = NATIVE_BASELINE.read_bytes()

    assert hash_after_end == hash_after_start, (
        f"session_end ALTERED the native baseline after start!\n"
        f"  SHA-256 after start: {hash_after_start}\n"
        f"  SHA-256 after end:   {hash_after_end}\n"
        f"  Byte length after start: {len(bytes_after_start)}\n"
        f"  Byte length after end:   {len(bytes_after_end)}"
    )

    _cleanup_session(session_id)


# ---------------------------------------------------------------------------
# Cache artifact assertions
# ---------------------------------------------------------------------------


def test_lite_cache_baseline_has_declared_schema_and_engine():
    """The lite cache baseline MUST declare its schema/engine identity."""
    if not LITE_BASELINE.is_file():
        pytest.skip("lite cache baseline does not exist yet")
    data = json.loads(LITE_BASELINE.read_text())
    has_tool = "tool" in data
    has_engine = "engine" in data
    assert has_tool or has_engine, (
        "lite cache baseline has neither 'tool' nor 'engine' field"
    )
    if has_engine:
        assert data["engine"].get("id") != "sentrux-native", (
            "lite cache baseline MUST NOT claim sentrux-native engine"
        )
    if "schema" in data:
        assert data["schema"] != "code-intel-sentrux-baseline.v4", (
            "lite cache baseline MUST NOT claim native v4 schema"
        )


# ---------------------------------------------------------------------------
# Adversarial: failure must exit non-zero
# ---------------------------------------------------------------------------


def test_session_start_with_nonexistent_path_fails_nonzero():
    """session_start on a non-existent directory MUST exit non-zero."""
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_start",
            str(REPO_ROOT / "_nonexistent_dir_"),
            "-SessionId", "test-nonexistent",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, (
        f"session_start on non-existent path should fail non-zero, "
        f"got exit {proc.returncode}"
    )
