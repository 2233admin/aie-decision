"""Black-box gate evidence for lite session cache-only contract (OpenSpec 1.6).

These tests probe the canonical beta.5 session tools WITHOUT any
CODE_INTEL_REPO_ROOT injection, no tools/sentrux-shim/ override, and
no environment manipulation.  They record the honest outcome as gate
evidence — a passing gate requires lite sessions to write only to
``.sentrux/cache/lite-baseline.json`` without altering the native baseline;
a blocked gate is reported as a test failure, never hidden behind skip/xfail.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTRUX_DIR = REPO_ROOT / ".sentrux"
NATIVE_BASELINE = SENTRUX_DIR / "baseline.json"
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


def _ambient_env() -> dict[str, str]:
    """Return the host environment with NO sentrux/code-intel overrides.

    No CODE_INTEL_REPO_ROOT, no CODE_INTEL_INTEGRATIONS_MANIFEST,
    no CODE_INTEL_HOME.  This is what a fresh PowerShell session sees.
    """
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("CODE_INTEL_"):
            env.pop(key, None)
    return env


def _run_session_start(session_id: str) -> dict:
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_start", str(REPO_ROOT),
            "-SessionId", session_id,
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        env=_ambient_env(),
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
        env=_ambient_env(),
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
    """Verify the canonical toolchain files and native baseline exist."""
    assert _AGENT_TOOL.is_file(), (
        f"Invoke-SentruxAgentTool.ps1 not found at {_AGENT_TOOL}"
    )
    assert NATIVE_BASELINE.is_file(), (
        f"Native baseline not found at {NATIVE_BASELINE}"
    )


def test_native_baseline_schema_is_v4():
    """The native baseline MUST have schema code-intel-sentrux-baseline.v4."""
    baseline_data = json.loads(NATIVE_BASELINE.read_text())
    assert baseline_data.get("schema") == "code-intel-sentrux-baseline.v4", (
        f"Native baseline schema mismatch: expected v4, "
        f"got {baseline_data.get('schema')}"
    )
    engine = baseline_data.get("engine", {})
    assert engine.get("id") == "sentrux-native", (
        f"Native baseline engine mismatch: expected sentrux-native, "
        f"got {engine.get('id')}"
    )


# ---------------------------------------------------------------------------
# Gate evidence: session_start alters the native baseline
# ---------------------------------------------------------------------------


def test_session_start_overwrites_native_baseline():
    """session_start overwrites the native baseline — 1.6 is BLOCKED.

    OpenSpec 1.6 mandatory conformance case:
      A lite session MUST write only to ``.sentrux/cache/lite-baseline.json``
      and MUST NOT alter the native baseline bytes or SHA-256.

    Current status (2026-08-05): **BLOCKED** — canonical beta.5
    ``Invoke-SentruxAgentTool.ps1`` calls ``sentrux gate --save`` which
    hard-codes ``.sentrux/baseline.json`` as the output path.  No repo-level
    extension surface exists to redirect this write.  The test below
    demonstrates the violation by running session_start and observing the
    native baseline change.

    This test is a fail-closed probe: it asserts the contract and FAILS
    when the contract is violated.  Do NOT hide the failure — the failure
    IS the gate evidence for the upstream blocker.
    """
    # Record the native baseline identity BEFORE session_start.
    hash_before = _sha256(NATIVE_BASELINE)
    bytes_before = NATIVE_BASELINE.read_bytes()

    # Clean up prior session artifacts.
    for pattern in ["test-lite-blocked-*.start.json", "test-lite-blocked-*.end.json"]:
        for stale in SESSION_DIR.glob(pattern):
            stale.unlink()

    session_id = f"test-lite-blocked-{hash_before[:12]}"
    _run_session_start(session_id)

    hash_after = _sha256(NATIVE_BASELINE)
    bytes_after = NATIVE_BASELINE.read_bytes()

    # === Mandatory conformance assertion ===
    # This asserts the native baseline MUST be unchanged.  It currently
    # FAILS because the canonical tool overwrites .sentrux/baseline.json.
    assert hash_after == hash_before, (
        f"BLOCKED: session_start ALTERED the native baseline!\n"
        f"  Upstream gap: Invoke-SentruxAgentTool.ps1 calls sentrux gate --save\n"
        f"  which hard-codes .sentrux/baseline.json.  No repo-level extension\n"
        f"  surface exists to redirect this write.\n"
        f"  SHA-256 before: {hash_before}\n"
        f"  SHA-256 after:  {hash_after}\n"
        f"  Byte length before: {len(bytes_before)}\n"
        f"  Byte length after:  {len(bytes_after)}"
    )

    _cleanup_session(session_id)


def test_session_end_overwrites_native_baseline():
    """session_end also overwrites the native baseline — 1.6 is BLOCKED.

    After session_start already changed the baseline, session_end writes
    again, confirming the hard-coded path in canonical beta.5 applies to
    both session boundaries.
    """
    hash_before = _sha256(NATIVE_BASELINE)
    session_id = f"test-lite-end-blocked-{hash_before[:12]}"
    _run_session_start(session_id)

    hash_after_start = _sha256(NATIVE_BASELINE)
    bytes_after_start = NATIVE_BASELINE.read_bytes()

    _run_session_end(session_id)

    hash_after_end = _sha256(NATIVE_BASELINE)
    bytes_after_end = NATIVE_BASELINE.read_bytes()

    assert hash_after_end == hash_after_start, (
        f"BLOCKED: session_end ALTERED the native baseline after start!\n"
        f"  Same upstream gap as session_start — sentrux gate --save\n"
        f"  hard-codes .sentrux/baseline.json.\n"
        f"  SHA-256 after start: {hash_after_start}\n"
        f"  SHA-256 after end:   {hash_after_end}\n"
        f"  Byte length after start: {len(bytes_after_start)}\n"
        f"  Byte length after end:   {len(bytes_after_end)}"
    )

    _cleanup_session(session_id)


# ---------------------------------------------------------------------------
# Adversarial: failure must exit non-zero (no tools/sentrux-shim needed)
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
        env=_ambient_env(),
    )
    assert proc.returncode != 0, (
        f"session_start on non-existent path should fail non-zero, "
        f"got exit {proc.returncode}"
    )
