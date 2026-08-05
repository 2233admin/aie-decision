"""Black-box gate evidence for lite session cache-only contract (OpenSpec 1.6).

These tests run session_start / session_end in a **temporary directory**
(pytest ``tmp_path``), NEVER against the real repository checkout.  The
real ``.sentrux/baseline.json`` MUST be bit-for-bit identical before and
after this test module runs.

Requirements derived from:
  openspec/changes/add-joint-wave-surface-mapping/tasks.md (task 1.6)
  openspec/changes/add-joint-wave-surface-mapping/design.md
  docs/CODE-INTEL-TOOLCHAIN-MIGRATION.md
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_BASELINE = REPO_ROOT / ".sentrux" / "baseline.json"

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
    """Return the host environment with NO code-intel overrides."""
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("CODE_INTEL_"):
            env.pop(key, None)
    return env


def _build_temp_repo(tmp: Path) -> tuple[Path, str]:
    """Build a minimal temp repo with a native baseline for session testing.

    Returns (temp_repo, baseline_sha256_before).
    """
    repo = tmp / "test-repo"
    repo.mkdir(parents=True)
    sentrux_dir = repo / ".sentrux"
    sentrux_dir.mkdir()

    # Copy the native v4 baseline from the real checkout into the temp repo.
    shutil.copy2(str(REAL_BASELINE), str(sentrux_dir / "baseline.json"))

    # Create a minimal rules.toml (required by the tool).
    (sentrux_dir / "rules.toml").write_text(
        "[policy]\nmax_cc = 25\nno_god_files = false\n"
    )

    # Create a minimal .git directory (required by ExplicitOverlay policy).
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    hash_before = _sha256(sentrux_dir / "baseline.json")
    return repo, hash_before


def _run_session_start(repo: Path, session_id: str) -> dict:
    """Run session_start in a temp repo (NOT the real checkout)."""
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_start", str(repo),
            "-SessionId", session_id,
        ],
        capture_output=True, text=True, cwd=str(repo), timeout=60,
        env=_ambient_env(),
    )
    assert proc.stdout, f"session_start produced no stdout (exit {proc.returncode})"
    result = json.loads(proc.stdout)
    result["_exit_code"] = proc.returncode
    return result


def _run_session_end(repo: Path, session_id: str) -> dict:
    """Run session_end in a temp repo (NOT the real checkout)."""
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_end", str(repo),
            "-SessionId", session_id,
        ],
        capture_output=True, text=True, cwd=str(repo), timeout=60,
        env=_ambient_env(),
    )
    assert proc.stdout, f"session_end produced no stdout (exit {proc.returncode})"
    result = json.loads(proc.stdout)
    result["_exit_code"] = proc.returncode
    return result


# ---------------------------------------------------------------------------
# Prerequisites (read-only, against real checkout)
# ---------------------------------------------------------------------------


def test_prerequisites():
    """Verify the canonical toolchain files and native baseline exist."""
    assert _AGENT_TOOL.is_file(), (
        f"Invoke-SentruxAgentTool.ps1 not found at {_AGENT_TOOL}"
    )
    assert REAL_BASELINE.is_file(), (
        f"Native baseline not found at {REAL_BASELINE}"
    )


def test_native_baseline_schema_is_v4():
    """The real native baseline MUST have schema v4 (read-only check).
    Upgrade to v5 is blocked until a clean-tree anchor with non-degraded
    metrics can be established."""
    baseline_data = json.loads(REAL_BASELINE.read_text())
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
# Gate evidence: session_start alters the baseline (in TEMP repo)
# ---------------------------------------------------------------------------


def test_session_start_overwrites_baseline_in_temp_repo(tmp_path):
    """session_start overwrites the baseline in a temp repo — 1.6 is BLOCKED.

    OpenSpec 1.6 mandatory conformance case:
      A lite session MUST write only to ``.sentrux/cache/lite-baseline.json``
      and MUST NOT alter the native baseline bytes or SHA-256.

    This test runs session_start in a **temporary directory** (pytest
    ``tmp_path``), never against the real checkout.  The real
    ``.sentrux/baseline.json`` is never read or written by this test
    beyond the prerequisite check.

    Current status (2026-08-05): **BLOCKED** — canonical beta.5
    ``Invoke-SentruxAgentTool.ps1`` calls ``sentrux gate --save`` which
    hard-codes ``.sentrux/baseline.json`` as the output path.
    """
    repo, hash_before = _build_temp_repo(tmp_path)
    baseline = repo / ".sentrux" / "baseline.json"
    bytes_before = baseline.read_bytes()
    session_id = "test-lite-tmp"

    try:
        _run_session_start(repo, session_id)

        hash_after = _sha256(baseline)
        bytes_after = baseline.read_bytes()

        # === Mandatory conformance assertion ===
        # FAILS because canonical tool overwrites .sentrux/baseline.json.
        assert hash_after == hash_before, (
            f"BLOCKED: session_start ALTERED the baseline in temp repo!\n"
            f"  Upstream gap: Invoke-SentruxAgentTool.ps1 calls sentrux gate\n"
            f"  --save which hard-codes .sentrux/baseline.json.\n"
            f"  SHA-256 before: {hash_before}\n"
            f"  SHA-256 after:  {hash_after}\n"
            f"  Byte length before: {len(bytes_before)}\n"
            f"  Byte length after:  {len(bytes_after)}"
        )
    finally:
        # Clean up all session/cache artifacts in the temp repo.
        _cleanup_temp_artifacts(repo, session_id)


def test_session_end_overwrites_baseline_in_temp_repo(tmp_path):
    """session_end also overwrites the baseline in a temp repo.

    Confirms the hard-coded path applies to both session boundaries.
    """
    repo, hash_before = _build_temp_repo(tmp_path)
    baseline = repo / ".sentrux" / "baseline.json"
    session_id = "test-lite-tmp-end"

    try:
        _run_session_start(repo, session_id)
        hash_after_start = _sha256(baseline)
        bytes_after_start = baseline.read_bytes()

        _run_session_end(repo, session_id)
        hash_after_end = _sha256(baseline)
        bytes_after_end = baseline.read_bytes()

        assert hash_after_end == hash_after_start, (
            f"BLOCKED: session_end ALTERED the baseline after start!\n"
            f"  SHA-256 after start: {hash_after_start}\n"
            f"  SHA-256 after end:   {hash_after_end}\n"
            f"  Byte length after start: {len(bytes_after_start)}\n"
            f"  Byte length after end:   {len(bytes_after_end)}"
        )
    finally:
        _cleanup_temp_artifacts(repo, session_id)


# ---------------------------------------------------------------------------
# Adversarial: failure must exit non-zero
# ---------------------------------------------------------------------------


def test_session_start_with_nonexistent_path_fails_nonzero(tmp_path):
    """session_start on a non-existent directory MUST exit non-zero."""
    nonexistent = tmp_path / "_nonexistent_dir_"
    proc = subprocess.run(
        [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(_AGENT_TOOL),
            "session_start",
            str(nonexistent),
            "-SessionId", "test-nonexistent",
        ],
        capture_output=True, text=True, timeout=30,
        env=_ambient_env(),
    )
    assert proc.returncode != 0, (
        f"session_start on non-existent path should fail non-zero, "
        f"got exit {proc.returncode}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup_temp_artifacts(repo: Path, session_id: str) -> None:
    """Remove all session/cache artifacts from the temp repo.

    Uses try/except on each path so one failure doesn't block others.
    """
    sentrux_dir = repo / ".sentrux"

    # Session evidence files.
    for pattern in [f"{session_id}.start.json", f"{session_id}.end.json"]:
        sessions_dir = sentrux_dir / "agent-sessions"
        if sessions_dir.is_dir():
            for path in sessions_dir.glob(pattern):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    # Cache directory (lite-baseline.json).
    cache_dir = sentrux_dir / "cache"
    if cache_dir.is_dir():
        try:
            shutil.rmtree(str(cache_dir), ignore_errors=True)
        except OSError:
            pass

