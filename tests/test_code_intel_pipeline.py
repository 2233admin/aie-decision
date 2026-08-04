"""Black-box tests for code-intel pipeline integration (OpenSpec 1.7).

These tests verify that ``code-intel . --mode normal --json`` passes with
exit code 0, no domain failures, and no process failures, using only
canonical Code Intel beta.5 supported discovery mechanisms.

Requirements derived from:
  openspec/changes/add-joint-wave-surface-mapping/tasks.md (task 1.7)
  openspec/changes/add-joint-wave-surface-mapping/design.md
  docs/CODE-INTEL-TOOLCHAIN-MIGRATION.md
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical beta.5 release directory containing the orchestration manifest
# and pipeline source tree that doctor bootstrap and orchestrate Validate
# check entrypoints against.
_BETA5_RELEASE = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "code-intel"
    / "releases"
    / "v0.7.0-beta.5"
)
_MANIFEST_PATH = _BETA5_RELEASE / "orchestration" / "integrations.json"

# The global CODE_INTEL_HOME env var on this host points to a non-existent
# development directory.  When set *and* the directory is absent, doctor
# bootstrap fails closed.  The test unsets it so the default derivation
# (<pipeline_root>) is used instead.
_STALE_CODE_INTEL_HOME = os.environ.get("CODE_INTEL_HOME", "")


def _code_intel_env() -> dict[str, str]:
    """Return a clean environment for code-intel default-mode invocation."""
    env = os.environ.copy()
    env["CODE_INTEL_INTEGRATIONS_MANIFEST"] = str(_MANIFEST_PATH)
    # Unset the stale CODE_INTEL_HOME so doctor bootstrap does not reject
    # it as a missing directory.
    env.pop("CODE_INTEL_HOME", None)
    return env


def _run_pipeline() -> subprocess.CompletedProcess:
    """Run the default pipeline and return the completed process."""
    return subprocess.run(
        ["code-intel", ".", "--mode", "normal", "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_code_intel_env(),
        timeout=120,
    )


def _parse_result(proc: subprocess.CompletedProcess) -> dict:
    """Parse the JSON result, failing the test if it is not valid JSON."""
    assert proc.stdout, "pipeline produced no stdout"
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"pipeline stdout is not valid JSON: {exc}\n{proc.stdout}")


# ---------------------------------------------------------------------------
# Positive: the default command passes
# ---------------------------------------------------------------------------


def test_pipeline_default_mode_passes():
    """``code-intel . --mode normal --json`` exits 0 with no failures.

    Requirement (OpenSpec 1.7):
      The default pipeline command MUST complete with outcome "completed",
      exit code 0, and both ``failures.domain`` and ``failures.process``
      empty.  A ``domain_failed`` or ``process_failed`` outcome with a
      non-null diagnostic is a gate failure.
    """
    proc = _run_pipeline()
    result = _parse_result(proc)

    assert proc.returncode == 0, (
        f"expected exit 0, got {proc.returncode}\n"
        f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    )
    assert result["outcome"] == "completed", (
        f"expected outcome 'completed', got {result['outcome']!r}\n"
        f"diagnostic: {result.get('diagnostic')}"
    )
    assert result["failureNode"] is None, (
        f"unexpected failure node: {result['failureNode']}"
    )
    assert result["failures"]["domain"] == [], (
        f"domain failures: {result['failures']['domain']}"
    )
    assert result["failures"]["process"] == [], (
        f"process failures: {result['failures']['process']}"
    )


# ---------------------------------------------------------------------------
# Negative / adversarial: without the manifest env var the pipeline MUST
# fail closed, not silently report success.
# ---------------------------------------------------------------------------


def test_pipeline_without_manifest_env_var_fails_closed():
    """Without CODE_INTEL_INTEGRATIONS_MANIFEST, the pipeline must fail.

    The fallback discovery (ancestor walk from the exe parent) resolves the
    pipeline root to ``bin/``, which does not contain the source files the
    doctor and orchestrate Validate expect.  The pipeline MUST report a
    failure — never a silent "completed".

    Adversarial case derived from the OpenSpec requirement that missing or
    misconfigured tooling must fail closed.
    """
    env = os.environ.copy()
    env.pop("CODE_INTEL_HOME", None)
    # Deliberately omit CODE_INTEL_INTEGRATIONS_MANIFEST.
    proc = subprocess.run(
        ["code-intel", ".", "--mode", "normal", "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    result = _parse_result(proc)

    assert proc.returncode != 0, (
        f"expected non-zero exit without manifest env var, got {proc.returncode}"
    )
    assert result["outcome"] != "completed", (
        f"expected non-completed outcome without manifest env var, "
        f"got {result['outcome']!r}"
    )


# ---------------------------------------------------------------------------
# Stale CODE_INTEL_HOME must also fail closed
# ---------------------------------------------------------------------------


def test_pipeline_with_stale_code_intel_home_fails_closed():
    """When CODE_INTEL_HOME points to a non-existent directory, the doctor
    bootstrap MUST report it as missing and fail the run."""
    if not _STALE_CODE_INTEL_HOME:
        pytest.skip("CODE_INTEL_HOME is not set — nothing to test")
    env = os.environ.copy()
    env["CODE_INTEL_INTEGRATIONS_MANIFEST"] = str(_MANIFEST_PATH)
    # Keep the stale CODE_INTEL_HOME (do NOT pop it).
    proc = subprocess.run(
        ["code-intel", ".", "--mode", "normal", "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    result = _parse_result(proc)

    assert proc.returncode != 0, (
        f"expected non-zero exit with stale CODE_INTEL_HOME, got {proc.returncode}"
    )
    assert result["outcome"] != "completed", (
        f"expected non-completed outcome with stale CODE_INTEL_HOME, "
        f"got {result['outcome']!r}"
    )


# ---------------------------------------------------------------------------
# Artifact publication exists and is non-empty
# ---------------------------------------------------------------------------


def test_pipeline_publication_marker_exists():
    """A successful run publishes a run-complete.json marker."""
    proc = _run_pipeline()
    result = _parse_result(proc)
    if result["outcome"] != "completed":
        pytest.skip("pipeline did not complete — publication skipped")

    marker = Path(result["publication"]["marker"])
    assert marker.is_file(), f"publication marker not found: {marker}"
    raw = marker.read_bytes()
    assert raw, f"publication marker is empty: {marker}"
    # The marker must itself be valid JSON.
    json.loads(raw)
