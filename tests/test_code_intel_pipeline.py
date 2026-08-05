"""Black-box gate evidence for code-intel pipeline (OpenSpec 1.7).

These tests probe the ambient ``code-intel . --mode normal --json`` command
WITHOUT any environment injection (no CODE_INTEL_INTEGRATIONS_MANIFEST,
no CODE_INTEL_HOME, no PATH manipulation).  They record the honest outcome
as gate evidence — a passing gate requires the ambient command to succeed;
a blocked gate is reported as a test failure, never hidden behind skip/xfail
or env-injected passes.

Requirements derived from:
  openspec/changes/add-joint-wave-surface-mapping/tasks.md (task 1.7)
  openspec/changes/add-joint-wave-surface-mapping/design.md
  docs/CODE-INTEL-TOOLCHAIN-MIGRATION.md
"""

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ambient_env() -> dict[str, str]:
    """Return the host environment with NO code-intel overrides injected.

    This is the critical invariant: the test must use exactly what a fresh
    PowerShell session would see, with no CODE_INTEL_INTEGRATIONS_MANIFEST,
    CODE_INTEL_HOME unset, and no PATH augmentation.  If the pipeline cannot
    resolve its manifest from the ambient environment alone, 1.7 is blocked.
    """
    env = os.environ.copy()
    # Remove any code-intel-specific env vars that might have leaked in.
    for key in list(env.keys()):
        if key.startswith("CODE_INTEL_"):
            env.pop(key, None)
    return env


def _run_pipeline() -> subprocess.CompletedProcess:
    """Run the default pipeline with ambient environment and no injections."""
    return subprocess.run(
        ["code-intel", ".", "--mode", "normal", "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_ambient_env(),
        timeout=120,
    )


def _parse_result(proc: subprocess.CompletedProcess) -> dict:
    """Parse the JSON result, failing the test if it is not valid JSON."""
    assert proc.stdout, "pipeline produced no stdout"
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"pipeline stdout is not valid JSON: {exc}\n{proc.stdout}"
        ) from exc


# ---------------------------------------------------------------------------
# Gate evidence: the ambient command and its honest outcome
# ---------------------------------------------------------------------------


def test_ambient_pipeline_must_pass_exit_zero():
    """``code-intel . --mode normal --json`` MUST exit 0 (mandatory conformance).

    This is the AIE architecture gate.  Exit 10 / ``domain_failed`` remains a
    real failure.  Success must also use the compiled execution-result
    contract: ``outcome=completed`` with no domain/process failures.

    The prior version of this test accepted exit 10 as a "known baseline"
    — that was a weakening of the mandatory conformance assertion.  This
    test restores the requirement: exit 0 is the only acceptable outcome.
    """
    proc = _run_pipeline()
    result = _parse_result(proc)

    # Mandatory conformance: the pipeline must exit 0 with a successful
    # outcome.  Any other exit code or outcome is a gate failure.
    assert proc.returncode == 0, (
        f"Code Intel pipeline MUST exit 0 (mandatory conformance).  "
        f"Got exit {proc.returncode} — architecture gate is FAILING.\n"
        f"  Diagnostic: {result.get('diagnostic', 'none')}\n"
        f"  Failure node: {result.get('failureNode', 'none')}\n"
        f"  stderr: {proc.stderr}\n"
        f"This is a FAILING RATCHET — do not save a new baseline."
    )
    assert result["outcome"] == "completed", (
        f"Expected canonical outcome 'completed', got {result['outcome']!r}.  "
        f"Architecture gate is FAILING."
    )
    assert result["schema"] == "code-intel-primary-result.v1"
    assert result.get("failureNode") is None
    assert result.get("diagnostic") is None
    assert result.get("failures") == {"domain": [], "process": []}


def test_ambient_pipeline_failure_must_be_manifest_reconciliation():
    """When the pipeline fails, the diagnostic must identify manifest reconciliation.

    This test isolates the root cause: canonical beta.5 cannot resolve the
    correct manifest from the ambient environment.  It documents the specific
    blocker so future resolution targets the correct mechanism.

    If the pipeline passes (exit 0), this test is a no-op — the architecture
    gate is resolved and the mandatory-conformance test above proves it.
    """
    proc = _run_pipeline()
    result = _parse_result(proc)

    if proc.returncode == 0:
        # Pipeline passes — architecture gate resolved.  No-op.
        return

    diagnostic = result.get("diagnostic", "")
    valid_diagnostics = [
        "manifest reconciliation failed",
        "architecture gate failure",
        "bootstrap readiness failed",
    ]
    matched = any(d in diagnostic for d in valid_diagnostics)
    assert matched, (
        f"Expected one of {valid_diagnostics} in diagnostic, "
        f"got: {diagnostic!r}"
    )
    assert result["failureNode"] is not None, (
        "Expected a non-null failureNode for blocked pipeline"
    )


# ---------------------------------------------------------------------------
# Honest gate evidence: the explicit orchestrate command (no env injection)
# ---------------------------------------------------------------------------


def test_explicit_orchestrate_manifest_command_passes():
    """``code-intel orchestrate --manifest <release>/integrations.json`` passes.

    The ``orchestrate`` subcommand accepts ``--manifest`` as an explicit
    CLI flag (no env var needed).  This is the canonical explicit invocation
    and serves as evidence that the toolchain itself is functional — the
    blockage is in the default-mode manifest resolution, not in the engine.

    This test does NOT prove 1.7; it proves the toolchain is healthy when
    pointed at the correct manifest explicitly.
    """
    release_manifest = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "code-intel"
        / "releases"
        / "v0.7.0-beta.5"
        / "orchestration"
        / "integrations.json"
    )
    if not release_manifest.is_file():
        raise AssertionError(
            f"Canonical release manifest not found at {release_manifest}"
        )

    proc = subprocess.run(
        [
            "code-intel", "orchestrate",
            "--repo", ".",
            "--mode", "normal",
            "--manifest", str(release_manifest),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_ambient_env(),
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"orchestrate with explicit manifest failed: exit {proc.returncode}\n"
        f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    )
    result = json.loads(proc.stdout)
    assert result.get("ok") is True, (
        f"orchestrate returned ok=false: {json.dumps(result, indent=2)}"
    )
