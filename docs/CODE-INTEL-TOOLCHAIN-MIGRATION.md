# Code Intel / Sentrux toolchain migration evidence

Date: 2026-08-05

This record is for the P0 MVP gate. It separates the installed pipeline
diagnostic from the repository implementation and does not weaken any test or
gate.

## Canonical toolchain

- Compiled entry: `code-intel 0.7.0-beta.5`.
- Native Sentrux: `sentrux-native 2.2.0`.
- `code-intel doctor bootstrap --repo-path . --platform windows --json`:
  `ok: true`, `missing: []`.
- Repowise is present at `C:\Users\Administrator\.local\bin\repowise.exe`.
- The installed beta.5 `pipeline.config.json` and `legacy/run-code-intel.ps1`
  were restored into the canonical `bin` directory; no repository fallback
  runner was introduced.

## Baseline migration

- Old baseline: `.sentrux/baseline.json`, tool `sentrux-lite`, SHA-256
  `86D0E9A27957D1026361AFA2FBE275CD29C81332D36DE8EB733CE69BB51DA762`.
- Recoverable copy: `.sentrux/legacy-baseline.sentrux-lite.20260805.json`.
- New baseline: schema `code-intel-sentrux-baseline.v4`, engine
  `sentrux-native 2.2.0`, SHA-256
  `72C0C6967BEFB406F09DA9A7DD3FC1D38ADE2BC3F06B9F32B4BC77E8BEF0128D`.
- Native metrics: 79 files, 1024 functions, quality 8795, coupling 60.51,
  0 cycles, 6 god files, max complexity 14.
- Native `sentrux check . --no-ratchet`: passed; native `sentrux check .`:
  passed with no degradation.

## Full pipeline result

The default convenience invocation `code-intel . --mode normal --json` now
passes with exit code 0, outcome `completed`, and empty domain/process
failures, after two root causes were resolved:

1. **Snapshot identity instability (TOCTOU).** The `ExplicitOverlay`
   working-tree policy includes untracked files in the snapshot digest.
   Transient directories (`.codex/`, `.omc/`, `.sentrux/agent-sessions/`)
   were being created/modified between the parent and child snapshot
   computations, causing `repo.snapshot` to fail with "repository inputs do
   not match the expected snapshot identity".  Fixed by adding these
   directories to `.gitignore`, which excludes them from the overlay via
   `git ls-files --others --exclude-per-directory=.gitignore`.

2. **Wrong pipeline root in manifest discovery.** The installed
   `bin/orchestration/integrations.json` causes `pipeline_root()` to resolve
   to `<AppData>/code-intel/bin/` instead of
   `<AppData>/code-intel/releases/v0.7.0-beta.5/`.  The doctor and
   orchestrate Validate checks then look for source files (e.g.
   `crates/code-intel-cli/src/doctor_bootstrap/mod.rs`) relative to the wrong
   root.  Fixed by setting the canonical supported env var
   `CODE_INTEL_INTEGRATIONS_MANIFEST` to the manifest inside the release
   directory.

   Additionally, the host's global `CODE_INTEL_HOME` env var pointed to a
   non-existent development directory; when set, doctor bootstrap fails
   closed with "CODE_INTEL_HOME: directory does not exist".  Unsetting it
   allows the default derivation (`<pipeline_root>`) to be used.

Verified command:
```text
CODE_INTEL_INTEGRATIONS_MANIFEST=<release>/orchestration/integrations.json \
  code-intel . --mode normal --json
```
Observed: exit 0, outcome `completed`, `failures.domain: []`,
`failures.process: []`, all DAG nodes passed.

Tests: `tests/test_code_intel_pipeline.py` — 4/4 pass including adversarial
fail-closed cases (missing manifest env var, stale CODE_INTEL_HOME).

## Lite-session boundary (ESCALATED — OpenSpec 1.6)

The beta.5 compatibility `legacy/Invoke-SentruxAgentTool.ps1` still reads and
writes `.sentrux/baseline.json` for `session_start` / `session_end`.  This
was verified with a black-box test (`tests/test_lite_session_contract.py`)
that:

1. Records the native baseline SHA-256 before `session_start`.
2. Invokes `pwsh -File Invoke-SentruxAgentTool.ps1 session_start .`.
3. Asserts the native baseline SHA-256 is bit-for-bit identical.
4. Asserts `.sentrux/cache/lite-baseline.json` was written with a clear
   tool/engine identity.

**Result: the test FAILS** — `session_start` overwrites the native baseline
(v4, engine `sentrux-native 2.2.0`, quality 8795) with the lite format
(tool `sentrux-lite`, quality 8776).  The SHA-256 changes and byte length
drops from 589 to 387 bytes.

### Upstream gap analysis

No repo-level extension surface exists in canonical beta.5 to redirect lite
session baseline writes without either:

- Modifying the installed `Invoke-SentruxAgentTool.ps1` (in AppData —
  forbidden by the task contract).
- Reimplementing the Sentrux gate evaluator (forbidden — “不要发明第二套
  Sentrux evaluator”).
- Adding a compatibility fallback wrapper (forbidden — “不要加兼容 fallback”).

The `CODE_INTEL_REPO_ROOT` env var allows redirecting `sentrux-shim.ps1`
location, but:
- `Invoke-SentruxAgentTool.ps1` calls `Invoke-Native “sentrux”` which
  resolves through `sentrux.cmd` → `sentrux-shim.ps1`.  The shim falls
  back to `sentrux-lite-core.ps1` when no native `sentrux.exe` is found.
- `sentrux-lite-core.ps1` hard-codes `Get-BaselinePath` → `.sentrux/baseline.json`.
- Intercepting `gate --save` in a custom shim would require reimplementing
  the gate comparison logic (reading from cache baseline, computing metrics,
  comparing) = second evaluator = forbidden.

### Required upstream changes

To make the cache-only contract implementable from the repository:

- **File**: `crates/code-intel-cli/src/sentrux_gate.rs` or the native
  `sentrux` binary — add a `--baseline-path <path>` flag to `gate --save`
  to control the output path.
- **File**: `legacy/Invoke-SentruxAgentTool.ps1`, function
  `Invoke-SessionStartTool` (line 506) — accept a `-LiteBaselinePath`
  parameter and pass it through to the native evaluator.
- **File**: `legacy/tools/sentrux-shim/sentrux-lite-core.ps1`, function
  `Get-BaselinePath` (line 253) and `Write-Baseline` (line 258) — accept
  an override path from an environment variable or command-line flag.

## Review conclusion

The default pipeline now passes with the correct `CODE_INTEL_INTEGRATIONS_MANIFEST`
env var and `.gitignore` exclusions for transient directories.  The
lite-session cache-only contract is escalated as an upstream toolchain gap;
no repo-local implementation is possible without violating the task
constraints.  Test evidence in `tests/test_lite_session_contract.py`
demonstrates the violation precisely.
