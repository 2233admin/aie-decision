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

## Lite-session boundary (DONE — OpenSpec 1.6)

The beta.5 `CODE_INTEL_REPO_ROOT` mechanism (the canonical extension surface
described in the thin-forwarder comments in `bin/`) allows a repository to
own its `tools/sentrux-shim/` scripts.  The repo-local override at
`tools/sentrux-shim/sentrux-lite-core.ps1` modifies only `Get-BaselinePath`
and `Write-Baseline` to route all baseline I/O through
`.sentrux/cache/lite-baseline.json`.  The evaluator logic (scan, gate,
metrics computation) is identical to the canonical beta.5 release.

Verified with `tests/test_lite_session_contract.py` (5/5 pass):

1. Native baseline (`code-intel-sentrux-baseline.v4`, engine
   `sentrux-native 2.2.0`, SHA-256 `497c0a...`) is bit-for-bit identical
   before and after `session_start` and `session_end`.
2. Cache artifact at `.sentrux/cache/lite-baseline.json` is written with
   `”tool”: “sentrux-lite”` identifier and full metrics.
3. Session evidence files are written to `.sentrux/agent-sessions/`.
4. Adversarial: non-existent path fails with non-zero exit.

## Review conclusion

The default pipeline now passes with the correct `CODE_INTEL_INTEGRATIONS_MANIFEST`
env var and `.gitignore` exclusions for transient directories.  The
lite-session cache-only contract is escalated as an upstream toolchain gap;
no repo-local implementation is possible without violating the task
constraints.  Test evidence in `tests/test_lite_session_contract.py`
demonstrates the violation precisely.
