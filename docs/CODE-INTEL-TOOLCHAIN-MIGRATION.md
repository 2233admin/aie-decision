# Code Intel / Sentrux toolchain migration evidence

Date: 2026-08-05

This record is for the P0 MVP gate. It separates the installed pipeline
diagnostic from the repository implementation and does not weaken any test or
gate.

## Canonical toolchain

- Compiled entry: `code-intel 0.7.0-beta.5`.
- Native Sentrux: `sentrux-native 2.2.0`.
- `code-intel doctor bootstrap --repo-path . --platform windows --json`:
  `ok: false`, `missing: ["CODE_INTEL_HOME: directory does not exist"]`.
  The host's global `CODE_INTEL_HOME` env var points to a non-existent
  development directory; when set, doctor bootstrap fails closed.
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

## Snapshot identity (TOCTOU fix)

The `ExplicitOverlay` working-tree policy includes untracked files in the
snapshot digest.  Transient directories (`.codex/`, `.omc/`,
`.sentrux/agent-sessions/`) were being created/modified between parent and
child snapshot computations, causing `repo.snapshot` to fail with "repository
inputs do not match the expected snapshot identity".  Fixed by adding these
directories to `.gitignore`, which excludes them from the overlay via
`git ls-files --others --exclude-per-directory=.gitignore`.

This is a legitimate infrastructure fix and does not constitute an
evaluator, fallback, or environment injection.

## Full pipeline result — BLOCKED (OpenSpec 1.7)

The default convenience invocation `code-intel . --mode normal --json`
is honestly recorded as `domain_failed` with exit code 10:

```text
code-intel . --mode normal --json
→ exit 10, outcome domain_failed
→ diagnostic: "doctor diagnosis: bootstrap readiness failed; manifest reconciliation failed"
→ failureNode: doctor
```

### Root cause: pipeline root resolution

The installed `bin/orchestration/integrations.json` causes `pipeline_root()`
to resolve to `<AppData>/code-intel/bin/` instead of the release directory
`<AppData>/code-intel/releases/v0.7.0-beta.5/`.  The doctor and orchestrate
Validate checks then look for source files relative to the wrong root.

The canonical `CODE_INTEL_INTEGRATIONS_MANIFEST` env var is the supported
mechanism to override manifest discovery.  Setting it to the manifest inside
the release directory resolves the issue:

```text
CODE_INTEL_INTEGRATIONS_MANIFEST=<release>/orchestration/integrations.json \
  code-intel . --mode normal --json
→ exit 0, outcome completed, empty failures
```

The explicit `orchestrate --manifest` flag also works without env vars:

```text
code-intel orchestrate --repo . --mode normal \
  --manifest <release>/orchestration/integrations.json --json
→ ok: true
```

### Upstream gap

No repo-local manifest discovery mechanism exists in canonical beta.5. The
product does not search for a `.code-intel/` directory,
`pipeline.config.json`, `integrations.json`, or any other config file within
the repository working tree when resolving the manifest for the default
`code-intel .` command.  Without `CODE_INTEL_INTEGRATIONS_MANIFEST` set, the
fallback (ancestor-walk from the exe parent directory) resolves to `bin/`,
which is incorrect for all releases.

This is an upstream product design issue.  The task constraint prohibits
environment injection to work around it, so 1.7 remains **BLOCKED**.

### Test evidence

`tests/test_code_intel_pipeline.py` — 1/3 pass:
- `test_ambient_pipeline_reports_domain_failed` — **FAILS** (asserts exit 0,
  observed exit 10).  This is honest gate evidence.
- `test_ambient_pipeline_diagnostic_is_manifest_reconciliation` — **PASSES**
  (confirms the specific root cause).
- `test_explicit_orchestrate_manifest_command_passes` — **PASSES** (proves
  the toolchain is healthy when pointed at the correct manifest via CLI flag).

## Lite-session boundary — BLOCKED (OpenSpec 1.6)

Canonical beta.5 `legacy/Invoke-SentruxAgentTool.ps1` calls `sentrux gate
--save` which hard-codes `.sentrux/baseline.json` as the output path.  It
overwrites the native baseline (v4, engine `sentrux-native 2.2.0`) with the
lite format (tool `sentrux-lite`).  The SHA-256 changes and byte length
drops from 589 to ~387 bytes.

No repo-level extension surface exists to redirect this write without:
- Modifying the installed `Invoke-SentruxAgentTool.ps1` (in AppData — forbidden).
- Reimplementing the Sentrux gate evaluator (forbidden — "不要发明第二套
  Sentrux evaluator").
- Adding a compatibility fallback wrapper (forbidden — "不要加兼容 fallback").

The `CODE_INTEL_REPO_ROOT` env var redirects `sentrux-shim.ps1` location, but
creating a repo-local override that modifies baseline path routing requires
duplicating the evaluator logic (scan, metrics, gate comparison) — a second
evaluator.  This is explicitly forbidden.

### Previous violation (corrected 2026-08-05)

Commit 300718f introduced `tools/sentrux-shim/sentrux-shim.ps1` and
`tools/sentrux-shim/sentrux-lite-core.ps1` (~514 lines total) claiming to
use `CODE_INTEL_REPO_ROOT` as a "canonical extension mechanism".  These files
constituted:
1. A complete second Sentrux evaluator (scan, check, gate, baseline
   management — ~316 lines).
2. A catch-fallback mechanism (try native core → catch → fallback to lite
   core, lines 169-176 of sentrux-shim.ps1).

Both violate the task constraints ("不得第二套 evaluator、不得 fallback").
These files have been removed in the correction commit.

### Required upstream changes

To make the cache-only contract implementable from the repository:
- `crates/code-intel-cli/src/sentrux_gate.rs` or the native `sentrux` binary:
  add a `--baseline-path <path>` flag to `gate --save`.
- `legacy/Invoke-SentruxAgentTool.ps1`: accept a `-LiteBaselinePath` parameter
  and pass it through to the native evaluator.
- `legacy/tools/sentrux-shim/sentrux-lite-core.ps1`: accept an override path
  from an environment variable or command-line flag.

### Test evidence

`tests/test_lite_session_contract.py` — 3/5 pass, 2 intentional FAILS:
- `test_prerequisites` — **PASSES** (canonical toolchain files exist).
- `test_native_baseline_schema_is_v4` — **PASSES** (baseline is correct schema).
- `test_session_start_overwrites_native_baseline` — **FAILS** (session_start
  alters the native baseline).  Honest gate evidence for upstream blocker.
- `test_session_end_overwrites_native_baseline` — **FAILS** (session_end
  alters the native baseline).  Honest gate evidence for upstream blocker.
- `test_session_start_with_nonexistent_path_fails_nonzero` — **PASSES**
  (adversarial fail-closed probe).

## Review conclusion

Both 1.6 and 1.7 are **BLOCKED** by upstream gaps in canonical beta.5:

1. **1.7 (manifest reconciliation):** No repo-local manifest discovery
   mechanism.  The default `code-intel . --mode normal --json` command
   resolves the pipeline root to `bin/` instead of the release directory.
   Workaround exists (`CODE_INTEL_INTEGRATIONS_MANIFEST` env var or
   `orchestrate --manifest` flag) but the task constraint prohibits
   environment injection to claim the default command passes.

2. **1.6 (lite session cache-only contract):** No repo-level extension
   surface to redirect `gate --save` output path.  Any implementation
   requires either modifying installed AppData files (forbidden) or
   duplicating the evaluator (forbidden).

The snapshot identity TOCTOU fix (`.gitignore` additions) is a legitimate
infrastructure improvement and is retained.  Both test files have been
rewritten as honest fail-closed probes with zero environment injection,
zero pytest.skip/xfail, and zero fallback mechanisms.

No evaluator duplication, mock file writes, or env-injected false passes
exist in the corrected state.
