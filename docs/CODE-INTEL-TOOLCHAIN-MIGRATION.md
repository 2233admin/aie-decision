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

The default convenience invocation `code-intel . --mode normal --json`
remains honestly recorded as `domain_failed` because the repository does not
contain the pipeline-owned `orchestration/integrations.json`; the diagnostic
was `manifest reconciliation failed`. This is not a product failure and is
not hidden.

The canonical compiled runner succeeds when given the canonical beta.5
manifest explicitly:

```text
code-intel run execute --repo . --out <staging> --authority-root <authority> \
  --final-name <run> \
  --manifest C:\Users\Administrator\AppData\Local\code-intel\releases\v0.7.0-beta.5\orchestration\integrations.json \
  --doctor-require-repowise false
```

Observed result: exit code 0, `failures.domain: []`, `failures.process: []`,
and doctor, graph, native-code, Sentrux, inventory, and diagnosis nodes all
passed. The run used snapshot
`f72fd148a7affbf6afa9f6225d108ee49a35402e2679a026c1940b5b3595e35c`.

## Lite-session boundary

The beta.5 compatibility `legacy/Invoke-SentruxAgentTool.ps1` still reads and
writes `.sentrux/baseline.json` for `session_start`; it does not satisfy the
required “lite writes only `.sentrux/cache/lite-baseline.json`” contract.
That task remains open. It was not executed because doing so would mutate the
native baseline and would conceal the boundary failure.

## Review conclusion

The MVP gate may use the explicit canonical manifest command above. The
default wrapper's missing-manifest diagnosis and the lite-session contract are
real toolchain debt, not reasons to alter product assertions or claim a clean
unqualified pipeline pass.
