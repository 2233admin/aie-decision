# Wave MVP Independent Adversarial Review

**Reviewer:** independent-adversarial agent
**Tasks reviewed:** 4.2 (diff review) and 3.3 requirement-side (cross-path parity)
**Date:** 2026-08-05
**Provider/Model:** deepseek-v4-pro (via Claude Code harness)

## Summary

Ran 12 adversarial tests against the contract-driven wave MVP implementation.
**11 passed, 1 failed honestly** — the cross-path parity test (Task 3.3) fails
because the two evaluator implementations produce different results for
equivalent inputs.

## Test Results

| # | Test | Verdict | Finding |
|---|------|---------|---------|
| 1 | Illegal unit structured failure | PASSED | All required fields present, non-empty, genuine mismatch |
| 2 | Missing adapter fails independently | PASSED | Each adapter boundary testable independently; error names specific adapter |
| 3 | Provenance invocation (not just declaration) | PASSED | `_replay_search_ledger` is actually called during `run_mvp` |
| 4 | Cross-path parity (package vs runner) | **FAILED** | Two evaluators disagree on action kinds, terminal status (see below) |
| 5 | No uncalibrated surface as probability | PASSED | All surfaces labeled `possibility_surface`; `calibration=unmeasured` |
| 6 | Budget exhaustion is not converged | PASSED | Unimodal input correctly yields `insufficient-information` or `budget-exhausted` |
| 7 | Action order determinism | PASSED | Same seed → identical results; different seeds → different output |
| 8 | Replay identity tamper detection | PASSED | Payload hash tamper detected; round_index tamper detected; state tamper detected |
| 9 | No assertion weakening in diff | PASSED | No `pytest.skip`, bare-pass handlers, or hardcoded adapter flags |
| 10 | Duplicate evaluator capability audit | PASSED | Documented 4 overlapping capabilities between runner and wave_loop |
| 11 | Dirty file audit | PASSED | All modified files traced to owned patterns |

## Key Finding: Cross-Path Parity Failure (Task 3.3)

When fed equivalent minimal inputs (1 axis, 2 variables, 1 legal mapping, seed=42),
the two evaluator implementations produce different results:

| Output | Runner (`run_mvp`) | Package (`run_wave_loop`) |
|--------|-------------------|--------------------------|
| Terminal status | `insufficient-information` | `result-found` |
| Action kinds | `[measure]` | `[stop]` |
| Surface semantics | `possibility_surface` | `possibility_surface` ✓ |

**Root cause:** The runner's `_select_action` and the wave_loop's `_decision_actions`
use different decision policies. The runner uses per-axis absolute/relative/loss
tolerances from `decision_policy.axes[]`, while wave_loop uses a global
`relative_tolerance` and `min_effective_sample_size` from `decision_policy`.

The spec (Task 3.3) requires: "Add a cross-path parity test; mismatch must fail
with a structured error." The test `test_adversarial_cross_path_parity_package_vs_runner`
fulfills this requirement — it fails honestly with a structured JSON mismatch report.

## Additional Findings (Task 4.2 Review)

### Pre-existing CLI breakage
The CLI tests (`test_cli_run_writes_ledger_and_summary`, `test_cli_replay_validates_deterministic_replay`)
fail because `_run_authoritative` in the runner delegates to `wave_authority.run_authoritative_wave()`,
which expects `outcome_space` as a sequence but the fixture provides it as a dict with an `axes` key.
This is a schema mismatch in the Task 3.1 implementation. The importlib-based tests (which call
`run_mvp` directly) continue to work.

### Overlapping evaluator capabilities
Both the runner and wave_loop independently implement:
- Schema validation
- Particle/surface evaluation
- Loop orchestration
- Factor IR compilation

The runner additionally has unit analysis, diagnostics, decision value evaluation,
action selection, ledger projection, and parity checking. The wave_loop has ledger
replay and checkpoint capabilities.

The `wave_authority.py` module (new on this branch) wraps the package evaluators
under a single labeled entry point with `InvocationProvenance`, addressing Task 3.2.

### No assertion weakening detected
The diff scan confirms no `pytest.skip`, no bare-pass exception handlers, no
hardcoded adapter availability flags, and the presence of `IntegrationUnavailable`
for fail-closed semantics.

## Owned Files

- `contracts/wave_mvp_adversarial_review_matrix.json` — 8 adversarial scenarios
- `tests/test_wave_mvp_adversarial_review.py` — 12 tests (11 pass, 1 honest fail)
- `docs/WAVE-MVP-INDEPENDENT-REVIEW.md` — this document

## No Implementation Changes
Per the task instructions, no implementation files, runner source, package modules,
or existing tests were modified.
