## Context

The existing MVP has typed modules, a standalone golden runner, and replay tests, but the runner and `wave_loop.py` duplicate core behavior. Optional imports and skip paths conceal missing integration. See `proposal.md` and `specs/contract-driven-wave-mvp/spec.md` for the contract being hardened.

## Goals / Non-Goals

**Goals:**

- Select one authoritative evaluator and expose a narrow instrumentation/provenance boundary.
- Make required imports and adapters fail closed.
- Turn every acceptance scenario into a non-skippable executable test.
- Keep task completion and Code Intel/Sentrux evidence explicit and machine-readable.

**Non-Goals:**

- GPU, empirical calibration, HMC, or public packaging.
- Replacing the existing scalar Fermi path.
- Treating the current Sentrux baseline mismatch as solved by creating a new baseline.

## Decisions

### 1. Make the package path authoritative

The package modules (`joint_schema`, `factor_ir`, `particle_surface`, `wave_diagnostics`, and `wave_loop`) become the implementation path. The fixture runner becomes a thin adapter that loads fixture data, calls the package evaluator, emits evidence, and performs compatibility checks. A self-contained implementation is not allowed to become a hidden second product path.

Alternative: keep two implementations and compare them. Rejected for this MVP because parity itself would become an unowned second project; any oracle must be small, explicit, and unable to publish results.

### 2. Fail closed at required boundaries

Remove conditional skips and no-op fallbacks for required candidate generation, replay, schema, numerical dependencies, and calibration gates. Use stable error codes and include the boundary, cause, and remediation. Optional enrichment remains explicitly optional and is reported as unavailable.

### 3. Bind scenarios to evidence

Add a scenario matrix mapping each hardening requirement to test ids, owned files, and gate commands. The matrix is checked by tests for existence and by the final review for actual execution. A passing unit test alone cannot close a cross-module task.

### 4. Use Code Intel as a gate with honest outcomes

Run the compiled Code Intel entry and Sentrux rules check after edits. Rules passing with an incompatible baseline remains a structured gate failure. No baseline write is part of this change.

## Risks / Trade-offs

- [Risk] Existing fixture behavior may differ from the package path. → Treat the fixture as an oracle only, add parity tests, and fail on mismatch.
- [Risk] Removing fallbacks exposes missing local dependencies. → Report structured `integration_unavailable`; do not install or fake a dependency silently.
- [Risk] Existing dirty files are user-owned. → Edit only new hardening files and the explicitly owned MVP runner/tests; never reset or normalize unrelated files.
