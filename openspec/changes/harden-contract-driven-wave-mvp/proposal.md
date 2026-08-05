## Why

The first CPU wave MVP passed its local tests while containing three separate implementations of the schema, factor evaluation, particle surface, and loop. Optional imports and `pytest.skip` paths allowed a broken integration to look successful. The change needs a contract-first hardening pass so OpenSpec scenarios, executable tests, task ownership, and Code Intel evidence agree before another worker can claim completion.

## Verified Current State

- `joint_schema.py` and `factor_ir.py` provide typed domain contracts, but `wave_loop.py` reimplements compatible concepts locally.
- `run_joint_wave_surface_mvp.py` is a second self-contained evaluator and only optionally imports legacy candidate/replay modules.
- Compatibility tests can skip when adapters are unavailable; the runner can silently degrade to no-op adapters.
- The four MVP commits are local and pass pytest, but the execution chain is not proven equivalent across implementations.
- Code Intel rules pass while the repository baseline has an incompatible schema/engine and the authoritative domain gate remains failed.

## What Changes

- Make one authoritative CPU execution path from versioned schema and FactorIR through particle surface, diagnostics, loop, ledger, and replay.
- Remove silent adapter degradation: missing required adapters, schema versions, dependencies, and calibration evidence fail closed with structured errors.
- Replace skip-or-weak assertions with mandatory integration assertions that prove the real modules are invoked and produce the expected business result.
- Add OpenSpec scenario-to-test mapping and task completion evidence so a worker cannot report a task complete without the corresponding executable contract and quality-gate record.
- Add Code Intel/Sentrux checks as a required implementation gate, preserving genuine baseline/domain failures.

## Capabilities

### New Capabilities

- `contract-driven-wave-mvp`: Defines the authoritative execution path, fail-closed integration behavior, and scenario-linked evidence required for the CPU MVP.

### Modified Capabilities

- None. This is a hardening capability layered over the existing joint-wave change; the original change remains open until its broader task list is independently satisfied.

## Non-goals

- GPU execution, historical calibration, product packaging, and HMC integration.
- Marking the original 62-task change complete merely because the MVP fixture passes.
- Generating or replacing a Sentrux baseline to hide a ratchet or architecture failure.

## Authority

- The three existing joint-wave capability specs and their scenarios remain the product behavior authority.
- The new contract-driven capability governs integration and failure semantics for this hardening change.
- OpenSpec task status, executable tests, git commits, and Code Intel/Sentrux artifacts must agree before completion is reported.

## Dependencies

- `add-joint-wave-surface-mapping` existing schema and MVP commits.
- Existing `candidate_generation.py`, `search.py`, `ledger.py`, and `search_replay.py` public boundaries.
- Current compiled Code Intel entry and native Sentrux rules; baseline debt remains an explicit blocker.
