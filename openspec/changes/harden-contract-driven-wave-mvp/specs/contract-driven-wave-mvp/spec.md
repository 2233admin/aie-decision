## Purpose

This capability makes the CPU wave MVP an auditable contract-driven product slice: one evaluator path, fail-closed integration boundaries, and executable evidence that maps each OpenSpec scenario to a real business result.

## ADDED Requirements

### Requirement: One authoritative evaluator path

The CPU MVP MUST evaluate a request through one authoritative path from versioned schema and restricted factor mapping to particle surface, diagnostics, typed action, ledger, and replay. A second evaluator MAY exist only as an explicitly labelled test oracle and MUST be compared against the authoritative path.

#### Scenario: CLI uses the authoritative path

- **WHEN** the golden fixture is executed through the MVP CLI
- **THEN** the result records the evaluator version and proves that the authoritative schema, factor, surface, diagnostic, loop, and replay components were invoked

#### Scenario: Oracle disagrees with the authoritative path

- **WHEN** an oracle and the authoritative path produce different surface summaries, actions, or ledger state
- **THEN** the run fails with a structured parity error and MUST NOT publish a successful result

### Requirement: Required boundaries fail closed

The MVP MUST reject missing required adapters, unsupported schema versions, missing numerical dependencies, invalid mappings, and unavailable replay support with structured failures. It MUST NOT silently skip tests, use a no-op adapter, substitute a different evaluator, or claim completion.

#### Scenario: Required adapter is unavailable

- **WHEN** the candidate-generation or replay adapter cannot be imported or invoked
- **THEN** the CLI returns a structured `integration_unavailable` failure and the acceptance test fails rather than skipping

#### Scenario: Uncalibrated probability is requested

- **WHEN** a caller requests `probability_surface` without validated calibration for every contributor
- **THEN** evaluation rejects the request with the blocking contributors and does not downgrade silently to a successful possibility result

### Requirement: Tests judge behavior and invariants

Acceptance tests MUST assert externally meaningful outputs, failure codes, provenance, action ordering, replay identity, and cross-component invocation. Tests MUST NOT replace a failed assertion with a weaker assertion, broad membership check, conditional skip, or accepted nonzero exit code unless the contract explicitly defines that outcome.

#### Scenario: Illegal unit mapping

- **WHEN** a fixture adds incompatible units
- **THEN** the test requires `unit_mismatch` plus mapping id, operand, expected unit, and actual unit

#### Scenario: Budget exhaustion

- **WHEN** the loop exhausts budget before decision criteria pass
- **THEN** the test requires unresolved/budget-exhausted status and rejects `converged` or `accepted`

### Requirement: Completion is evidence-bound

An implementation task MUST identify its owned files, prerequisite tasks, scenario ids, focused tests, full-test result, and Code Intel/Sentrux result. A task MUST remain incomplete when any required evidence is missing or a quality gate fails.

#### Scenario: Worker reports completion without integration evidence

- **WHEN** a worker reports code and unit tests but no scenario mapping or integration/quality evidence
- **THEN** the task remains incomplete and cannot be merged as accepted MVP work

#### Scenario: Quality gate has a real baseline failure

- **WHEN** Code Intel or Sentrux rules pass but the authoritative baseline/domain gate fails
- **THEN** the result records the failure as a blocker and MUST NOT be reported as fully green
