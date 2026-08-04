## 1. Contract and evidence matrix

- [x] 1.1 Add a machine-readable scenario matrix covering every requirement/scenario in `specs/contract-driven-wave-mvp/spec.md`, with test id, owner, input, expected business output, and failure semantics.
- [x] 1.2 Add a completion record schema/validator requiring focused tests, full pytest result, Code Intel result, Sentrux result, commit, and dirty-file declaration.

## 2. Fail-closed acceptance

- [x] 2.1 Remove required-boundary `pytest.skip` paths and optional no-op success from the MVP acceptance path; missing adapters become `integration_unavailable` failures.
- [x] 2.2 Strengthen assertions for illegal units, probability gating, budget exhaustion, action order, provenance, and replay identity.
- [x] 2.3 Add a negative test that proves a missing adapter and an oracle/authoritative mismatch cannot pass acceptance.

## 3. Authoritative integration

- [x] 3.1 Route the golden CLI through the package evaluator or explicitly mark the current runner as a non-authoritative oracle; do not allow two unlabelled implementations.
- [x] 3.2 Add invocation provenance proving the authoritative schema, FactorIR, surface, diagnostics, loop, ledger, and replay boundaries were called.
- [x] 3.3 Add a cross-path parity test; mismatch must fail with a structured error.

## 4. Quality gates and review

- [ ] 4.1 Run focused and full pytest, OpenSpec strict validation, compiled Code Intel, and Sentrux rules/ratchet checks; record real failures.
- [x] 4.2 Review the diff for assertion weakening, skips, no-op fallbacks, duplicate evaluators, and unrelated dirty-file changes.
- [ ] 4.3 Commit only the hardening change after all achievable gates pass; leave Code Intel baseline debt explicitly open.
