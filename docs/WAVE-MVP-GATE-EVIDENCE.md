# Wave MVP Gate Evidence

Fail-closed, machine-readable gate record for OpenSpec task 4.1 and completion traceability of the `harden-contract-driven-wave-mvp` change.

## Schema

`schemas/wave_mvp_gate_evidence.schema.json` — JSON Schema 2020-12 defining the gate evidence record.

### Required fields

| Field | Description |
|---|---|
| `schema_version` | `"wave-mvp-gate-evidence.v1"` |
| `gate_id` | Stable run identifier |
| `openspec_tasks` | Array of `{task_id, status, blocker?, owned_files?}` |
| `status` | Aggregate gate status: `pass`, `fail`, or `blocked` |
| `focused_pytest` | `{command, passed, output_path?}` |
| `full_pytest` | `{command, passed, output_path?}` |
| `code_intel` | `{command, outcome, publication_path, failure_detail?}` |
| `sentrux` | `{command, rules_passed, ratchet_passed, baseline_path?, failure_detail?}` |
| `independent_review` | `{reviewer, outcome, evidence_path, findings?}` |
| `commit` | 7+ character commit hash |
| `dirty_files` | Array of unowned modified files |
| `provider` | Claude Code provider (e.g. `"anthropic"`) |
| `model` | Actual model used (e.g. `"claude-fable-5"`) |

### Per-task statuses

- `implemented` — code exists for the task
- `planned` — scoped but not yet started
- `blocked` — known blocker prevents progress (requires `blocker` field)
- `verified` — independently confirmed complete

### Gate-level status consistency rules

The validator cross-checks the declared `status` against evidence:

- `status: "pass"` — requires all tests passing, Code Intel `passed` or `blocked`, Sentrux rules and ratchet both `true`, review `approved`
- `status: "fail"` — any real failure in focused tests, full tests, Sentrux rules, or review
- `status: "blocked"` — at least one task has status `blocked`

A mismatch triggers `gate_status_mismatch` — the validator rejects the record rather than inferring or downgrading.

## Validator

`scripts/validate_wave_mvp_gate_evidence.py` — CLI and programmatic validator.

```bash
# CLI usage
python scripts/validate_wave_mvp_gate_evidence.py path/to/gate-record.json
python scripts/validate_wave_mvp_gate_evidence.py path/to/gate-record.json --strict
```

Exit codes: `0` = valid, `2` = invalid.

Programmatic:
```python
from scripts.validate_wave_mvp_gate_evidence import validate_gate_evidence
result = validate_gate_evidence(record)
# {"gate_id": "...", "valid": true, "status": "pass", "warnings": [...]}
```

### Invariants enforced

1. **Missing evidence is rejected** — every required field must be present and non-empty
2. **Mislabeled evidence is rejected** — status must be consistent with actual test/CI/Sentrux/review results
3. **No baseline creation** — the validator is a pure function; it never writes files
4. **Real failures are preserved** — a Code Intel or Sentrux failure is never downgraded to `blocked` or `pass`

## Tests

`tests/test_wave_mvp_gate_evidence.py` — 29 fail-closed tests covering:

- Schema structural validation (missing fields, unknown fields, wrong consts)
- Semantic gate validation (status consistency for every evidence category)
- CLI integration (file I/O, JSON parsing, exit codes)
- Invariant tests (no disk writes, real failures preserved, ratchet failure explicit)

## Current Gate Status (2026-08-05, re-captured live)

| Gate | Result |
|---|---|
| Focused tests (220 wave/hardening) | **passed** |
| Full pytest (449 passed) | **passed** |
| Code Intel | **failed** (`process_failed` — bootstrap readiness failed; manifest reconciliation failed; dirty working tree prevents stable snapshot) |
| Sentrux rules | **passed** (All rules passed, Quality: 8795) |
| Sentrux ratchet | **fail** (baseline engine mismatch: `.sentrux/baseline.json` requires `sentrux-native` engine v4 but current checker reports unknown engine) |
| Independent review | **approved** (`12/12` adversarial tests pass; no assertion weakening, no skip paths, no duplicate evaluators) |

The overall gate status is **blocked** — task 4.3 cannot close because the Code Intel
snapshot identity and Sentrux ratchet engine mismatch are real blockers, both
outside the owned gate-evidence scope. No baseline was regenerated to hide them.

**Commit:** `da1e28b9e038d91fd2d15d3f80b9741a1be06111`
**Provider:** `deepseek` (harness-orchestrated)
**Model:** `deepseek-v4-pro`
**Publication path:** `C:/Users/Administrator/AppData/Local/code-intel/artifacts/aie-decision/1785871915797-57116-core`

Dirty files: 7 tracked modifications (deleted agents/, modified schemas, modified src/tests) plus many untracked artifacts — none owned by the gate evidence scope.

The machine-readable current capture is
`docs/wave_mvp_gate_evidence.current.json` and validates with the fail-closed
validator (`{"valid": true, "status": "blocked"}`).

### Why the previous evidence worker stopped after creating a validator

The previous worker (task 4.1, commit `08d5c38`) correctly created the schema,
validator, and 29 tests — the *infrastructure* for evidence gating. It did not
close the gate because it could not: Code Intel was `process_failed` (dirty tree)
and Sentrux ratchet was failing (baseline engine mismatch). Both are
environmental/infrastructure blockers outside the validator's scope. The worker
correctly preserved these real failures rather than masking them.

## Owned files

- `schemas/wave_mvp_gate_evidence.schema.json`
- `scripts/validate_wave_mvp_gate_evidence.py`
- `tests/test_wave_mvp_gate_evidence.py`
- `docs/WAVE-MVP-GATE-EVIDENCE.md`
- `docs/wave_mvp_gate_evidence.current.json`
