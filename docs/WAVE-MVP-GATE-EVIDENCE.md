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

## Current Gate Status (2026-08-05)

| Gate | Result |
|---|---|
| Focused tests (29/29) | **passed** |
| Full pytest (421 passed, 2 pre-existing failures) | **fail** (pre-existing; `wave_authority.py` TypeError — not owned) |
| Code Intel | **failed** (bootstrap readiness failed; manifest reconciliation failed; repository inputs do not match expected snapshot identity) |
| Sentrux rules | **passed** (architecture rules pass) |
| Sentrux ratchet | **fail** (baseline incompatible with new wave architecture layers) |
| Independent review | **approved** (no assertion weakening, no skip paths introduced) |

The overall gate status is **fail** — the Sentrux ratchet baseline mismatch and the pre-existing `wave_authority.py` failures are real blockers that must not be hidden.

## Owned files

- `schemas/wave_mvp_gate_evidence.schema.json`
- `scripts/validate_wave_mvp_gate_evidence.py`
- `tests/test_wave_mvp_gate_evidence.py`
- `docs/WAVE-MVP-GATE-EVIDENCE.md`
