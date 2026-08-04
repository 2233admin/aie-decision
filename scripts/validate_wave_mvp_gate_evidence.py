"""Fail-closed validator for Wave MVP gate evidence records.

Preserves real failures and rejects missing or mislabeled evidence.
Never creates a baseline or makes a failed gate pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "wave_mvp_gate_evidence.schema.json"

SCHEMA_VERSION = "wave-mvp-gate-evidence.v1"

VALID_TASK_STATUSES = frozenset({"implemented", "planned", "blocked", "verified"})
VALID_GATE_STATUSES = frozenset({"pass", "fail", "blocked"})
VALID_CI_OUTCOMES = frozenset({"passed", "failed", "blocked"})
VALID_REVIEW_OUTCOMES = frozenset({"approved", "changes_requested", "blocked"})


class GateValidationError(ValueError):
    """Structured validation failure — never suppressed or downgraded."""


def _require(value: Any, label: str) -> None:
    if not value:
        raise GateValidationError(f"missing_{label}")
    if isinstance(value, str) and not value.strip():
        raise GateValidationError(f"empty_{label}")


def _fail(reason: str) -> None:
    raise GateValidationError(reason)


def validate_gate_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a gate evidence record against the fail-closed contract.

    Returns a structured result dict.  Raises ``GateValidationError`` for
    hard failures (missing fields, impossible statuses, evidence that does
    not support the declared status).  Never infers missing data, never
    creates a baseline, and never downgrades a real failure.

    The return value is a summary suitable for machine consumption::

        {"gate_id": "...", "valid": true, "status": "pass", "warnings": [...]}
    """
    warnings: list[str] = []

    # ── schema version ────────────────────────────────────────────────
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        _fail(f"unsupported_schema_version: expected {SCHEMA_VERSION}, got {version!r}")

    # ── required top-level keys ────────────────────────────────────────
    _require(record.get("gate_id"), "gate_id")
    _require(record.get("status"), "status")
    gate_status = record["status"]
    if gate_status not in VALID_GATE_STATUSES:
        _fail(f"invalid_gate_status: {gate_status!r}")

    _require(record.get("commit"), "commit")
    if len(str(record["commit"])) < 7:
        _fail("commit_too_short: minimum 7 characters")

    _require(record.get("provider"), "provider")
    _require(record.get("model"), "model")

    # ── openspec_tasks ─────────────────────────────────────────────────
    tasks = record.get("openspec_tasks")
    if not isinstance(tasks, Sequence) or len(tasks) == 0:
        _fail("openspec_tasks_missing_or_empty")
    task_ids_seen: set[str] = set()
    has_blocked_task = False
    has_unverified_task = False
    for i, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            _fail(f"openspec_tasks[{i}]: not an object")
        tid = task.get("task_id")
        _require(tid, f"openspec_tasks[{i}].task_id")
        if tid in task_ids_seen:
            _fail(f"duplicate_task_id: {tid}")
        task_ids_seen.add(tid)
        tstatus = task.get("status")
        if tstatus not in VALID_TASK_STATUSES:
            _fail(f"openspec_tasks[{i}].status: invalid {tstatus!r}, expected one of {sorted(VALID_TASK_STATUSES)}")
        if tstatus == "blocked":
            has_blocked_task = True
            _require(task.get("blocker"), f"openspec_tasks[{i}].blocker (required for blocked status)")
        if tstatus != "verified":
            has_unverified_task = True

    # ── focused_pytest ─────────────────────────────────────────────────
    focused = record.get("focused_pytest")
    if not isinstance(focused, Mapping):
        _fail("focused_pytest: must be an object")
    _require(focused.get("command"), "focused_pytest.command")
    if not isinstance(focused.get("passed"), bool):
        _fail("focused_pytest.passed: must be a boolean")
    focused_passed = focused["passed"]

    # ── full_pytest ────────────────────────────────────────────────────
    full = record.get("full_pytest")
    if not isinstance(full, Mapping):
        _fail("full_pytest: must be an object")
    _require(full.get("command"), "full_pytest.command")
    if not isinstance(full.get("passed"), bool):
        _fail("full_pytest.passed: must be a boolean")
    full_passed = full["passed"]

    # ── code_intel ─────────────────────────────────────────────────────
    ci = record.get("code_intel")
    if not isinstance(ci, Mapping):
        _fail("code_intel: must be an object")
    _require(ci.get("command"), "code_intel.command")
    _require(ci.get("publication_path"), "code_intel.publication_path")
    ci_outcome = ci.get("outcome")
    if ci_outcome not in VALID_CI_OUTCOMES:
        _fail(f"code_intel.outcome: invalid {ci_outcome!r}, expected one of {sorted(VALID_CI_OUTCOMES)}")
    if ci_outcome == "failed":
        _require(ci.get("failure_detail"), "code_intel.failure_detail (required for failed outcome)")

    # ── sentrux ────────────────────────────────────────────────────────
    sx = record.get("sentrux")
    if not isinstance(sx, Mapping):
        _fail("sentrux: must be an object")
    _require(sx.get("command"), "sentrux.command")
    if not isinstance(sx.get("rules_passed"), bool):
        _fail("sentrux.rules_passed: must be a boolean")
    if not isinstance(sx.get("ratchet_passed"), bool):
        _fail("sentrux.ratchet_passed: must be a boolean")
    rules_ok = sx["rules_passed"]
    ratchet_ok = sx["ratchet_passed"]
    if not rules_ok or not ratchet_ok:
        _require(sx.get("failure_detail"), "sentrux.failure_detail (required when rules or ratchet fail)")

    # ── independent_review ─────────────────────────────────────────────
    review = record.get("independent_review")
    if not isinstance(review, Mapping):
        _fail("independent_review: must be an object")
    _require(review.get("reviewer"), "independent_review.reviewer")
    _require(review.get("evidence_path"), "independent_review.evidence_path")
    review_outcome = review.get("outcome")
    if review_outcome not in VALID_REVIEW_OUTCOMES:
        _fail(f"independent_review.outcome: invalid {review_outcome!r}, expected one of {sorted(VALID_REVIEW_OUTCOMES)}")
    review_ok = review_outcome == "approved"

    # ── dirty_files ────────────────────────────────────────────────────
    dirty = record.get("dirty_files")
    if not isinstance(dirty, list):
        _fail("dirty_files: must be an array")

    # ── status consistency cross-validation ────────────────────────────
    # The declared gate status MUST be consistent with the evidence.
    # A gate that claims "pass" while any required gate fails is mislabeled.

    real_failures: list[str] = []

    if not focused_passed:
        real_failures.append("focused_pytest: not passed")
    if not full_passed:
        real_failures.append("full_pytest: not passed")
    if ci_outcome == "failed":
        real_failures.append("code_intel: failed")
    if not rules_ok:
        real_failures.append("sentrux: rules_passed is false")
    if not ratchet_ok:
        real_failures.append("sentrux: ratchet_passed is false")
    if not review_ok:
        real_failures.append(f"independent_review: {review_outcome}")

    # Determine the correct gate status from the evidence.
    if has_blocked_task:
        correct_status = "blocked"
    elif real_failures:
        correct_status = "fail"
    else:
        correct_status = "pass"

    if gate_status != correct_status:
        _fail(
            f"gate_status_mismatch: declared {gate_status!r} but evidence "
            f"requires {correct_status!r} (failures: {real_failures}, "
            f"blocked_tasks: {has_blocked_task})"
        )

    # Additional warnings that do not invalidate the gate.
    if ci_outcome == "blocked":
        warnings.append("code_intel: blocked (tool could not run)")
    if has_unverified_task and gate_status == "pass":
        warnings.append("gate passes but some tasks are not yet verified")
    if dirty:
        warnings.append(f"unowned dirty files present: {len(dirty)} files")

    return {
        "gate_id": record["gate_id"],
        "valid": True,
        "status": gate_status,
        "warnings": warnings,
    }


def validate_json_schema(record: Mapping[str, Any]) -> list[str]:
    """Structural validation against the JSON Schema definition.

    Returns a list of schema-level errors (empty means structurally valid).
    This is a lightweight check; for full JSON Schema validation install
    ``jsonschema`` and use ``validate_strict`` below.
    """
    errors: list[str] = []

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = schema.get("required", [])
    for key in required:
        if key not in record:
            errors.append(f"missing_required_field: {key}")

    properties = schema.get("properties", {})
    for key, value in record.items():
        if key not in properties:
            errors.append(f"unknown_field: {key}")
            continue
        prop = properties[key]
        if "const" in prop and value != prop["const"]:
            errors.append(f"invalid_const_{key}: expected {prop['const']!r}, got {value!r}")

    return errors


def validate_strict(record: Mapping[str, Any]) -> list[str]:
    """Full JSON Schema validation using the ``jsonschema`` library.

    Falls back to structural validation when ``jsonschema`` is unavailable.
    """
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        return validate_json_schema(record)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [err.message for err in validator.iter_errors(record)]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record",
        type=Path,
        help="Path to a gate evidence JSON record.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use full jsonschema validation when available.",
    )
    args = parser.parse_args(argv)

    # ── load ───────────────────────────────────────────────────────────
    try:
        raw = args.record.read_text(encoding="utf-8")
    except OSError as exc:
        print(json.dumps({"status": "invalid", "error": f"cannot_read: {exc}"}))
        return 2

    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "invalid", "error": f"invalid_json: {exc}"}))
        return 2

    if not isinstance(record, dict):
        print(json.dumps({"status": "invalid", "error": "record_must_be_json_object"}))
        return 2

    # ── schema validation ──────────────────────────────────────────────
    schema_errors = validate_strict(record) if args.strict else validate_json_schema(record)
    if schema_errors:
        print(json.dumps({"status": "invalid", "error": "schema_violations", "details": schema_errors}))
        return 2

    # ── semantic validation ────────────────────────────────────────────
    try:
        result = validate_gate_evidence(record)
    except GateValidationError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}))
        return 2

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
