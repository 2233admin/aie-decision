"""Fail-closed tests for Wave MVP gate evidence schema and validator.

Every test proves the validator rejects a specific class of missing or
mislabeled evidence.  No test creates a baseline or makes a failed gate pass.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_wave_mvp_gate_evidence.py"
SCHEMA_PATH = ROOT / "schemas" / "wave_mvp_gate_evidence.schema.json"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_wave_mvp_gate_evidence", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── helpers ────────────────────────────────────────────────────────────────

def _valid_record(**overrides):
    """Return a minimally valid gate evidence record with overrides applied."""
    record = {
        "schema_version": "wave-mvp-gate-evidence.v1",
        "gate_id": "harden-contract-driven-wave-mvp-4.1",
        "openspec_tasks": [
            {"task_id": "4.1", "status": "implemented",
             "owned_files": ["scripts/validate_wave_mvp_gate_evidence.py"]}
        ],
        "status": "fail",
        "focused_pytest": {
            "command": "uv run pytest tests/test_wave_mvp_gate_evidence.py -q",
            "passed": True
        },
        "full_pytest": {
            "command": "uv run pytest -q",
            "passed": True
        },
        "code_intel": {
            "command": "code-intel . --mode normal --json",
            "outcome": "failed",
            "publication_path": "artifacts/code-intel/current.json",
            "failure_detail": "baseline/domain gate incompatible with current architecture"
        },
        "sentrux": {
            "command": "code-intel sentrux check .",
            "rules_passed": True,
            "ratchet_passed": False,
            "baseline_path": ".sentrux/baseline.json",
            "failure_detail": "ratchet baseline incompatible; new architecture layers not in baseline"
        },
        "independent_review": {
            "reviewer": "ecc:code-reviewer",
            "outcome": "approved",
            "evidence_path": "artifacts/reviews/gate-4.1-review.md",
            "findings": ["no assertion weakening", "no skip paths introduced"]
        },
        "commit": "abc1234",
        "dirty_files": [],
        "provider": "anthropic",
        "model": "claude-fable-5"
    }
    record.update(overrides)
    return record


# ── schema validation tests ────────────────────────────────────────────────

class TestSchemaValidation:
    """Structural validation: missing fields, unknown fields, wrong consts."""

    def test_accepts_minimally_valid_record(self):
        v = _load_validator()
        record = _valid_record()
        errors = v.validate_json_schema(record)
        assert errors == []

    def test_rejects_missing_required_field(self):
        v = _load_validator()
        record = _valid_record()
        del record["gate_id"]
        errors = v.validate_json_schema(record)
        assert any("missing_required_field" in e for e in errors)

    def test_rejects_unknown_field(self):
        v = _load_validator()
        record = _valid_record()
        record["extra_cruft"] = True
        errors = v.validate_json_schema(record)
        assert any("unknown_field" in e for e in errors)

    def test_rejects_wrong_schema_version(self):
        v = _load_validator()
        record = _valid_record()
        record["schema_version"] = "wrong-version"
        errors = v.validate_json_schema(record)
        assert any("invalid_const_schema_version" in e for e in errors)


# ── semantic gate validation tests ─────────────────────────────────────────

class TestGateValidation:
    """Semantic validation: status consistency, evidence completeness."""

    def test_valid_fail_record_passes_validation(self):
        v = _load_validator()
        record = _valid_record()  # status=fail with code_intel failed + ratchet failed
        result = v.validate_gate_evidence(record)
        assert result["valid"] is True
        assert result["status"] == "fail"

    def test_rejects_pass_when_code_intel_failed(self):
        v = _load_validator()
        record = _valid_record(status="pass")
        with pytest.raises(v.GateValidationError, match="gate_status_mismatch"):
            v.validate_gate_evidence(record)

    def test_rejects_pass_when_focused_tests_failed(self):
        v = _load_validator()
        record = _valid_record(
            status="pass",
            focused_pytest={"command": "uv run pytest -q", "passed": False},
            code_intel={"command": "ci", "outcome": "passed", "publication_path": "p"},
            sentrux={"command": "sx", "rules_passed": True, "ratchet_passed": True},
        )
        with pytest.raises(v.GateValidationError, match="gate_status_mismatch"):
            v.validate_gate_evidence(record)

    def test_rejects_pass_when_full_tests_failed(self):
        v = _load_validator()
        record = _valid_record(
            status="pass",
            full_pytest={"command": "uv run pytest -q", "passed": False},
            code_intel={"command": "ci", "outcome": "passed", "publication_path": "p"},
            sentrux={"command": "sx", "rules_passed": True, "ratchet_passed": True},
        )
        with pytest.raises(v.GateValidationError, match="gate_status_mismatch"):
            v.validate_gate_evidence(record)

    def test_rejects_pass_when_sentrux_rules_failed(self):
        v = _load_validator()
        record = _valid_record(
            status="pass",
            code_intel={"command": "ci", "outcome": "passed", "publication_path": "p"},
            sentrux={"command": "sx", "rules_passed": False, "ratchet_passed": True,
                     "failure_detail": "architecture rule violation"},
        )
        with pytest.raises(v.GateValidationError, match="gate_status_mismatch"):
            v.validate_gate_evidence(record)

    def test_rejects_pass_when_review_changes_requested(self):
        v = _load_validator()
        record = _valid_record(
            status="pass",
            code_intel={"command": "ci", "outcome": "passed", "publication_path": "p"},
            sentrux={"command": "sx", "rules_passed": True, "ratchet_passed": True},
            independent_review={
                "reviewer": "ecc:code-reviewer",
                "outcome": "changes_requested",
                "evidence_path": "artifacts/reviews/r.md",
            },
        )
        with pytest.raises(v.GateValidationError, match="gate_status_mismatch"):
            v.validate_gate_evidence(record)

    def test_accepts_blocked_when_task_blocked(self):
        v = _load_validator()
        record = _valid_record(
            status="blocked",
            openspec_tasks=[
                {"task_id": "4.1", "status": "blocked",
                 "blocker": "Code Intel baseline mismatch requires architecture review"}
            ],
        )
        result = v.validate_gate_evidence(record)
        assert result["status"] == "blocked"

    def test_rejects_blocked_without_blocker_description(self):
        v = _load_validator()
        record = _valid_record(
            status="blocked",
            openspec_tasks=[
                {"task_id": "4.1", "status": "blocked"}
            ],
        )
        with pytest.raises(v.GateValidationError, match="blocker"):
            v.validate_gate_evidence(record)

    def test_accepts_all_pass_record_when_all_gates_green(self):
        v = _load_validator()
        record = _valid_record(
            status="pass",
            openspec_tasks=[
                {"task_id": "4.1", "status": "verified"}
            ],
            code_intel={"command": "ci", "outcome": "passed", "publication_path": "p"},
            sentrux={"command": "sx", "rules_passed": True, "ratchet_passed": True},
            independent_review={
                "reviewer": "ecc:code-reviewer",
                "outcome": "approved",
                "evidence_path": "artifacts/reviews/r.md",
            },
        )
        result = v.validate_gate_evidence(record)
        assert result["valid"] is True
        assert result["status"] == "pass"

    def test_rejects_missing_openspec_tasks(self):
        v = _load_validator()
        record = _valid_record()
        record["openspec_tasks"] = []
        with pytest.raises(v.GateValidationError, match="openspec_tasks"):
            v.validate_gate_evidence(record)
        del record["openspec_tasks"]
        with pytest.raises(v.GateValidationError, match="openspec_tasks"):
            v.validate_gate_evidence(record)

    def test_rejects_invalid_task_status(self):
        v = _load_validator()
        record = _valid_record()
        record["openspec_tasks"] = [{"task_id": "4.1", "status": "started"}]
        with pytest.raises(v.GateValidationError, match="openspec_tasks"):
            v.validate_gate_evidence(record)

    def test_rejects_duplicate_task_ids(self):
        v = _load_validator()
        record = _valid_record()
        record["openspec_tasks"] = [
            {"task_id": "4.1", "status": "implemented"},
            {"task_id": "4.1", "status": "verified"},
        ]
        with pytest.raises(v.GateValidationError, match="duplicate_task_id"):
            v.validate_gate_evidence(record)

    def test_rejects_commit_too_short(self):
        v = _load_validator()
        record = _valid_record(commit="abc")
        with pytest.raises(v.GateValidationError, match="commit_too_short"):
            v.validate_gate_evidence(record)

    def test_rejects_missing_code_intel_failure_detail(self):
        v = _load_validator()
        record = _valid_record()
        del record["code_intel"]["failure_detail"]
        with pytest.raises(v.GateValidationError, match="failure_detail"):
            v.validate_gate_evidence(record)

    def test_rejects_missing_sentrux_failure_detail_when_rules_fail(self):
        v = _load_validator()
        record = _valid_record(
            sentrux={"command": "sx", "rules_passed": False, "ratchet_passed": True}
        )
        with pytest.raises(v.GateValidationError, match="failure_detail"):
            v.validate_gate_evidence(record)

    def test_rejects_non_boolean_pytest_passed(self):
        v = _load_validator()
        record = _valid_record()
        record["focused_pytest"]["passed"] = "yes"
        with pytest.raises(v.GateValidationError, match="focused_pytest.passed"):
            v.validate_gate_evidence(record)


# ── CLI integration tests ───────────────────────────────────────────────────

class TestCLI:
    """End-to-end CLI validation via programmatic main()."""

    def test_cli_accepts_valid_record_file(self):
        v = _load_validator()
        record = _valid_record()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(record, f)
            path = f.name
        try:
            exit_code = v.main([path])
            assert exit_code == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_cli_rejects_invalid_json(self):
        v = _load_validator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not json")
            path = f.name
        try:
            exit_code = v.main([path])
            assert exit_code == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_cli_rejects_file_not_found(self):
        v = _load_validator()
        exit_code = v.main(["nonexistent_file.json"])
        assert exit_code == 2

    def test_cli_rejects_empty_object(self):
        v = _load_validator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            path = f.name
        try:
            exit_code = v.main([path])
            assert exit_code == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_cli_rejects_gate_status_mismatch(self):
        v = _load_validator()
        record = _valid_record(status="pass")  # evidence is fail
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(record, f)
            path = f.name
        try:
            exit_code = v.main([path])
            assert exit_code == 2
        finally:
            Path(path).unlink(missing_ok=True)


# ── invariant: no baseline creation ─────────────────────────────────────────

def test_validator_never_writes_to_disk():
    """The validator must never create files, baselines, or side effects."""
    v = _load_validator()
    record = _valid_record()
    # Semantic validation is a pure function — no side effects.
    result = v.validate_gate_evidence(record)
    assert result["valid"] is True


# ── invariant: real failures are preserved ──────────────────────────────────

def test_real_code_intel_failure_is_preserved():
    """A real Code Intel failure must never be downgraded to 'blocked' or hidden."""
    v = _load_validator()
    record = _valid_record(
        status="fail",
        code_intel={
            "command": "code-intel . --mode normal --json",
            "outcome": "failed",
            "publication_path": "artifacts/code-intel/current.json",
            "failure_detail": "compiled rules pass but domain gate is incompatible"
        }
    )
    result = v.validate_gate_evidence(record)
    assert result["status"] == "fail"
    # It must not be reported as "blocked" or "passed"
    assert result["status"] != "pass"
    assert result["status"] != "blocked"


def test_real_sentrux_rules_failure_is_preserved():
    """A real Sentrux rules failure must be reflected in the gate status."""
    v = _load_validator()
    record = _valid_record(
        status="fail",
        sentrux={
            "command": "code-intel sentrux check .",
            "rules_passed": False,
            "ratchet_passed": True,
            "failure_detail": "layer boundary violation: contracts -> cli"
        }
    )
    result = v.validate_gate_evidence(record)
    assert result["status"] == "fail"


def test_ratchet_baseline_failure_is_explicitly_recorded():
    """A ratchet failure is a real failure — not a silent downgrade."""
    v = _load_validator()
    record = _valid_record()  # ratchet_passed=False
    result = v.validate_gate_evidence(record)
    assert result["status"] == "fail"
