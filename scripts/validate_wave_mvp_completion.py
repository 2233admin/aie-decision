"""Fail-closed validator for OpenSpec task completion evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED = (
    "schema_version",
    "change",
    "task_id",
    "owned_files",
    "focused_tests",
    "full_pytest",
    "code_intel",
    "sentrux",
    "commit",
    "dirty_files",
)


def validate_completion(record: Mapping[str, Any]) -> None:
    """Reject incomplete worker evidence instead of inferring completion."""

    missing = [key for key in REQUIRED if key not in record]
    if missing:
        raise ValueError("completion_evidence_missing: " + ",".join(missing))
    if record["schema_version"] != "wave-mvp-completion.v1":
        raise ValueError("unsupported_completion_schema")
    if not record["owned_files"] or not record["commit"]:
        raise ValueError("completion_identity_missing")
    for key in ("focused_tests", "full_pytest"):
        value = record[key]
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise ValueError(f"{key}_not_passed")
    if record["code_intel"].get("outcome") not in {"passed", "failed", "blocked"}:
        raise ValueError("code_intel_outcome_missing")
    sentrux = record["sentrux"]
    if not isinstance(sentrux, Mapping) or not isinstance(sentrux.get("rules_passed"), bool) or not isinstance(sentrux.get("ratchet_passed"), bool):
        raise ValueError("sentrux_evidence_missing")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    try:
        validate_completion(json.loads(args.record.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "valid"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
