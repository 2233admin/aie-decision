"""Tests for the versioned JSON CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aie_decision.agent_cli import build_default_kernel, main
from aie_decision.agent_runtime import PROTOCOL_VERSION, SCHEMA_VERSION
from aie_decision.trajectory import EventStatus


def _run_cli(args, *, input_text=None):
    """Invoke the CLI in-process and return (returncode, stdout, stderr)."""

    from io import StringIO

    stdin = StringIO(input_text or "")
    stdout = StringIO()
    stderr = StringIO()
    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
    try:
        code = main(args)
    finally:
        sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr
    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_lists_action_specs_and_legal_actions():
    code, out, _ = _run_cli(["discover"])
    assert code == 0
    payload = json.loads(out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["protocol_version"] == PROTOCOL_VERSION
    names = {spec["name"] for spec in payload["action_specs"]}
    assert {"expand", "estimate", "rollback", "finalize"}.issubset(names)
    assert "expand" in payload["legal_next_actions"]


# ---------------------------------------------------------------------------
# Start / apply / inspect
# ---------------------------------------------------------------------------


def test_start_creates_session_document_with_trajectory(tmp_path):
    session_path = tmp_path / "session.json"
    code, out, _ = _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    assert code == 0
    payload = json.loads(out)
    assert payload["session_id"] == "s1"
    document = json.loads(session_path.read_text(encoding="utf-8"))
    assert document["session_id"] == "s1"
    assert document["question"] == "How many widgets?"
    assert document["trajectory"]["events"][0]["action"] == "start"


def test_apply_records_action_and_persists_session(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    payload = {"action": "expand", "payload": {"node_id": "root", "children": [{"id": "a", "label": "A"}]}}
    input_path = tmp_path / "apply.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    code, out, _ = _run_cli([
        "apply",
        "--session", str(session_path),
        "--input", str(input_path),
    ])
    assert code == 0
    result = json.loads(out)
    assert result["accepted"] is True
    document = json.loads(session_path.read_text(encoding="utf-8"))
    actions = [event["action"] for event in document["trajectory"]["events"] if event["kind"] == "action"]
    assert actions == ["start", "expand"]


def test_apply_via_action_flag(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    input_path = tmp_path / "payload.json"
    input_path.write_text(json.dumps({"node_id": "root", "children": [{"id": "a", "label": "A"}]}), encoding="utf-8")
    code, out, _ = _run_cli([
        "apply",
        "--session", str(session_path),
        "--action", "expand",
        "--input", str(input_path),
    ])
    assert code == 0
    result = json.loads(out)
    assert result["accepted"] is True


def test_apply_rejects_invalid_payload_and_persists_rejection(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    payload = {"action": "expand", "payload": {"node_id": "root"}}
    input_path = tmp_path / "apply.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    code, out, _ = _run_cli([
        "apply",
        "--session", str(session_path),
        "--input", str(input_path),
    ])
    assert code == 2
    result = json.loads(out)
    assert result["accepted"] is False
    assert any(issue["code"] == "missing_field" for issue in result["error"]["issues"])
    document = json.loads(session_path.read_text(encoding="utf-8"))
    statuses = [event["status"] for event in document["trajectory"]["events"] if event["kind"] == "result"]
    assert EventStatus.REJECTED.value in statuses


def test_inspect_returns_state(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    code, out, _ = _run_cli(["inspect", "--session", str(session_path)])
    assert code == 0
    payload = json.loads(out)
    assert payload["session_id"] == "s1"
    assert payload["state"]["question"] == "How many widgets?"


def test_finalize_delegates_to_kernel_and_persists_verdict(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    code, out, _ = _run_cli(["finalize", "--session", str(session_path)])
    assert code == 2  # default kernel returns insufficient
    payload = json.loads(out)
    assert payload["status"] == "insufficient"
    document = json.loads(session_path.read_text(encoding="utf-8"))
    assert document["status"] == "insufficient"
    assert document["frontier_evaluation"]["status"] == "insufficient"


def test_replay_round_trip_through_cli(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "How many widgets?",
        "--session", str(session_path),
    ])
    payload = {"action": "expand", "payload": {"node_id": "root", "children": [{"id": "a", "label": "A"}]}}
    input_path = tmp_path / "apply.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    _run_cli(["apply", "--session", str(session_path), "--input", str(input_path)])
    code, out, _ = _run_cli(["replay", "--session", str(session_path)])
    assert code == 0
    payload = json.loads(out)
    assert payload["verdict"] == "match"
    assert payload["mismatches"] == []


# ---------------------------------------------------------------------------
# Round-trip / persistence
# ---------------------------------------------------------------------------


def test_session_persists_across_cli_invocations(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "Q?",
        "--session", str(session_path),
    ])
    payload = {"action": "expand", "payload": {"node_id": "root", "children": [{"id": "a", "label": "A"}]}}
    input_path = tmp_path / "apply.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    _run_cli(["apply", "--session", str(session_path), "--input", str(input_path)])
    _run_cli([
        "start",
        "--session-id", "s2",
        "--question", "Q2?",
        "--session", str(tmp_path / "session2.json"),
    ])
    # The original session should still be present and replay correctly.
    code, out, _ = _run_cli(["replay", "--session", str(session_path)])
    assert code == 0
    assert json.loads(out)["verdict"] == "match"


def test_protocol_version_matches_in_all_outputs(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "Q?",
        "--session", str(session_path),
    ])
    payload = {"action": "expand", "payload": {"node_id": "root", "children": [{"id": "a", "label": "A"}]}}
    input_path = tmp_path / "apply.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    _, discover_out, _ = _run_cli(["discover"])
    _, apply_out, _ = _run_cli(["apply", "--session", str(session_path), "--input", str(input_path)])
    _, inspect_out, _ = _run_cli(["inspect", "--session", str(session_path)])
    _, replay_out, _ = _run_cli(["replay", "--session", str(session_path)])
    for output in (discover_out, apply_out, inspect_out, replay_out):
        payload = json.loads(output)
        assert payload["protocol_version"] == PROTOCOL_VERSION
        assert payload["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_json_input_returns_safe_error(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "Q?",
        "--session", str(session_path),
    ])
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    code, out, err = _run_cli([
        "apply",
        "--session", str(session_path),
        "--input", str(bad),
    ])
    assert code == 2
    assert "Traceback" not in err
    body = json.loads(err)
    assert body["error"] in {"JSONDecodeError", "ValueError"}


def test_apply_without_action_returns_missing_action(tmp_path):
    session_path = tmp_path / "session.json"
    _run_cli([
        "start",
        "--session-id", "s1",
        "--question", "Q?",
        "--session", str(session_path),
    ])
    payload = {"payload": {"node_id": "root", "children": []}}
    input_path = tmp_path / "apply.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    code, out, _ = _run_cli([
        "apply",
        "--session", str(session_path),
        "--input", str(input_path),
    ])
    assert code == 2
    body = json.loads(out)
    assert body["error"]["issues"][0]["code"] == "missing_field"


# ---------------------------------------------------------------------------
# Default kernel shape
# ---------------------------------------------------------------------------


def test_default_kernel_is_provider_free():
    kernel = build_default_kernel()
    state = kernel.initial_state("How many?")
    assert state["question"] == "How many?"
    new_state = kernel.execute("expand", {"node_id": "root", "children": [{"id": "a"}]}, state)
    assert new_state["depth"] == 1
    evaluation = kernel.evaluate_frontier(new_state)
    assert evaluation["status"] == "insufficient"
