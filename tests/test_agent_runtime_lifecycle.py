"""Tests for the agent decomposition runtime."""

from __future__ import annotations

import json

from typing import Any, Mapping

import pytest

from aie_decision.agent_runtime import (
    ActionResult,
    AgentRuntime,
    BudgetCounters,
    BudgetPolicy,
    KernelProtocol,
    PROTOCOL_VERSION,
    RuntimeError_,
    SCHEMA_VERSION,
    SessionStatus,
)

from aie_decision.trajectory import EventStatus

class _CountingKernel:
    """A deterministic kernel with explicit frontier certification behaviour."""

    def __init__(
        self,
        *,
        certified_after: int = 0,
        frontier: list[dict[str, Any]] | None = None,
        legal_actions: list[str] | None = None,
    ) -> None:
        self._certified_after = certified_after
        self._base_frontier = list(frontier or [])
        self._legal_actions = list(legal_actions or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def initial_state(self, question: str) -> dict[str, Any]:
        return {
            "question": question,
            "expansions": [],
            "estimates": [],
            "depth": 0,
            "frontier": list(self._base_frontier),
        }

    def action_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": "expand", "category": "structural", "required_fields": ["node_id", "children"]},
            {"name": "estimate", "category": "measurement", "required_fields": ["node_id", "value", "unit"]},
            {"name": "evaluate", "category": "evaluation", "required_fields": []},
            {"name": "rollback", "category": "control", "required_fields": ["target_sequence"]},
            {"name": "finalize", "category": "frontier", "required_fields": []},
        ]

    def validate(self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if action == "expand":
            if not isinstance(payload.get("children"), list) or not payload.get("children"):
                issues.append({"code": "missing_field", "path": "$.children", "message": "children required"})
        if action == "estimate":
            for field_name in ("node_id", "value", "unit"):
                if field_name not in payload:
                    issues.append({"code": "missing_field", "path": f"$.{field_name}", "message": f"{field_name} required"})
        return issues

    def execute(self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((action, dict(payload)))
        new_state = {key: value for key, value in state.items()}
        expansions = list(new_state.get("expansions", []))
        estimates = list(new_state.get("estimates", []))
        depth = int(new_state.get("depth", 0))
        frontier = list(new_state.get("frontier", []))
        if action == "expand":
            expansions.append({"node_id": payload["node_id"], "children": payload["children"]})
            new_state["depth"] = depth + 1
            frontier = [
                {"id": child.get("id", f"{payload['node_id']}::{i}"), "label": child.get("label", "")}
                for i, child in enumerate(payload["children"])
                if isinstance(child, Mapping)
            ]
        elif action == "estimate":
            estimates.append({"node_id": payload["node_id"], "value": payload["value"], "unit": payload["unit"]})
            frontier = [node for node in frontier if node.get("id") != payload.get("node_id")]
        elif action == "rollback":
            # Drop the last accepted expansion; in a real kernel this would
            # undo the target action's effect.  Here we simply roll back the
            # most recent expansion.
            if expansions:
                expansions.pop()
            new_state["depth"] = max(0, depth - 1)
        new_state["expansions"] = expansions
        new_state["estimates"] = estimates
        new_state["frontier"] = frontier
        return new_state

    def legal_next_actions(self, state: Mapping[str, Any]) -> list[str]:
        if self._legal_actions:
            return list(self._legal_actions)
        if not state.get("frontier"):
            return ["finalize"]
        return ["expand", "estimate", "finalize"]

    def active_frontier(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return list(state.get("frontier", []))

    def evaluate_frontier(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if not state.get("frontier") and len(self.calls) >= self._certified_after:
            return {"status": "certified", "reasons": ["frontier empty"]}
        if state.get("frontier"):
            return {"status": "insufficient", "reasons": ["frontier not empty"], "blocking_issues": []}
        return {"status": "ineligible", "reasons": ["no expansions evaluated"]}

def test_start_records_start_event_and_returns_initial_state():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="how many?", kernel=kernel)
    assert runtime.status is SessionStatus.ACTIVE
    assert runtime.state["question"] == "how many?"
    assert runtime.trajectory.events[0].action == "start"
    assert runtime.trajectory.events[1].status is EventStatus.ACCEPTED

def test_start_without_question_raises():
    with pytest.raises(ValueError):
        AgentRuntime.start(session_id="s1", question="", kernel=_CountingKernel())

def test_discover_returns_action_specs_and_legal_actions():
    kernel = _CountingKernel(legal_actions=["expand", "estimate"])
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    payload = runtime.discover()
    assert payload["session_id"] == "s1"
    assert any(spec["name"] == "expand" for spec in payload["action_specs"])
    assert "expand" in payload["legal_next_actions"]

def test_apply_accepts_valid_action_and_updates_state():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]},
    )
    assert result.accepted
    assert result.status == "accepted"
    assert result.sequence == 3
    assert runtime.state["expansions"][0]["node_id"] == "root"

def test_apply_rejects_invalid_action_with_structured_error():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(action="expand", payload={"node_id": "root"})  # missing children
    assert not result.accepted
    assert result.error is not None
    assert any(issue["code"] == "missing_field" for issue in result.error["issues"])
    # Rejection must not mutate state
    assert runtime.state.get("expansions") == []

def test_apply_records_rejection_in_trajectory():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(action="expand", payload={"node_id": "root"})
    last_pair = runtime.trajectory.pairs()[-1]
    assert last_pair[0].action == "expand"
    assert last_pair[1].status is EventStatus.REJECTED

def test_apply_rejects_stale_prior_revision():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
        prior_revision="deadbeef",
    )
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "stale_revision"

def test_export_includes_trajectory_state_and_metadata():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(action="expand", payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]})
    document = runtime.export()
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["protocol_version"] == PROTOCOL_VERSION
    assert document["session_id"] == "s1"
    assert document["status"] == "active"
    assert len(document["trajectory"]["events"]) >= 4

def test_inspect_returns_read_only_projection():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    snapshot = runtime.inspect()
    assert snapshot["session_id"] == "s1"
    assert snapshot["state"]["question"] == "q"
    assert "active_frontier" in snapshot

def test_apply_with_no_action_returns_missing_action_error():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(action="")
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "missing_action"

def test_illegal_transition_after_finalize_is_rejected():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.finalize()
    result = runtime.apply(action="expand", payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "session_not_active"

def test_action_result_to_dict_is_json_serialisable():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(action="expand", payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]})
    json.dumps(result.to_dict())

def test_budget_counters_increment_for_rejected_actions():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    initial = runtime.counters.actions
    runtime.apply(action="expand", payload={"node_id": "root"})
    assert runtime.counters.actions == initial + 1

def test_budget_policy_from_dict_ignores_unknown_keys():
    policy = BudgetPolicy.from_dict({"max_actions": 3, "max_depth": None, "extra": "ignored"})
    assert policy.max_actions == 3
    assert policy.max_depth is None

def test_budget_policy_from_dict_rejects_non_int():
    with pytest.raises(ValueError):
        BudgetPolicy.from_dict({"max_actions": "3"})
