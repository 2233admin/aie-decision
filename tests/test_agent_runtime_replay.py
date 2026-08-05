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

def test_replay_reconstructs_identical_state():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    runtime.apply(
        action="estimate",
        payload={"node_id": "a", "value": 10, "unit": "people"},
    )
    original_state = dict(runtime.state)
    replay = runtime.replay()
    assert replay["verdict"] == "match"
    assert replay["mismatches"] == []
    assert replay["reconstructed_state"]["expansions"] == original_state["expansions"]
    assert replay["reconstructed_state"]["estimates"] == original_state["estimates"]

def test_replay_ignores_rolled_back_actions():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    target = 3
    runtime.apply(
        action="rollback",
        payload={"target_sequence": target},
        prior_revision=runtime.trajectory.last_revision(),
        rollback_target_sequence=target,
    )
    replay = runtime.replay()
    assert replay["verdict"] == "match"

def test_replay_applies_later_action_after_earlier_rollback():
    """Replay must include later accepted actions that follow an earlier rollback."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    # expand 1
    runtime.apply(
        action="expand",
        payload={"node_id": "n1", "children": [{"id": "a", "label": "A"}]},
    )
    # expand 2 (will be rolled back)
    runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    # rollback target=expand 2 (sequence 5)
    runtime.apply(
        action="rollback",
        payload={"target_sequence": 5},
        prior_revision=runtime.trajectory.last_revision(),
        rollback_target_sequence=5,
    )
    # expand 3 (after rollback, must appear in replayed state)
    runtime.apply(
        action="expand",
        payload={"node_id": "n1", "children": [{"id": "c", "label": "C"}]},
    )
    live_ids = [e["node_id"] for e in runtime.state["expansions"]]
    replay = runtime.replay()
    assert replay["verdict"] == "match"
    replayed_ids = [e["node_id"] for e in replay["reconstructed_state"]["expansions"]]
    assert replayed_ids == live_ids == ["n1", "n1"]
    assert replay["reconstructed_state"]["depth"] == runtime.state["depth"]

def test_replay_handles_interleaved_rollbacks():
    """Multiple rollbacks interleaved with later actions must all be honoured."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    # expand 1, 2, 3
    runtime.apply(
        action="expand",
        payload={"node_id": "n1", "children": [{"id": "a", "label": "A"}]},
    )
    runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    runtime.apply(
        action="expand",
        payload={"node_id": "b", "children": [{"id": "c", "label": "C"}]},
    )
    # rollback expand 3
    runtime.apply(
        action="rollback",
        payload={"target_sequence": 7},
        prior_revision=runtime.trajectory.last_revision(),
        rollback_target_sequence=7,
    )
    # expand 4
    runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "d", "label": "D"}]},
    )
    # rollback expand 2
    runtime.apply(
        action="rollback",
        payload={"target_sequence": 5},
        prior_revision=runtime.trajectory.last_revision(),
        rollback_target_sequence=5,
    )
    # expand 5
    runtime.apply(
        action="expand",
        payload={"node_id": "n1", "children": [{"id": "e", "label": "E"}]},
    )
    replay = runtime.replay()
    assert replay["verdict"] == "match"
    assert (
        replay["reconstructed_state"]["expansions"]
        == runtime.state["expansions"]
    )
    assert replay["reconstructed_state"]["depth"] == runtime.state["depth"]
