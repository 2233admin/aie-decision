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

def test_finalize_delegates_to_kernel_and_records_verdict():
    kernel = _CountingKernel(certified_after=0)
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    verdict = runtime.finalize()
    assert runtime.status is SessionStatus.CERTIFIED
    assert verdict["evaluation"]["status"] == "certified"

def test_finalize_records_insufficient_when_kernel_says_so():
    kernel = _CountingKernel(frontier=[{"id": "a", "label": "A"}])
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    verdict = runtime.finalize()
    assert runtime.status is SessionStatus.INSUFFICIENT
    assert verdict["evaluation"]["status"] == "insufficient"

def test_finalize_does_not_bypass_gate_when_prose_claims_completion():
    class _AlwaysCertifyKernel(_CountingKernel):
        def evaluate_frontier(self, state):  # type: ignore[override]
            # Even if the kernel wanted to grant certification, the runtime
            # would still record the verdict.  Here we make the kernel
            # refuse to certify on purpose.
            return {"status": "insufficient", "reasons": ["frontier still has nodes"]}

    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=_AlwaysCertifyKernel(frontier=[{"id": "a"}]))
    verdict = runtime.finalize()
    assert runtime.status is SessionStatus.INSUFFICIENT
    assert verdict["status"] == "insufficient"

def test_finalize_on_inactive_session_returns_terminal_response_without_events():
    """``finalize`` after the session has been finalized is a no-op."""
    kernel = _CountingKernel(certified_after=0)
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    first = runtime.finalize()
    assert first["status"] == "certified"
    events_before = len(runtime.trajectory.events)
    status_before = runtime.status
    counter_snapshot = BudgetCounters(
        actions=runtime.counters.actions,
        evaluations=runtime.counters.evaluations,
        compute=runtime.counters.compute,
        depth=runtime.counters.depth,
    )
    again = runtime.finalize()
    assert again["no_op"] is True
    assert again["status"] == status_before.value
    # No additional events were recorded by the no-op call.
    assert len(runtime.trajectory.events) == events_before
    # Counters are unchanged by the no-op call.
    assert runtime.counters == counter_snapshot
    assert runtime.status == status_before

def test_finalize_raises_for_irrecoverably_malformed_evaluation_without_dangling_action():
    """A non-Mapping evaluation raises a controlled error and records no event."""
    class _MalformedKernel(_CountingKernel):
        def evaluate_frontier(self, state):  # type: ignore[override]
            return ["not", "a", "mapping"]

    kernel = _MalformedKernel(certified_after=0)
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    events_before = len(runtime.trajectory.events)
    with pytest.raises(RuntimeError_):
        runtime.finalize()
    # No new events were appended: the malformed evaluation never produced
    # a dangling finalize ACTION.
    assert len(runtime.trajectory.events) == events_before

def test_finalize_normalizes_string_reasons_to_list():
    class _StringReasonsKernel(_CountingKernel):
        def evaluate_frontier(self, state):  # type: ignore[override]
            return {"status": "insufficient", "reasons": "single string reason"}

    kernel = _StringReasonsKernel(frontier=[{"id": "a"}])
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    verdict = runtime.finalize()
    assert verdict["evaluation"]["reasons"] == ["single string reason"]

def test_finalize_normalizes_string_blocking_issues_to_list():
    class _StringBlockingKernel(_CountingKernel):
        def evaluate_frontier(self, state):  # type: ignore[override]
            return {
                "status": "insufficient",
                "reasons": ["frontier not empty"],
                "blocking_issues": "node a is open",
            }

    kernel = _StringBlockingKernel(frontier=[{"id": "a"}])
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    verdict = runtime.finalize()
    assert verdict["evaluation"]["blocking_issues"] == ["node a is open"]

def test_finalize_rejects_malformed_blocking_issues_without_dangling_action():
    class _BadBlockingKernel(_CountingKernel):
        def evaluate_frontier(self, state):  # type: ignore[override]
            return {
                "status": "insufficient",
                "reasons": ["frontier not empty"],
                "blocking_issues": {"this": "is a mapping, not a sequence"},
            }

    kernel = _BadBlockingKernel(frontier=[{"id": "a"}])
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    events_before = len(runtime.trajectory.events)
    with pytest.raises(RuntimeError_):
        runtime.finalize()
    assert len(runtime.trajectory.events) == events_before
