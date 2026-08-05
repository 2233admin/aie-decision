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

def test_action_budget_exhaustion_rejects_subsequent_actions():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_actions=2),
    )
    # start already used 1; one accepted action leaves 1 left.
    first = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert first.accepted
    second = runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    assert not second.accepted
    assert second.error["issues"][0]["code"] == "budget_exhausted"
    assert "actions" in second.budget_remaining

def test_action_budget_rejection_cannot_reach_the_kernel_or_mutate_state():
    """Adversarial: a valid-looking action after exhaustion stays inert."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="budget-adversarial",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_actions=2),
    )
    accepted = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert accepted.accepted is True
    state_at_exhaustion = dict(runtime.state)
    calls_at_exhaustion = list(kernel.calls)

    rejected = runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "poison", "label": "P"}]},
    )

    assert rejected.accepted is False
    assert rejected.error["issues"][0]["code"] == "budget_exhausted"
    assert runtime.state == state_at_exhaustion
    assert kernel.calls == calls_at_exhaustion

def test_apply_rejects_finalize_with_structured_message():
    """``apply`` must reject ``finalize`` and direct callers to the dedicated method."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(action="finalize", payload={})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "use_finalize_method"
    # The trajectory records a REJECTED event so the misuse is auditable.
    last_pair = runtime.trajectory.pairs()[-1]
    assert last_pair[0].action == "finalize"
    assert last_pair[1].status is EventStatus.REJECTED

def test_evaluation_budget_exhaustion_rejects_finalize():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_evaluations=0),
    )
    result = runtime.apply(action="evaluate", payload={})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "budget_exhausted"
    assert result.error["issues"][0]["path"] == "$.evaluations"

def test_compute_budget_exhaustion_rejects_expensive_actions():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_compute=1),
    )
    result = runtime.apply(action="expand", payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "budget_exhausted"

def test_depth_budget_exhaustion_rejects_expansion():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_depth=0),
    )
    result = runtime.apply(action="expand", payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "budget_exhausted"
    assert result.error["issues"][0]["path"] == "$.depth"

def test_budget_exhausted_status_transitions_to_partial_on_finalize():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_actions=2),
    )
    # Start uses one of the two actions; finalize uses the second and
    # exhausts the budget.  The certified verdict is demoted to PARTIAL.
    verdict = runtime.finalize()
    assert runtime.status is SessionStatus.PARTIAL
    assert verdict["status"] == "partial"
    assert "reasons" in verdict["evaluation"]

def test_exact_action_budget_accepts_at_boundary():
    """Accepting the action that reaches exactly max_actions is allowed."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_actions=2),
    )
    # start (1) + first expand (2) → exactly at limit; next expand rejected.
    first = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert first.accepted
    assert runtime.counters.actions == 2
    second = runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    assert not second.accepted
    assert second.error["issues"][0]["code"] == "budget_exhausted"

def test_exact_evaluation_budget_accepts_at_boundary():
    """Accepting the action that reaches exactly max_evaluations is allowed."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_evaluations=1),
    )
    # ``finalize`` is the dedicated API.  It consumes the evaluation
    # budget and the boundary test confirms the inclusive check.
    verdict = runtime.finalize()
    assert verdict["status"] in {"insufficient", "partial", "certified"}
    assert runtime.counters.evaluations == 1
    assert runtime.status is SessionStatus.PARTIAL

def test_exact_compute_budget_accepts_at_boundary():
    """Accepting the action whose cost reaches exactly max_compute is allowed."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_compute=2),
    )
    # start costs 1 compute; expand costs 1 → total 2, exactly at boundary.
    result = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert result.accepted
    assert runtime.counters.compute == 2
    next_ = runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    assert not next_.accepted
    assert next_.error["issues"][0]["code"] == "budget_exhausted"

def test_exact_depth_budget_accepts_at_boundary():
    """Accepting the expand that reaches exactly max_depth is allowed."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_depth=1),
    )
    result = runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert result.accepted
    assert runtime.counters.depth == 1
    next_ = runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    assert not next_.accepted
    assert next_.error["issues"][0]["code"] == "budget_exhausted"
