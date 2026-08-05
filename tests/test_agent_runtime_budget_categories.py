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

class _CategorisedKernel:
    """A kernel whose action_specs surface explicit ``category`` fields."""

    def initial_state(self, question: str) -> dict[str, Any]:
        return {"question": question, "depth": 0}

    def action_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": "structural_action", "category": "structural"},
            {"name": "evaluation_action", "category": "evaluation"},
            {"name": "control_action", "category": "control"},
        ]

    def validate(self, action, payload, state):
        return []

    def execute(self, action, payload, state):
        new_state = dict(state)
        new_state["depth"] = new_state.get("depth", 0) + (1 if action == "structural_action" else 0)
        return new_state

    def legal_next_actions(self, state):
        return ["structural_action", "evaluation_action", "control_action"]

    def active_frontier(self, state):
        return []

    def evaluate_frontier(self, state):
        return {"status": "insufficient", "reasons": []}

def test_evaluation_actions_derived_from_kernel_specs():
    kernel = _CategorisedKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    assert runtime._evaluation_actions() == {"evaluation_action"}

def test_depth_actions_derived_from_kernel_specs():
    kernel = _CategorisedKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    assert runtime._depth_actions() == {"structural_action"}

def test_depth_budget_uses_derived_categories():
    """The depth budget must gate the kernel's structural action, not a hard-coded name."""
    kernel = _CategorisedKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_depth=0),
    )
    # The start action itself exhausts the depth budget (0 + 0 reaches 0
    # which is at the limit), so the session is PARTIAL but the
    # rejection code must be ``budget_exhausted`` (driven by depth
    # category metadata), not a hard-coded name like ``expand``.
    result = runtime.apply(action="structural_action", payload={})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "budget_exhausted"

def test_evaluation_budget_uses_derived_categories():
    """The evaluation budget must gate the kernel's evaluation action."""
    kernel = _CategorisedKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_evaluations=0),
    )
    result = runtime.apply(action="evaluation_action", payload={})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "budget_exhausted"

def test_action_categories_fallback_for_kernel_without_categories():
    """When the kernel omits ``category`` metadata, the runtime uses name-based fallbacks."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    # The CountingKernel does surface categories, so verify the
    # fallback path with a kernel that omits them entirely.
    class _NoCategoryKernel(_CountingKernel):
        def action_specs(self):  # type: ignore[override]
            return [{"name": "expand"}, {"name": "evaluate"}, {"name": "finalize"}]

    no_cat = _NoCategoryKernel()
    fallback_runtime = AgentRuntime.start(
        session_id="s1", question="q", kernel=no_cat,
    )
    assert "finalize" in fallback_runtime._evaluation_actions()
    assert "expand" in fallback_runtime._depth_actions()
