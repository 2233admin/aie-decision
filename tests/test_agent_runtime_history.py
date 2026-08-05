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

class _FailingTrajectory:
    """A stand-in trajectory that fails on the first ``record_action`` call.

    The runtime must leave the in-memory state and counters unchanged when
    recording fails.  Subsequent calls (used to record the rejection
    bookkeeping) succeed so the test can observe the returned
    ActionResult.
    """

    def __init__(self, real_trajectory) -> None:
        self._real = real_trajectory
        self.actions_recorded = 0
        self.results_recorded = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def record_action(self, **kwargs):
        self.actions_recorded += 1
        if self.actions_recorded == 1:
            raise RuntimeError("synthetic trajectory failure on record_action")
        return self._real.record_action(**kwargs)

    def record_result(self, **kwargs):
        self.results_recorded += 1
        return self._real.record_result(**kwargs)

def test_rollback_removes_prior_state_effect_but_keeps_history():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    target_sequence = 3  # the expand action's sequence
    state_before_rollback = dict(runtime.state)
    result = runtime.apply(
        action="rollback",
        payload={"target_sequence": target_sequence},
        prior_revision=runtime.trajectory.last_revision(),
        rollback_target_sequence=target_sequence,
    )
    assert result.accepted
    assert runtime.trajectory.is_live(target_sequence) is False
    # Current state effect is reversed.
    assert runtime.state.get("expansions") == []
    # History still shows the original action and its result.
    actions = [event.action for event in runtime.trajectory.events if event.kind.name == "ACTION"]
    assert "expand" in actions and "rollback" in actions

def test_rollback_targeting_start_is_rejected():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(
        action="rollback",
        payload={"target_sequence": 1},
        rollback_target_sequence=1,
    )
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "invalid_rollback_target"

def test_rollback_with_unknown_target_is_rejected():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    result = runtime.apply(
        action="rollback",
        payload={"target_sequence": 99},
        rollback_target_sequence=99,
    )
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "invalid_rollback_target"

def test_rollback_projection_rebuilds_state_without_kernel_rollback():
    """Rollback must rebuild state from initial_state + live non-rollback actions.

    A kernel whose rollback execute is a no-op must still see the target
    mutation disappear from the projection, and replay() must equal current
    state after rollback.
    """

    class _NoOpRollbackKernel(_CountingKernel):
        """Kernel that fails if runtime projection delegates rollback."""

        def execute(self, action, payload, state):
            if action == "rollback":
                raise AssertionError("runtime must not execute rollback in the kernel")
            return super().execute(action, payload, state)

    kernel = _NoOpRollbackKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert runtime.state["expansions"][0]["node_id"] == "root"
    target_seq = 3
    result = runtime.apply(
        action="rollback",
        payload={"target_sequence": target_seq},
        prior_revision=runtime.trajectory.last_revision(),
        rollback_target_sequence=target_seq,
    )
    assert result.accepted
    # The target expansion must not appear, and the kernel rollback branch
    # above must never have been invoked.
    assert runtime.state.get("expansions") == []
    # replay() must produce a state identical to the current projection.
    replay = runtime.replay()
    assert replay["verdict"] == "match"
    assert replay["reconstructed_state"].get("expansions") == []
    assert (
        replay["reconstructed_state"].get("depth")
        == runtime.state.get("depth")
    )

def test_apply_rejects_finalize_records_rejection_event():
    """A rejected finalize through apply() leaves a REJECTED pair in the log."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    events_before = len(runtime.trajectory.events)
    result = runtime.apply(action="finalize", payload={})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "use_finalize_method"
    # One ACTION + one RESULT appended.
    assert len(runtime.trajectory.events) == events_before + 2
    last_pair = runtime.trajectory.pairs()[-1]
    assert last_pair[0].action == "finalize"
    assert last_pair[1].status is EventStatus.REJECTED

def test_apply_recording_failure_leaves_state_and_candidate_counters_unchanged():
    """When ``record_action`` fails, the candidate action's state effect is not committed.

    The runtime must not apply the candidate state or the candidate
    counters (depth, evaluations, compute) when the trajectory rejects
    the event.  The action counter is allowed to advance by one for
    the rejection bookkeeping.
    """
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    # First call uses the real trajectory; second one installs a failing wrapper.
    runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    state_before = dict(runtime.state)
    depth_before = runtime.counters.depth
    compute_before = runtime.counters.compute
    real_trajectory = runtime.trajectory
    failing = _FailingTrajectory(real_trajectory)
    # Swap the trajectory in the dataclass field.
    object.__setattr__(runtime, "trajectory", failing)
    result = runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    # Restore so we can inspect state.
    object.__setattr__(runtime, "trajectory", real_trajectory)
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "recording_failed"
    # The candidate action never made it to in-memory state.
    assert runtime.state == state_before
    # The candidate counters (depth, compute) are not committed.
    assert runtime.counters.depth == depth_before
    assert runtime.counters.compute == compute_before

def test_apply_depth_uses_candidate_state_not_committed_state():
    """The depth counter after apply equals the depth surfaced by the candidate state."""
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(
        action="expand",
        payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]},
    )
    assert runtime.counters.depth == 1
    runtime.apply(
        action="expand",
        payload={"node_id": "a", "children": [{"id": "b", "label": "B"}]},
    )
    assert runtime.counters.depth == 2
