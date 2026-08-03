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
    SCHEMA_VERSION,
    SessionStatus,
)
from aie_decision.trajectory import EventStatus


# ---------------------------------------------------------------------------
# In-memory kernels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------


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


def test_evaluation_budget_exhaustion_rejects_finalize():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_evaluations=0),
    )
    result = runtime.apply(action="finalize", payload={})
    assert not result.accepted
    assert result.error["issues"][0]["code"] == "budget_exhausted"


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


def test_budget_exhausted_status_transitions_to_partial_on_finalize():
    kernel = _CountingKernel()
    runtime = AgentRuntime.start(
        session_id="s1",
        question="q",
        kernel=kernel,
        budgets=BudgetPolicy(max_actions=1),
    )
    verdict = runtime.finalize()
    assert runtime.status is SessionStatus.PARTIAL
    assert "frontier" in verdict["evaluation"] or "reasons" in verdict["evaluation"]


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
    result = runtime.apply(action="finalize", payload={})
    assert result.accepted
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


# ---------------------------------------------------------------------------
# Certification delegation tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Replay and provider independence
# ---------------------------------------------------------------------------


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


def test_two_kernels_with_same_protocol_produce_equivalent_results():
    kernel_a = _CountingKernel()
    kernel_b = _CountingKernel()
    runtime_a = AgentRuntime.start(session_id="s1", question="q", kernel=kernel_a)
    runtime_b = AgentRuntime.start(session_id="s1", question="q", kernel=kernel_b)
    payload = {"node_id": "root", "children": [{"id": "a", "label": "A"}]}
    res_a = runtime_a.apply(action="expand", payload=payload)
    res_b = runtime_b.apply(action="expand", payload=payload)
    assert res_a.result_revision == res_b.result_revision
    assert res_a.state_digest == res_b.state_digest


def test_provider_independence_no_model_invocation():
    """The runtime must not invoke any model provider or prompt builder."""

    class _SpyKernel(_CountingKernel):
        def __init__(self) -> None:
            super().__init__()
            self.invocations: list[str] = []

        # If the runtime ever tried to call a model, this hook would fire.
        def invoke_model(self, *args, **kwargs):  # pragma: no cover - intentional guard
            self.invocations.append("invoke_model")
            raise AssertionError("runtime must not invoke models")

    kernel = _SpyKernel()
    runtime = AgentRuntime.start(session_id="s1", question="q", kernel=kernel)
    runtime.apply(action="expand", payload={"node_id": "root", "children": [{"id": "a", "label": "A"}]})
    runtime.finalize()
    assert kernel.invocations == []


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


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
