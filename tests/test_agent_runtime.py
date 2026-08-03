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


# ---------------------------------------------------------------------------
# Item 1 — reject finalize in apply
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Item 2 — record trajectory before committing in-memory state and counters
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Item 3 — finalize is a no-op on inactive session
# ---------------------------------------------------------------------------


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
    from aie_decision.agent_runtime import RuntimeError_

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


# ---------------------------------------------------------------------------
# Item 4 — validate/normalize frontier fields
# ---------------------------------------------------------------------------


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
    from aie_decision.agent_runtime import RuntimeError_

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


# ---------------------------------------------------------------------------
# Item 5 — replay after rollback correctness
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Item 6 — derive action categories from kernel specs
# ---------------------------------------------------------------------------


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
