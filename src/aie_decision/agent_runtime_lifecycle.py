from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .agent_runtime_support import (
    ActionResult,
    BudgetCounters,
    BudgetPolicy,
    KernelProtocol,
    RuntimeError_,
    SessionStatus,
    _copy_state,
    _default_compute_cost,
    _normalize_frontier_evaluation,
    _now_iso,
    _structure_error,
)
from .trajectory import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    EventStatus,
    Trajectory,
    payload_digest as payload_digest_fn,
    state_digest,
)


class LifecycleMixin:
    # --- factory ---------------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        question: str,
        kernel: KernelProtocol,
        budgets: BudgetPolicy | None = None,
        metadata: Mapping[str, Any] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> "AgentRuntime":
        """Initialise a session.  The ``start`` action is recorded eagerly."""

        if not question:
            raise RuntimeError_("question is required to start a session")
        trajectory = Trajectory(session_id, clock=clock)
        initial_state = _copy_state(kernel.initial_state(question))
        runtime = cls(
            session_id=session_id,
            kernel=kernel,
            trajectory=trajectory,
            state=initial_state,
            question=question,
            budgets=budgets or BudgetPolicy(),
            metadata=dict(metadata or {}),
            _clock=clock,
        )
        runtime._record_start(initial_state)
        return runtime

    def _record_start(self, initial_state: dict[str, Any]) -> None:
        if self._start_recorded:
            return
        self.trajectory.record_action(
            action="start",
            payload={"question": self.question, "metadata": dict(self.metadata)},
            prior_revision=None,
            recorded_at=_now_iso(self._clock),
        )
        self.trajectory.record_result(
            status=EventStatus.ACCEPTED,
            result_revision=state_digest(initial_state),
            state_digest_after=state_digest(initial_state),
            recorded_at=_now_iso(self._clock),
            metadata={"initial_state": True},
        )
        self._start_recorded = True
        self.counters = BudgetCounters(
            actions=self.counters.actions + 1,
            evaluations=self.counters.evaluations,
            compute=self.counters.compute + 1,
            depth=self.counters.depth,
        )
        self._mark_budget_partial()  # type: ignore[attr-defined]

    # --- discovery -------------------------------------------------------

    def discover(self) -> dict[str, Any]:
        """Return the action catalogue, legal-next surface, and budget status."""

        specs = list(self.kernel.action_specs())
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "status": self.status.value,
            "current_revision": self.trajectory.last_revision(),
            "action_specs": specs,
            "legal_next_actions": list(self.kernel.legal_next_actions(self.state)),
            "budget_policy": self.budgets.to_dict(),
            "budget_counters": self.counters.to_dict(),
            "budget_remaining": self._budget_remaining(),
            "active_frontier": list(self.kernel.active_frontier(self.state)),
            "frontier_evaluation": self.frontier_evaluation,
        }

    # --- inspect / finalize / replay ------------------------------------

    def inspect(self) -> dict[str, Any]:
        """Return a read-only projection of the current session."""

        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status.value,
            "current_revision": self.trajectory.last_revision(),
            "state": self.state,
            "active_frontier": list(self.kernel.active_frontier(self.state)),
            "legal_next_actions": list(self.kernel.legal_next_actions(self.state)),
            "budget_policy": self.budgets.to_dict(),
            "budget_counters": self.counters.to_dict(),
            "budget_remaining": self._budget_remaining(),
            "frontier_evaluation": self.frontier_evaluation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def finalize(self) -> dict[str, Any]:
        """Request a frontier evaluation.  The verdict is delegated to the kernel.

        The runtime records the finalisation action and result, then
        transitions the session to ``CERTIFIED``, ``INSUFFICIENT``, or
        ``PARTIAL`` according to the kernel's verdict.  The runtime never
        overrides the *gating* verdict: prose or a finalize request
        cannot bypass the kernel's certification logic.  However, when
        any configured budget is exhausted, the runtime demotes a
        ``certified`` verdict to ``PARTIAL`` so the caller cannot mistake
        a budget-driven stop for a genuine frontier certification.

        Once the session leaves ``ACTIVE``, ``finalize`` is a no-op: it
        must not append events or mutate counters.  It returns a
        structured terminal response describing the already-decided
        status.  A truly malformed frontier evaluation (a non-Mapping
        result, or a ``reasons``/``blocking_issues`` field that cannot
        be coerced to a list of strings) is a controlled runtime error
        and never produces a dangling ``finalize`` ACTION event.
        """

        if self.status is not SessionStatus.ACTIVE:
            return self._terminal_finalize_response(reason="session_not_active")

        # Validate the kernel evaluation BEFORE recording any events so
        # an irrecoverably malformed evaluation never leaves a dangling
        # finalize ACTION behind.
        raw_evaluation = self.kernel.evaluate_frontier(self.state)
        evaluation = _normalize_frontier_evaluation(raw_evaluation)
        if evaluation is None:
            raise RuntimeError_(
                "kernel returned a malformed frontier evaluation: "
                "expected a mapping with at least a 'status' field and "
                "list-of-strings 'reasons' and 'blocking_issues' fields"
            )

        verdict = str(evaluation.get("status", "insufficient"))
        # Finalize itself counts as one action and one evaluation, so
        # the post-finalize counters are the right reference point for
        # the budget-exhaustion override.
        projected_counters = BudgetCounters(
            actions=self.counters.actions + 1,
            evaluations=self.counters.evaluations + 1,
            compute=self.counters.compute + _default_compute_cost("finalize"),
            depth=self.counters.depth,
        )
        budget_exhausted = self._budget_is_exhausted_for(projected_counters)
        if budget_exhausted and verdict == "certified":
            evaluation = dict(evaluation)
            evaluation["status"] = "partial"
            evaluation.setdefault(
                "reasons", []
            ).append("budget exhausted; finalise produced an honest partial result")
            verdict = "partial"

        # Record the ACTION and RESULT together so a mid-recording
        # failure cannot leave the trajectory half-written.  In-memory
        # state and counters are only committed after the log accepts
        # both events.
        prior_revision = self.trajectory.last_revision()
        try:
            action_event = self.trajectory.record_action(
                action="finalize",
                payload={"counters": self.counters.to_dict()},
                prior_revision=prior_revision,
                recorded_at=_now_iso(self._clock),
            )
            self.trajectory.record_result(
                status=EventStatus.ACCEPTED,
                result_revision=state_digest(self.state),
                state_digest_after=state_digest(self.state),
                recorded_at=_now_iso(self._clock),
                metadata={"finalize": True, "previous_status": self.status.value},
            )
        except Exception as exc:
            self._close_dangling_action(
                prior_revision=prior_revision,
                code="recording_failed",
                message=f"trajectory recording failed: {type(exc).__name__}: {exc}",
            )
            raise RuntimeError_(
                f"finalize aborted: trajectory recording failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        previous_status = self.status
        self.frontier_evaluation = dict(evaluation)
        if verdict == "certified":
            self.status = SessionStatus.CERTIFIED
        elif verdict == "partial":
            self.status = SessionStatus.PARTIAL
        elif verdict == "ineligible":
            self.status = SessionStatus.PARTIAL
        else:
            self.status = SessionStatus.INSUFFICIENT
        self.counters = projected_counters
        self.updated_at = _now_iso(self._clock)
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "sequence": action_event.sequence,
            "evaluation": self.frontier_evaluation,
            "status": self.status.value,
            "current_revision": self.trajectory.last_revision(),
            "budget_remaining": self._budget_remaining(),
        }

    def _terminal_finalize_response(self, *, reason: str) -> dict[str, Any]:
        """Return a structured no-op response for an already-terminal session."""
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "sequence": None,
            "evaluation": self.frontier_evaluation,
            "status": self.status.value,
            "current_revision": self.trajectory.last_revision(),
            "budget_remaining": self._budget_remaining(),
            "no_op": True,
            "reason": reason,
        }

    # --- persistence -----------------------------------------------------

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status.value,
            "state": self.state,
            "budget_policy": self.budgets.to_dict(),
            "budget_counters": self.counters.to_dict(),
            "frontier_evaluation": self.frontier_evaluation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "trajectory": self.trajectory.export(),
        }
