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


class ExecutionMixin:
    # --- apply -----------------------------------------------------------

    def apply(
        self,
        *,
        action: str,
        payload: Mapping[str, Any] | None = None,
        prior_revision: str | None = None,
        rollback_target_sequence: int | None = None,
        compute_cost: int | None = None,
    ) -> ActionResult:
        """Validate, execute, and record one action.

        The runtime always returns an :class:`ActionResult`.  Rejected
        actions are appended to the trajectory as REJECTED results with a
        structured error; their state effect is recorded as none.
        """

        payload_dict = dict(payload or {})
        if action == "rollback" and rollback_target_sequence is None:
            payload_target = payload_dict.get("target_sequence")
            if isinstance(payload_target, int) and not isinstance(payload_target, bool):
                rollback_target_sequence = payload_target

        if not action:
            # An empty action name is a client error: no event is recorded
            # because the trajectory refuses to store a nameless event.
            error = _structure_error("missing_action", "$.action", "action name is required")
            return ActionResult(
                accepted=False,
                status="rejected",
                action="",
                payload_digest=payload_digest_fn(payload_dict),
                prior_revision=self.trajectory.last_revision(),
                result_revision=None,
                state_digest=self.trajectory.last_revision(),
                sequence=0,
                error={"issues": [error]},
                legal_next_actions=tuple(self.kernel.legal_next_actions(self.state)),
                active_frontier=tuple(self.kernel.active_frontier(self.state)),
                compute_cost=0,
                budget_remaining=self._budget_remaining(),
            )

        if action == "finalize":
            # ``finalize`` is the dedicated terminal-request API; it must
            # never be silently accepted as a kernel action through
            # ``apply``.  Reject with a structured message so callers
            # learn to invoke the dedicated ``finalize`` method.
            return self._rejected_result(
                action="finalize",
                payload=payload_dict,
                prior_revision=self.trajectory.last_revision(),
                code="use_finalize_method",
                message=(
                    "finalize is not accepted through apply(); call the dedicated "
                    "finalize() method to request a frontier evaluation"
                ),
                payload_digest=payload_digest_fn(payload_dict),
            )

        payload_hash = payload_digest_fn(payload_dict)
        current_revision = self.trajectory.last_revision()

        # Optimistic budget check before the status check so the
        # structured path (e.g. ``$.depth``, ``$.evaluations``) survives
        # even when the session is already in ``PARTIAL`` because of a
        # prior exhaustion.  The status check below still rejects
        # non-budget sessions with ``session_not_active``.
        budget_violation = self._budget_violation(action=action, compute_cost=compute_cost)
        if budget_violation is not None:
            return self._rejected_result(
                action=action,
                payload=payload_dict,
                prior_revision=current_revision,
                code=budget_violation["code"],
                message=budget_violation["message"],
                path=budget_violation.get("path", "$"),
                payload_digest=payload_hash,
            )

        if self.status is not SessionStatus.ACTIVE:
            return self._rejected_result(
                action=action,
                payload=payload_dict,
                prior_revision=current_revision,
                code="session_not_active",
                message=(
                    f"session status is {self.status.value}; no further actions accepted"
                ),
                payload_digest=payload_hash,
            )

        if prior_revision is not None and prior_revision != current_revision:
            return self._rejected_result(
                action=action,
                payload=payload_dict,
                prior_revision=current_revision,
                code="stale_revision",
                message="prior_revision does not match the current state",
                payload_digest=payload_hash,
            )

        # Rollback target must exist and target a real, non-``start`` action.
        if action == "rollback":
            try:
                self._validate_rollback_target(rollback_target_sequence)
            except RuntimeError_ as exc:
                return self._rejected_result(
                    action=action,
                    payload=payload_dict,
                    prior_revision=current_revision,
                    code="invalid_rollback_target",
                    message=str(exc),
                    payload_digest=payload_hash,
                )

        errors = list(self.kernel.validate(action, payload_dict, self.state))
        if errors:
            return self._record_rejected(
                action=action,
                payload=payload_dict,
                payload_digest=payload_hash,
                prior_revision=current_revision,
                errors=errors,
            )

        projected_sequence = len(self.trajectory.events) + 1
        try:
            new_state = self._project_state(
                extra_action=(
                    projected_sequence,
                    action,
                    payload_dict,
                    rollback_target_sequence,
                )
            )
        except Exception as exc:
            return self._record_rejected(
                action=action,
                payload=payload_dict,
                payload_digest=payload_hash,
                prior_revision=current_revision,
                errors=[
                    _structure_error(
                        "projection_failed",
                        "$",
                        f"accepted-state projection failed: {type(exc).__name__}: {exc}",
                    )
                ],
            )
        new_digest = state_digest(new_state)
        applied_cost = compute_cost if compute_cost is not None else _default_compute_cost(action)
        # The candidate counters depend on the candidate state (depth) and
        # on the evaluation category, so compute them in memory before
        # touching the trajectory.  ``self.state`` and ``self.counters`` are
        # only committed after the trajectory has accepted the events.
        candidate_counters = BudgetCounters(
            actions=self.counters.actions + 1,
            evaluations=(
                self.counters.evaluations + 1
                if action in self._evaluation_actions()
                else self.counters.evaluations
            ),
            compute=self.counters.compute + applied_cost,
            depth=self._candidate_depth(new_state),
        )

        # Record trajectory events first.  If recording fails, the
        # in-memory state and counters remain at their pre-call values
        # so the runtime cannot diverge from the authoritative log.
        try:
            action_event = self.trajectory.record_action(
                action=action,
                payload=payload_dict,
                prior_revision=current_revision,
                recorded_at=_now_iso(self._clock),
                rollback_target_sequence=rollback_target_sequence,
            )
            self.trajectory.record_result(
                status=EventStatus.ACCEPTED,
                result_revision=new_digest,
                state_digest_after=new_digest,
                recorded_at=_now_iso(self._clock),
                metadata={"compute_cost": applied_cost},
            )
        except Exception as exc:
            # Trajectory refused the event.  In-memory state and
            # counters are deliberately left untouched.  Best-effort
            # close out a dangling ACTION (if any) with a REJECTED
            # result via the trajectory's own append-only path so the
            # log remains well-formed, then build a rejected
            # ``ActionResult`` directly so we never re-invoke the
            # failing trajectory to record the rejection itself.
            self._close_dangling_action(
                prior_revision=current_revision,
                code="recording_failed",
                message=(
                    f"trajectory recording failed: {type(exc).__name__}: {exc}"
                ),
            )
            error_message = (
                f"trajectory recording failed: {type(exc).__name__}: {exc}"
            )
            last_seq = 0
            try:
                if len(self.trajectory.events) >= 2:
                    last_seq = self.trajectory.events[-2].sequence
            except Exception:
                last_seq = 0
            try:
                last_rev = self.trajectory.last_revision()
            except Exception:
                last_rev = current_revision
            return ActionResult(
                accepted=False,
                status="rejected",
                action=action,
                payload_digest=payload_hash,
                prior_revision=current_revision,
                result_revision=None,
                state_digest=last_rev,
                sequence=last_seq,
                error={"issues": [_structure_error("recording_failed", "$", error_message)]},
                legal_next_actions=tuple(self.kernel.legal_next_actions(self.state)),
                active_frontier=tuple(self.kernel.active_frontier(self.state)),
                compute_cost=0,
                budget_remaining=self._budget_remaining(),
            )

        self.state = new_state
        self.counters = candidate_counters
        self.updated_at = _now_iso(self._clock)
        self._mark_budget_partial()  # type: ignore[attr-defined]
        legal_next = tuple(self.kernel.legal_next_actions(self.state))
        frontier = tuple(self.kernel.active_frontier(self.state))

        return ActionResult(
            accepted=True,
            status=(
                "partial"
                if self.status is SessionStatus.PARTIAL
                else "accepted"
            ),
            action=action,
            payload_digest=payload_hash,
            prior_revision=current_revision,
            result_revision=new_digest,
            state_digest=new_digest,
            sequence=action_event.sequence,
            error=None,
            legal_next_actions=legal_next,
            active_frontier=frontier,
            compute_cost=applied_cost,
            budget_remaining=self._budget_remaining(),
            rollback_target_sequence=rollback_target_sequence,
        )

    def _record_rejected(
        self,
        *,
        action: str,
        payload: Mapping[str, Any],
        payload_digest: str,
        prior_revision: str | None,
        errors: Sequence[Mapping[str, Any]],
    ) -> ActionResult:
        self.trajectory.record_action(
            action=action,
            payload=payload,
            prior_revision=prior_revision,
            recorded_at=_now_iso(self._clock),
        )
        self.trajectory.record_result(
            status=EventStatus.REJECTED,
            error={"issues": list(errors)},
            recorded_at=_now_iso(self._clock),
        )
        self.counters = BudgetCounters(
            actions=self.counters.actions + 1,
            evaluations=self.counters.evaluations,
            compute=self.counters.compute,
            depth=self.counters.depth,
        )
        self.updated_at = _now_iso(self._clock)
        return ActionResult(
            accepted=False,
            status="rejected",
            action=action,
            payload_digest=payload_digest,
            prior_revision=prior_revision,
            result_revision=None,
            state_digest=self.trajectory.last_revision(),
            sequence=self.trajectory.events[-2].sequence,
            error={"issues": list(errors)},
            legal_next_actions=tuple(self.kernel.legal_next_actions(self.state)),
            active_frontier=tuple(self.kernel.active_frontier(self.state)),
            compute_cost=0,
            budget_remaining=self._budget_remaining(),
        )
    def _rejected_result(
        self,
        *,
        action: str,
        payload: Mapping[str, Any],
        prior_revision: str | None,
        code: str,
        message: str,
        path: str = "$",
        payload_digest: str | None = None,
    ) -> ActionResult:
        # ``payload_digest`` here is the caller's hash for the rejected
        # payload; the imported ``payload_digest`` is the hashing function.
        # Resolve the local before shadowing the import.
        digest = payload_digest if payload_digest is not None else payload_digest_fn(payload)
        # A budget or status rejection is still appended to history as a
        # REJECTED event so that the AI cannot bypass the gate by simply
        # moving on to a different action.
        issue = _structure_error(code, path, message)
        self.trajectory.record_action(
            action=action,
            payload=payload,
            prior_revision=prior_revision,
            recorded_at=_now_iso(self._clock),
        )
        self.trajectory.record_result(
            status=EventStatus.REJECTED,
            error={"issues": [issue]},
            recorded_at=_now_iso(self._clock),
        )
        self.counters = BudgetCounters(
            actions=self.counters.actions + 1,
            evaluations=self.counters.evaluations,
            compute=self.counters.compute,
            depth=self.counters.depth,
        )
        self.updated_at = _now_iso(self._clock)
        return ActionResult(
            accepted=False,
            status="rejected",
            action=action,
            payload_digest=digest,
            prior_revision=prior_revision,
            result_revision=None,
            state_digest=self.trajectory.last_revision(),
            sequence=self.trajectory.events[-2].sequence,
            error={"issues": [issue]},
            legal_next_actions=tuple(self.kernel.legal_next_actions(self.state)),
            active_frontier=tuple(self.kernel.active_frontier(self.state)),
            compute_cost=0,
            budget_remaining=self._budget_remaining(),
        )
