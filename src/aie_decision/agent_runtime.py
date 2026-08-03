"""Agent decomposition runtime — Track C orchestrator.

This module is the stateful, replayable tool loop described by
``specs/agent-decomposition-runtime/spec.md``.  It owns:

* the append-only trajectory (delegated to :mod:`aie_decision.trajectory`);
* optimistic revision checks against the latest accepted state;
* legal-action discovery driven by an injected kernel;
* structured accept/reject responses with stable error codes;
* per-session turn, action, depth, and computation budgets;
* honest partial termination when budgets are exhausted;
* certification that is entirely delegated to an injected deterministic
  kernel — the runtime never certifies frontier claims on its own, and
  prose or a ``finalize`` request cannot bypass the kernel's gates.

The runtime imports neither Track A (tree) nor Track B (uncertainty /
frontier) code.  Kernel behaviour is supplied by callers through the
:class:`KernelProtocol` (or any object that satisfies it).  This lets the
three tracks land independently and meet only at the integration adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from .trajectory import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    EventKind,
    EventStatus,
    Trajectory,
    TrajectoryError,
    payload_digest as payload_digest_fn,
    state_digest,
    utc_now,
)


# ---------------------------------------------------------------------------
# Kernel protocol
# ---------------------------------------------------------------------------


class KernelProtocol(Protocol):
    """Deterministic contract between the runtime and an injected kernel.

    The runtime is policy-free.  It does not know what ``expand``,
    ``estimate``, or ``evaluate`` mean — the kernel does.  The kernel
    returns a new state as a plain JSON-compatible mapping; the runtime
    treats that mapping as opaque and only uses it to compute digests,
    budgets, and the next-action surface.
    """

    def initial_state(self, question: str) -> dict[str, Any]:
        """Return the deterministic initial state for a new session."""

    def action_specs(self) -> list[dict[str, Any]]:
        """Return the catalogue of action names, required fields, and costs."""

    def validate(
        self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Return a list of structured errors.  Empty list means valid."""

    def execute(
        self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Apply ``action`` to a copy of ``state`` and return the new state.

        Must be a pure function of its arguments.  Raising is allowed only
        for unrecoverable programmer errors; validation failures must be
        reported through :meth:`validate`.
        """

    def legal_next_actions(self, state: Mapping[str, Any]) -> list[str]:
        """Return the action names currently considered legal."""

    def active_frontier(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return the unresolved frontier nodes for budget-aware scheduling."""

    def evaluate_frontier(
        self, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return a structured frontier evaluation.

        The runtime treats the returned mapping's ``status`` field as the
        canonical verdict.  Recognised statuses are ``certified``,
        ``insufficient``, ``partial``, and ``ineligible``.  Anything else
        is treated as ``insufficient`` and surfaced as a rejected action.
        """


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):
    """Lifecycle of a runtime session.

    ``ACTIVE`` is the working state.  ``CERTIFIED`` is only set when an
    injected kernel reports a passing frontier evaluation.  ``INSUFFICIENT``
    records a finalised evaluation that did not pass every gate.  ``PARTIAL``
    is the honest, budget-driven stop: the kernel still has open work and
    the runtime refuses to claim completion.  ``INVALID`` is reserved for
    a corrupt or unrecoverable session.
    """

    ACTIVE = "active"
    CERTIFIED = "certified"
    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    INVALID = "invalid"


class RuntimeError_(ValueError):
    """Raised for unrecoverable runtime errors (bad usage, corrupt state)."""


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Per-session turn, action, depth, and computation budgets.

    A value of ``None`` disables the corresponding check.  ``max_compute``
    is an abstract counter that the kernel can charge through
    ``ActionResult.compute_cost``; the runtime enforces the cap.
    """

    max_actions: int | None = None
    max_depth: int | None = None
    max_evaluations: int | None = None
    max_compute: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_actions": self.max_actions,
            "max_depth": self.max_depth,
            "max_evaluations": self.max_evaluations,
            "max_compute": self.max_compute,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "BudgetPolicy":
        return cls(
            max_actions=_as_int(document.get("max_actions")),
            max_depth=_as_int(document.get("max_depth")),
            max_evaluations=_as_int(document.get("max_evaluations")),
            max_compute=_as_int(document.get("max_compute")),
        )


@dataclass(frozen=True, slots=True)
class BudgetCounters:
    """Current usage of the budget policy."""

    actions: int = 0
    evaluations: int = 0
    compute: int = 0
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "evaluations": self.evaluations,
            "compute": self.compute,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class ActionResult:
    """The outcome of a single ``apply`` call."""

    accepted: bool
    status: str
    action: str
    payload_digest: str
    prior_revision: str | None
    result_revision: str | None
    state_digest: str | None
    sequence: int
    error: dict[str, Any] | None = None
    legal_next_actions: tuple[str, ...] = ()
    active_frontier: tuple[dict[str, Any], ...] = ()
    compute_cost: int = 0
    budget_remaining: dict[str, Any] = field(default_factory=dict)
    rollback_target_sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "accepted": self.accepted,
            "status": self.status,
            "action": self.action,
            "payload_digest": self.payload_digest,
            "prior_revision": self.prior_revision,
            "result_revision": self.result_revision,
            "state_digest": self.state_digest,
            "sequence": self.sequence,
            "error": self.error,
            "legal_next_actions": list(self.legal_next_actions),
            "active_frontier": list(self.active_frontier),
            "compute_cost": self.compute_cost,
            "budget_remaining": dict(self.budget_remaining),
            "rollback_target_sequence": self.rollback_target_sequence,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError_(f"budget value must be int or None, got {type(value).__name__}")
    return value


def _now_iso(clock: Callable[[], str] | None) -> str:
    return (clock or utc_now)()


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    # ``json`` round-trip gives a defensive, JSON-compatible copy that
    # strips any non-serialisable values and re-emits tuples as lists.
    return json.loads(json.dumps(state, ensure_ascii=False, default=str))


def _default_compute_cost(action: str) -> int:
    # Frontier evaluations and rollbacks are more expensive than ordinary
    # expansions or estimates.  The kernel may override these defaults by
    # passing ``compute_cost`` explicitly to ``apply``.
    if action in {"evaluate", "finalize", "certify"}:
        return 5
    if action == "rollback":
        return 3
    return 1


# Compatibility fallback for the budget category lookups.  Used when the
# injected kernel's ``action_specs`` lacks a ``category`` field, so the
# runtime does not drift onto a specific kernel's action names.
_FALLBACK_EVALUATION_ACTIONS = frozenset({"evaluate", "test_frontier", "finalize", "certify"})
_FALLBACK_DEPTH_ACTIONS = frozenset({"expand", "propose_alternative", "alternative"})

_EVALUATION_CATEGORIES = frozenset({
    "evaluation",
    "frontier",
    "certification",
    "frontier_evaluation",
})
_DEPTH_CATEGORIES = frozenset({
    "structural",
    "expansion",
    "tree",
    "decomposition",
})


def _structure_error(
    code: str, path: str, message: str, **extra: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "path": path, "message": message}
    for key, value in extra.items():
        body[key] = value
    return body


def _coerce_str_list(value: Any) -> list[str] | None:
    """Coerce ``value`` to a list of strings, or ``None`` if it is malformed.

    Returns ``None`` only for values that cannot be coerced at all (e.g.
    a bare ``int``, ``None``, or a mapping).  An empty container, a
    bare string, or a sequence of arbitrary objects is normalised.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        coerced: list[str] = []
        for item in value:
            if item is None:
                continue
            coerced.append(str(item))
        return coerced
    return None


def _normalize_frontier_evaluation(
    evaluation: Any,
) -> dict[str, Any] | None:
    """Normalize a kernel frontier evaluation.

    Returns the evaluation as a JSON-compatible mapping, or ``None`` if
    the kernel returned something the runtime cannot safely record.
    ``reasons`` and ``blocking_issues`` are coerced to lists of
    strings; ``status`` defaults to ``"insufficient"`` when absent.
    """
    if not isinstance(evaluation, Mapping):
        return None
    normalised: dict[str, Any] = dict(evaluation)
    status = normalised.get("status")
    if status is None:
        normalised["status"] = "insufficient"
    elif not isinstance(status, str) or not status.strip():
        return None
    reasons = _coerce_str_list(normalised.get("reasons"))
    if reasons is None:
        return None
    normalised["reasons"] = reasons
    blocking = _coerce_str_list(normalised.get("blocking_issues"))
    if blocking is None:
        return None
    normalised["blocking_issues"] = blocking
    return normalised


# ---------------------------------------------------------------------------
# Session runtime
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentRuntime:
    """Stateful, replayable tool loop for an external AI.

    The runtime owns the trajectory, the current state, the budget policy,
    and the budget counters.  All behaviour beyond persistence and
    bookkeeping is delegated to the injected kernel.
    """

    session_id: str
    kernel: KernelProtocol
    trajectory: Trajectory
    state: dict[str, Any] = field(default_factory=dict)
    question: str = ""
    budgets: BudgetPolicy = field(default_factory=BudgetPolicy)
    counters: BudgetCounters = field(default_factory=BudgetCounters)
    status: SessionStatus = SessionStatus.ACTIVE
    frontier_evaluation: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _clock: Callable[[], str] | None = field(default=None, repr=False, compare=False)
    _start_recorded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise RuntimeError_("session_id is required")
        if not self.created_at:
            self.created_at = _now_iso(self._clock)
        self.updated_at = self.created_at

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

    def _project_state(
        self,
        *,
        pairs: Sequence[tuple[Any, Any]] | None = None,
        extra_action: tuple[int, str, Mapping[str, Any], int | None] | None = None,
    ) -> dict[str, Any]:
        """Rebuild visible state from accepted live semantic actions.

        Rollback is an event-projection operation, not a kernel operation.  An
        accepted rollback marks its target action as non-live; neither the
        rollback nor its target is executed while constructing the current
        projection.  ``extra_action`` lets ``apply`` calculate the exact state
        that a prospective accepted event would produce before appending its
        result digest.
        """

        source_pairs = list(self.trajectory.pairs() if pairs is None else pairs)
        accepted: list[tuple[int, str, Mapping[str, Any], int | None]] = []
        rolled_back: set[int] = set()
        for action_event, result_event in source_pairs:
            if result_event.status is not EventStatus.ACCEPTED:
                continue
            item = (
                action_event.sequence,
                str(action_event.action),
                dict(action_event.payload or {}),
                action_event.rollback_target_sequence,
            )
            accepted.append(item)
            if item[1] == "rollback" and item[3] is not None:
                rolled_back.add(item[3])

        if extra_action is not None:
            accepted.append(extra_action)
            if extra_action[1] == "rollback" and extra_action[3] is not None:
                rolled_back.add(extra_action[3])

        state = _copy_state(self.kernel.initial_state(self.question))
        for sequence, action, payload, _ in accepted:
            if action in {"start", "rollback", "finalize"}:
                continue
            if sequence in rolled_back:
                continue
            state = _copy_state(self.kernel.execute(action, payload, state))
        return state

    def rebuild_projection(self) -> dict[str, Any]:
        """Replace and return current state from the trajectory authority."""

        self.state = self._project_state()
        return _copy_state(self.state)

    def _validate_rollback_target(self, target_sequence: int | None) -> None:
        if target_sequence is None:
            raise RuntimeError_("rollback action requires rollback_target_sequence")
        for action_event, result_event in self.trajectory.pairs():
            if action_event.sequence != target_sequence:
                continue
            if result_event.status is not EventStatus.ACCEPTED:
                raise RuntimeError_("rollback target must be an accepted action")
            if action_event.action in {"start", "rollback", "finalize"}:
                raise RuntimeError_(
                    f"the {action_event.action} action cannot be rolled back"
                )
            if not self.trajectory.is_live(target_sequence):
                raise RuntimeError_("rollback target is already rolled back")
            return
        raise RuntimeError_(
            f"rollback target sequence {target_sequence} does not match any action event"
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

    def replay(self) -> dict[str, Any]:
        """Re-run the trajectory through the kernel and verify digests.

        Returns the reconstructed state together with a verdict describing
        whether every accepted action produced the recorded digest.  The
        runtime does not mutate the running state — it walks the
        trajectory in a sandboxed copy.

        Control actions (``start``, ``rollback``, ``finalize``) are not
        kernel calls: their effect is the trajectory's projection up to
        that pair.  Re-projecting at each control action lets a later
        accepted action that follows an earlier rollback still see the
        state the runtime would have computed, so replay reconstructs
        the same live state as ``_project_state`` after a rollback.
        """

        if self.trajectory.is_empty():
            return {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "session_id": self.session_id,
                "reconstructed_state": {},
                "matches": [],
                "mismatches": [],
                "verdict": "empty",
            }

        try:
            state = _copy_state(self.kernel.initial_state(self.question))
        except Exception:
            state = {}
        matches: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        # Collect rolled-back sequences before replay so that neither the
        # rollback action nor its target is executed in the projection.
        rolled_back: set[int] = self.trajectory.rolled_back_sequences()
        all_pairs = self.trajectory.pairs()
        cumulative_pairs: list[tuple[Any, Any]] = []
        for action_event, result_event in all_pairs:
            if result_event.status is not EventStatus.ACCEPTED:
                cumulative_pairs.append((action_event, result_event))
                continue
            # Rollback's target is excluded from the live state regardless
            # of when the rollback was issued.
            if action_event.sequence in rolled_back:
                cumulative_pairs.append((action_event, result_event))
                continue
            cumulative_pairs.append((action_event, result_event))
            try:
                if action_event.action in {"start", "rollback", "finalize"}:
                    # Re-project the state up to and including this
                    # control pair so a later accepted action that
                    # follows an earlier rollback sees the same live
                    # state the runtime would have computed.
                    candidate = self._project_state(pairs=cumulative_pairs)
                else:
                    candidate = _copy_state(self.kernel.execute(
                        action_event.action, action_event.payload, state
                    ))
            except Exception as exc:
                mismatches.append({
                    "sequence": action_event.sequence,
                    "action": action_event.action,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                break
            actual_digest = state_digest(candidate)
            expected_digest = result_event.state_digest_after
            if actual_digest == expected_digest:
                state = candidate
                matches.append({
                    "sequence": action_event.sequence,
                    "action": action_event.action,
                })
            else:
                mismatches.append({
                    "sequence": action_event.sequence,
                    "action": action_event.action,
                    "expected": expected_digest,
                    "actual": actual_digest,
                })
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "reconstructed_state": state,
            "matches": matches,
            "mismatches": mismatches,
            "verdict": "match" if not mismatches else "mismatch",
        }

    # --- budget helpers --------------------------------------------------

    def _compute_depth(self) -> int:
        # The kernel is responsible for tracking the actual tree depth.
        # If it chooses to surface a ``depth`` integer on the state, the
        # runtime uses that value; otherwise the depth counter freezes.
        if "depth" in self.state and isinstance(self.state["depth"], int):
            return int(self.state["depth"])
        return self.counters.depth

    def _candidate_depth(self, candidate_state: Mapping[str, Any]) -> int:
        # Depth derived strictly from the candidate state, before the
        # in-memory ``self.state`` has been committed.  This lets the
        # runtime build new counters without mutating self.state.
        if isinstance(candidate_state, Mapping) and "depth" in candidate_state:
            value = candidate_state["depth"]
            if isinstance(value, bool):
                return self.counters.depth
            if isinstance(value, int):
                return int(value)
        return self.counters.depth

    def _action_categories(self) -> dict[str, str]:
        """Return ``{action_name: category}`` derived from kernel ``action_specs``.

        Kernels may surface a ``category`` on each spec.  When no
        category is available the returned mapping is empty and the
        runtime falls back to its compatibility name sets.  Exceptions
        Kernel exceptions are swallowed so a buggy spec cannot crash
        bookkeeping.
        """
        categories: dict[str, str] = {}
        try:
            specs = list(self.kernel.action_specs())
        except Exception:
            return categories
        for spec in specs:
            if not isinstance(spec, Mapping):
                continue
            name = spec.get("name")
            category = spec.get("category")
            if isinstance(name, str) and name and isinstance(category, str) and category:
                categories[name] = category
        return categories

    def _evaluation_actions(self) -> set[str]:
        """Action names that consume the evaluation budget.

        If the kernel supplies ``category`` metadata, we trust that
        taxonomy.  Otherwise we fall back to a name-based compatibility
        set so the runtime does not hard-code a specific kernel's
        action names.
        """
        categories = self._action_categories()
        if categories:
            return {
                name
                for name, category in categories.items()
                if category in _EVALUATION_CATEGORIES
            }
        return set(_FALLBACK_EVALUATION_ACTIONS)

    def _depth_actions(self) -> set[str]:
        """Action names that consume the depth budget."""
        categories = self._action_categories()
        if categories:
            return {
                name
                for name, category in categories.items()
                if category in _DEPTH_CATEGORIES
            }
        return set(_FALLBACK_DEPTH_ACTIONS)

    def _close_dangling_action(
        self, *, prior_revision: str | None, code: str, message: str
    ) -> None:
        """Best-effort repair of a trailing ACTION that never received a RESULT.

        Used when ``record_action`` succeeded but ``record_result`` raised
        (or vice versa).  The runtime only invokes this for programming
        errors so a failure here is itself swallowed quietly; the
        in-memory state and counters remain at their pre-call values.
        """
        events = self.trajectory.events
        if not events or events[-1].kind is not EventKind.ACTION:
            return
        try:
            self.trajectory.record_result(
                status=EventStatus.REJECTED,
                error={"issues": [_structure_error(code, "$", message)]},
                recorded_at=_now_iso(self._clock),
            )
        except Exception:
            # If the trajectory is in an unrecoverable state, leave it
            # alone.  The runtime already refused to mutate state.
            pass
        _ = prior_revision  # accepted for symmetry; not currently used.

    def _budget_remaining(self) -> dict[str, Any]:
        remaining: dict[str, Any] = {}
        if self.budgets.max_actions is not None:
            remaining["actions"] = max(0, self.budgets.max_actions - self.counters.actions)
        if self.budgets.max_evaluations is not None:
            remaining["evaluations"] = max(0, self.budgets.max_evaluations - self.counters.evaluations)
        if self.budgets.max_compute is not None:
            remaining["compute"] = max(0, self.budgets.max_compute - self.counters.compute)
        if self.budgets.max_depth is not None:
            remaining["depth"] = max(0, self.budgets.max_depth - self.counters.depth)
        return remaining

    def _budget_violation(
        self, *, action: str, compute_cost: int | None
    ) -> dict[str, str] | None:
        if (
            self.budgets.max_actions is not None
            and self.counters.actions + 1 > self.budgets.max_actions
        ):
            return _structure_error(
                "budget_exhausted",
                "$.actions",
                "action budget exhausted; remaining actions required before certification",
            )
        if (
            self.budgets.max_evaluations is not None
            and action in self._evaluation_actions()
            and self.counters.evaluations + 1 > self.budgets.max_evaluations
        ):
            return _structure_error(
                "budget_exhausted",
                "$.evaluations",
                "evaluation budget exhausted; remaining frontier cannot be certified",
            )
        cost = compute_cost if compute_cost is not None else _default_compute_cost(action)
        if (
            self.budgets.max_compute is not None
            and self.counters.compute + cost > self.budgets.max_compute
        ):
            return _structure_error(
                "budget_exhausted",
                "$.compute",
                "computation budget exhausted; remaining frontier cannot be expanded",
            )
        if (
            self.budgets.max_depth is not None
            and action in self._depth_actions()
            and self.counters.depth + 1 > self.budgets.max_depth
        ):
            return _structure_error(
                "budget_exhausted",
                "$.depth",
                "depth budget exhausted; further expansion is not permitted",
            )
        return None

    def _mark_budget_partial(self) -> None:
        """Mark the session as PARTIAL if any budget is already exhausted.

        Called after actions that consume budget even when the action itself
        is accepted — the session cannot return a CERTIFIED result once a
        configured budget is depleted.
        """
        if self.status is SessionStatus.CERTIFIED:
            return
        if self._budget_is_exhausted():
            self.status = SessionStatus.PARTIAL

    def _budget_is_exhausted(self) -> bool:
        """True when any configured budget has been fully consumed."""

        return self._budget_is_exhausted_for(self.counters)

    def _budget_is_exhausted_for(self, counters: BudgetCounters) -> bool:
        if self.budgets.max_actions is not None and counters.actions >= self.budgets.max_actions:
            return True
        if (
            self.budgets.max_evaluations is not None
            and counters.evaluations >= self.budgets.max_evaluations
        ):
            return True
        if self.budgets.max_compute is not None and counters.compute >= self.budgets.max_compute:
            return True
        if self.budgets.max_depth is not None and counters.depth >= self.budgets.max_depth:
            return True
        return False

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


__all__ = [
    "AgentRuntime",
    "ActionResult",
    "BudgetCounters",
    "BudgetPolicy",
    "KernelProtocol",
    "PROTOCOL_VERSION",
    "RuntimeError_",
    "SCHEMA_VERSION",
    "SessionStatus",
]
