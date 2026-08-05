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

__all__ = [
    "ActionResult",
    "BudgetCounters",
    "BudgetPolicy",
    "KernelProtocol",
    "RuntimeError_",
    "SessionStatus",
]
