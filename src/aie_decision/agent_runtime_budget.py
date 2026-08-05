from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .agent_runtime_support import (
    ActionResult,
    BudgetCounters,
    BudgetPolicy,
    KernelProtocol,
    RuntimeError_,
    SessionStatus,
    _DEPTH_CATEGORIES,
    _EVALUATION_CATEGORIES,
    _FALLBACK_DEPTH_ACTIONS,
    _FALLBACK_EVALUATION_ACTIONS,
    _copy_state,
    _default_compute_cost,
    _normalize_frontier_evaluation,
    _now_iso,
    _structure_error,
)
from .trajectory import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    EventKind,
    EventStatus,
    Trajectory,
    payload_digest as payload_digest_fn,
    state_digest,
)


class BudgetMixin:
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
