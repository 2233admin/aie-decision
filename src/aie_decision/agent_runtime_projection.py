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


class ProjectionMixin:
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
