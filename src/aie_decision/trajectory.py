"""Append-only event-sourced trajectory for the agent decomposition runtime.

The trajectory is the authoritative, replayable record of every attempted
action and its deterministic outcome.  State is a pure projection of
accepted events.  Rejected and rolled-back actions remain visible in
history but contribute no current state effect.

The runtime treats the trajectory as a JSON-serialisable sequence of
``(ACTION, RESULT)`` event pairs.  Each event records a canonical payload
digest, prior and resulting revisions, a deterministic timestamp, and the
status that controls projection.  Replaying accepted events through the
injected kernel MUST reconstruct the same state.

This module deliberately imports neither Track A (tree) nor Track B
(uncertainty/frontier) code.  Kernel behaviour is supplied through
``agent_runtime`` by the integration adapter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "track-c/1.0.0"

# A trajectory event occupies an odd sequence position when it is an
# ACTION and an even position when it is a RESULT.  The strict alternation
# lets replay tools recognise the (action, result) pairing without needing
# additional metadata.
_ACTION_KIND_PARITY = 1
_RESULT_KIND_PARITY = 0


class EventKind(str, Enum):
    """Two event kinds form the strict (ACTION, RESULT) pairing."""

    ACTION = "action"
    RESULT = "result"


class EventStatus(str, Enum):
    """Status of a result or projected state of an action.

    ``ACCEPTED`` means the runtime applied the action's effect.  ``REJECTED``
    means validation failed; the action is preserved for audit but never
    mutates state.  ``ROLLED_BACK`` is only used during projection to mark
    a previously accepted action whose effect was later reversed by a
    ``rollback`` action.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class TrajectoryError(ValueError):
    """Raised when a trajectory is asked to record an invalid event."""


def _to_primitive(value: Any) -> Any:
    """Convert ``value`` into a JSON-compatible, hash-stable form."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot canonicalize value of type {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Render ``value`` as deterministic UTF-8 JSON bytes."""

    return json.dumps(
        _to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_digest(payload: Mapping[str, Any]) -> str:
    """Compute the canonical SHA-256 digest of a payload mapping."""

    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def state_digest(state: Mapping[str, Any]) -> str:
    """Compute the canonical SHA-256 digest of a state mapping."""

    return hashlib.sha256(canonical_bytes(state)).hexdigest()


def utc_now() -> str:
    """Return a deterministic ISO-8601 UTC timestamp for event ordering."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    """A single append-only event in the trajectory."""

    sequence: int
    kind: EventKind
    action: str | None
    payload: Mapping[str, Any] | None
    payload_digest: str | None
    parent_sequence: int | None
    prior_revision: str | None
    result_revision: str | None
    state_digest_after: str | None
    status: EventStatus | None
    error: Mapping[str, Any] | None
    rollback_target_sequence: int | None
    recorded_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the event as a JSON-compatible dictionary."""

        return {
            "sequence": self.sequence,
            "kind": self.kind.value,
            "action": self.action,
            "payload": _to_primitive(self.payload) if self.payload is not None else None,
            "payload_digest": self.payload_digest,
            "parent_sequence": self.parent_sequence,
            "prior_revision": self.prior_revision,
            "result_revision": self.result_revision,
            "state_digest_after": self.state_digest_after,
            "status": self.status.value if self.status is not None else None,
            "error": _to_primitive(self.error) if self.error is not None else None,
            "rollback_target_sequence": self.rollback_target_sequence,
            "recorded_at": self.recorded_at,
            "metadata": _to_primitive(self.metadata),
        }


def _assert_sequence_parity(sequence: int, kind: EventKind) -> None:
    expected = _ACTION_KIND_PARITY if kind is EventKind.ACTION else _RESULT_KIND_PARITY
    if sequence % 2 != expected:
        raise TrajectoryError(
            f"{kind.value.upper()} event must occupy a "
            f"{'odd' if expected else 'even'} sequence position; got {sequence}"
        )


def _validate_payload_shape(payload: Mapping[str, Any]) -> None:
    # Eagerly canonicalise so non-serialisable values are rejected before
    # they are appended to history.
    _to_primitive(payload)


class Trajectory:
    """Append-only event store with strict (ACTION, RESULT) pairing.

    The trajectory never rewrites history.  ``record_action`` appends an
    attempted action; ``record_result`` appends its outcome.  Pairing is
    enforced by sequence parity and contiguity.
    """

    def __init__(self, session_id: str, *, clock: Callable[[], str] | None = None) -> None:
        if not session_id:
            raise TrajectoryError("session_id is required")
        self.session_id = session_id
        self._events: list[TrajectoryEvent] = []
        self._clock = clock or utc_now

    # --- introspection ----------------------------------------------------

    @property
    def events(self) -> tuple[TrajectoryEvent, ...]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def is_empty(self) -> bool:
        return not self._events

    def last_revision(self) -> str | None:
        """Return the most recent accepted state digest, or ``None`` if empty."""

        for event in reversed(self._events):
            if event.kind is EventKind.RESULT and event.state_digest_after is not None:
                return event.state_digest_after
        return None

    # --- append -----------------------------------------------------------

    def record_action(
        self,
        *,
        action: str,
        payload: Mapping[str, Any],
        prior_revision: str | None,
        recorded_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        rollback_target_sequence: int | None = None,
    ) -> TrajectoryEvent:
        if not action:
            raise TrajectoryError("action name is required")
        _validate_payload_shape(payload)
        sequence = len(self._events) + 1
        _assert_sequence_parity(sequence, EventKind.ACTION)
        if rollback_target_sequence is not None:
            if action != "rollback":
                raise TrajectoryError(
                    "rollback_target_sequence is only valid on the rollback action"
                )
            if rollback_target_sequence < 1 or rollback_target_sequence >= sequence:
                raise TrajectoryError(
                    "rollback_target_sequence must reference a prior action event"
                )
        event = TrajectoryEvent(
            sequence=sequence,
            kind=EventKind.ACTION,
            action=action,
            payload=dict(payload),
            payload_digest=payload_digest(payload),
            parent_sequence=None,
            prior_revision=prior_revision,
            result_revision=None,
            state_digest_after=None,
            status=None,
            error=None,
            rollback_target_sequence=rollback_target_sequence,
            recorded_at=recorded_at or self._clock(),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    def record_result(
        self,
        *,
        status: EventStatus,
        result_revision: str | None = None,
        state_digest_after: str | None = None,
        error: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryEvent:
        if status is EventStatus.ROLLED_BACK:
            raise TrajectoryError(
                "ROLLED_BACK is applied to the parent ACTION during projection; "
                "the rollback action's own RESULT must be ACCEPTED"
            )
        if not self._events:
            raise TrajectoryError("cannot record a RESULT without a preceding ACTION")
        sequence = len(self._events) + 1
        _assert_sequence_parity(sequence, EventKind.RESULT)
        parent = self._events[-1]
        if parent.kind is not EventKind.ACTION:
            raise TrajectoryError("RESULT must immediately follow an ACTION event")
        if status is EventStatus.ACCEPTED:
            if result_revision is None:
                raise TrajectoryError("accepted RESULT requires a result_revision")
            if state_digest_after is None:
                raise TrajectoryError("accepted RESULT requires a state_digest_after")
        event = TrajectoryEvent(
            sequence=sequence,
            kind=EventKind.RESULT,
            action=None,
            payload=None,
            payload_digest=None,
            parent_sequence=parent.sequence,
            prior_revision=parent.prior_revision,
            result_revision=result_revision,
            state_digest_after=state_digest_after,
            status=status,
            error=dict(error) if error else None,
            rollback_target_sequence=None,
            recorded_at=recorded_at or self._clock(),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    # --- projection -------------------------------------------------------

    def pairs(self) -> list[tuple[TrajectoryEvent, TrajectoryEvent]]:
        """Return all (action, result) pairs in append order."""

        pairs_: list[tuple[TrajectoryEvent, TrajectoryEvent]] = []
        index = 0
        while index < len(self._events) - 1:
            action = self._events[index]
            result = self._events[index + 1]
            if action.kind is not EventKind.ACTION or result.kind is not EventKind.RESULT:
                raise TrajectoryError(
                    f"event sequence {action.sequence} is not an ACTION event"
                )
            pairs_.append((action, result))
            index += 2
        return pairs_

    def rolled_back_sequences(self) -> set[int]:
        """Sequences of previously accepted actions that have been rolled back."""

        rolled: set[int] = set()
        for action, result in self.pairs():
            if (
                result.status is EventStatus.ACCEPTED
                and action.action == "rollback"
                and action.rollback_target_sequence is not None
            ):
                rolled.add(action.rollback_target_sequence)
        return rolled

    def is_live(self, action_sequence: int) -> bool:
        """Whether the action's state effect still contributes to current state."""

        return action_sequence not in self.rolled_back_sequences()

    def live_pairs(self) -> list[tuple[TrajectoryEvent, TrajectoryEvent]]:
        """Pairs whose state effect still contributes to current state."""

        return [
            (action, result)
            for action, result in self.pairs()
            if result.status is EventStatus.ACCEPTED and self.is_live(action.sequence)
        ]

    # --- persistence ------------------------------------------------------

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "events": [event.to_dict() for event in self._events],
        }

    @classmethod
    def from_export(
        cls,
        document: Mapping[str, Any],
        *,
        clock: Callable[[], str] | None = None,
    ) -> "Trajectory":
        if document.get("schema_version") != SCHEMA_VERSION:
            raise TrajectoryError(
                f"unsupported schema_version: {document.get('schema_version')!r}"
            )
        if document.get("protocol_version") != PROTOCOL_VERSION:
            raise TrajectoryError(
                f"unsupported protocol_version: {document.get('protocol_version')!r}"
            )
        session_id = str(document.get("session_id") or "")
        if not session_id:
            raise TrajectoryError("exported trajectory is missing session_id")
        events = document.get("events") or []
        if not isinstance(events, Sequence):
            raise TrajectoryError("exported events must be a sequence")
        trajectory = cls(session_id, clock=clock)
        for raw in events:
            if not isinstance(raw, Mapping):
                raise TrajectoryError("exported event must be an object")
            kind_value = raw.get("kind")
            try:
                kind = EventKind(kind_value)
            except ValueError as exc:
                raise TrajectoryError(f"unknown event kind: {kind_value!r}") from exc
            if kind is EventKind.ACTION:
                action = raw.get("action")
                if not action:
                    raise TrajectoryError("exported ACTION is missing action name")
                payload = raw.get("payload") or {}
                if not isinstance(payload, Mapping):
                    raise TrajectoryError("exported ACTION payload must be an object")
                metadata = raw.get("metadata") or {}
                if not isinstance(metadata, Mapping):
                    raise TrajectoryError("exported ACTION metadata must be an object")
                trajectory.record_action(
                    action=str(action),
                    payload=dict(payload),
                    prior_revision=raw.get("prior_revision"),
                    recorded_at=raw.get("recorded_at"),
                    metadata=dict(metadata),
                    rollback_target_sequence=raw.get("rollback_target_sequence"),
                )
            else:
                status_value = raw.get("status")
                try:
                    status = EventStatus(status_value)
                except ValueError as exc:
                    raise TrajectoryError(f"unknown event status: {status_value!r}") from exc
                error = raw.get("error")
                if error is not None and not isinstance(error, Mapping):
                    raise TrajectoryError("exported RESULT error must be an object")
                metadata = raw.get("metadata") or {}
                if not isinstance(metadata, Mapping):
                    raise TrajectoryError("exported RESULT metadata must be an object")
                trajectory.record_result(
                    status=status,
                    result_revision=raw.get("result_revision"),
                    state_digest_after=raw.get("state_digest_after"),
                    error=dict(error) if error else None,
                    recorded_at=raw.get("recorded_at"),
                    metadata=dict(metadata),
                )
        return trajectory
