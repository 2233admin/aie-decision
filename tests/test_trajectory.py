"""Tests for the append-only trajectory module."""

from __future__ import annotations

import json

import pytest

from aie_decision.trajectory import (
    EventKind,
    EventStatus,
    Trajectory,
    TrajectoryError,
    canonical_bytes,
    payload_digest,
    state_digest,
    utc_now,
)


def test_canonical_bytes_is_key_sorted_and_stable():
    first = canonical_bytes({"b": 2, "a": 1})
    second = canonical_bytes({"a": 1, "b": 2})
    assert first == second
    assert first == b'{"a":1,"b":2}'


def test_payload_and_state_digests_ignore_key_order():
    assert payload_digest({"x": 1, "y": 2}) == payload_digest({"y": 2, "x": 1})
    assert state_digest({"a": [1, 2], "b": "c"}) == state_digest({"b": "c", "a": [1, 2]})


def test_record_action_requires_name_and_stores_canonical_digest():
    trajectory = Trajectory("s1")
    event = trajectory.record_action(action="expand", payload={"node": 1}, prior_revision=None)
    assert event.sequence == 1
    assert event.kind is EventKind.ACTION
    assert event.payload_digest == payload_digest({"node": 1})
    assert event.prior_revision is None
    assert event.result_revision is None


def test_record_result_must_immediately_follow_action():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="expand", payload={}, prior_revision=None)
    result = trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="r", state_digest_after="r")
    assert result.parent_sequence == 1
    assert result.sequence == 2
    assert result.status is EventStatus.ACCEPTED


def test_illegal_order_result_before_action_raises():
    trajectory = Trajectory("s1")
    with pytest.raises(TrajectoryError):
        trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="r", state_digest_after="r")


def test_result_without_action_raises():
    trajectory = Trajectory("s1")
    with pytest.raises(TrajectoryError):
        trajectory.record_result(status=EventStatus.REJECTED, error={"issues": [{"code": "x", "path": "$", "message": "y"}]})


def test_accepted_result_requires_revisions():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="expand", payload={}, prior_revision=None)
    with pytest.raises(TrajectoryError):
        trajectory.record_result(status=EventStatus.ACCEPTED)


def test_rolled_back_status_cannot_be_recorded_directly():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="expand", payload={}, prior_revision=None)
    with pytest.raises(TrajectoryError):
        trajectory.record_result(status=EventStatus.ROLLED_BACK)


def test_rollback_action_carries_target_sequence():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="expand", payload={"i": 1}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d1", state_digest_after="d1")
    rollback = trajectory.record_action(
        action="rollback",
        payload={"target_sequence": 1},
        prior_revision="d1",
        rollback_target_sequence=1,
    )
    assert rollback.rollback_target_sequence == 1


def test_rollback_target_must_reference_prior_action():
    trajectory = Trajectory("s1")
    with pytest.raises(TrajectoryError):
        trajectory.record_action(
            action="rollback",
            payload={},
            prior_revision=None,
            rollback_target_sequence=5,
        )


def test_pairs_returns_action_result_pairs():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="expand", payload={}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="r1", state_digest_after="r1")
    trajectory.record_action(action="estimate", payload={}, prior_revision="r1")
    trajectory.record_result(status=EventStatus.REJECTED, error={"issues": []})
    pairs = trajectory.pairs()
    assert [a.action for a, _ in pairs] == ["expand", "estimate"]
    assert [r.status for _, r in pairs] == [EventStatus.ACCEPTED, EventStatus.REJECTED]


def test_rolled_back_sequences_tracks_accepted_rollbacks():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="expand", payload={}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d1", state_digest_after="d1")
    trajectory.record_action(
        action="rollback",
        payload={"target_sequence": 1},
        prior_revision="d1",
        rollback_target_sequence=1,
    )
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d0", state_digest_after="d0")
    assert trajectory.rolled_back_sequences() == {1}
    assert trajectory.is_live(1) is False


def test_live_pairs_excludes_rolled_back():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="start", payload={"q": "x"}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d0", state_digest_after="d0")
    trajectory.record_action(action="expand", payload={}, prior_revision="d0")
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d1", state_digest_after="d1")
    trajectory.record_action(
        action="rollback",
        payload={"target_sequence": 3},
        prior_revision="d1",
        rollback_target_sequence=3,
    )
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d0", state_digest_after="d0")
    live = trajectory.live_pairs()
    assert [a.action for a, _ in live] == ["start", "rollback"]


def test_rejected_event_does_not_change_revision():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="start", payload={}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d0", state_digest_after="d0")
    trajectory.record_action(action="expand", payload={"bad": True}, prior_revision="d0")
    trajectory.record_result(status=EventStatus.REJECTED, error={"issues": [{"code": "x", "path": "$", "message": "y"}]})
    assert trajectory.last_revision() == "d0"


def test_export_round_trip_preserves_all_events():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="start", payload={"q": "x"}, prior_revision=None, recorded_at="t1")
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d0", state_digest_after="d0", recorded_at="t2")
    trajectory.record_action(action="expand", payload={"i": 1}, prior_revision="d0", recorded_at="t3")
    trajectory.record_result(status=EventStatus.REJECTED, error={"issues": [{"code": "x", "path": "$", "message": "y"}]}, recorded_at="t4")
    exported = trajectory.export()
    assert exported["session_id"] == "s1"
    assert len(exported["events"]) == 4
    rehydrated = Trajectory.from_export(exported)
    assert rehydrated.events[0].payload == {"q": "x"}
    assert rehydrated.events[1].status is EventStatus.ACCEPTED
    assert rehydrated.events[3].status is EventStatus.REJECTED


def test_from_export_rejects_unknown_schema_version():
    with pytest.raises(TrajectoryError):
        Trajectory.from_export({"schema_version": "0.0.0", "protocol_version": "track-c/1.0.0", "session_id": "s", "events": []})


def test_from_export_rejects_unknown_protocol_version():
    with pytest.raises(TrajectoryError):
        Trajectory.from_export({"schema_version": "1.0.0", "protocol_version": "x", "session_id": "s", "events": []})


def test_export_payload_is_json_serialisable():
    trajectory = Trajectory("s1")
    trajectory.record_action(action="start", payload={"q": "x"}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d", state_digest_after="d")
    text = json.dumps(trajectory.export())
    payload = json.loads(text)
    assert payload["session_id"] == "s1"


def test_last_revision_returns_latest_accepted_state_digest():
    trajectory = Trajectory("s1")
    assert trajectory.last_revision() is None
    trajectory.record_action(action="start", payload={}, prior_revision=None)
    trajectory.record_result(status=EventStatus.ACCEPTED, result_revision="d0", state_digest_after="d0")
    trajectory.record_action(action="expand", payload={}, prior_revision="d0")
    trajectory.record_result(status=EventStatus.REJECTED, error={"issues": [{"code": "x", "path": "$", "message": "y"}]})
    assert trajectory.last_revision() == "d0"


def test_utc_now_is_iso_format():
    timestamp = utc_now()
    assert "T" in timestamp
    assert timestamp.endswith("+00:00") or timestamp.endswith("Z")
