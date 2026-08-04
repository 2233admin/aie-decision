"""Validation and deterministic replay for exported search ledgers.

The module intentionally depends only on JSON-shaped mappings.  It can therefore
be used by persistence and recovery adapters without importing the search engine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

LEDGER_SCHEMA_VERSION = "1.0.0"
CHECKPOINT_SCHEMA_VERSION = "fermi-search-checkpoint.v1"

_ACTIVATION_STATES = frozenset({"SEED", "EXPAND", "MINIMIZE"})
_COMPLETION_STATES = frozenset({"EVALUATE", "REJECT"})
_TERMINAL_STATES = frozenset({"RESULT", "STOP"})
_STATES = (
    _ACTIVATION_STATES
    | _COMPLETION_STATES
    | _TERMINAL_STATES
    | frozenset({"VALIDATE", "RANK"})
)


class SearchReplayError(ValueError):
    """Raised when a ledger or checkpoint cannot be trusted or replayed."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SearchReplayError("value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SearchReplayError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SearchReplayError(f"{name} must be an array")
    return value


def replay_search_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Verify an exported search ledger and rebuild its recoverable state."""

    source = _object(ledger, "ledger")
    if source.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise SearchReplayError("unsupported ledger schema_version")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SearchReplayError("ledger.run_id is required")
    entries = _array(source.get("entries"), "ledger.entries")

    seen_events: set[str] = set()
    seen_revisions: set[str] = set()
    activated: dict[str, tuple[int, dict[str, Any]]] = {}
    validated: set[str] = set()
    evaluated: dict[str, dict[str, Any]] = {}
    ranked: set[str] = set()
    terminal: dict[str, Any] | None = None
    last_round = 0
    last_state: str | None = None

    for offset, raw_entry in enumerate(entries, start=1):
        entry = _object(raw_entry, f"ledger.entries[{offset - 1}]")
        if entry.get("sequence") != offset:
            raise SearchReplayError(
                f"ledger sequence must be contiguous at entry {offset}"
            )
        if entry.get("record_type") != "search_event":
            raise SearchReplayError(f"entry {offset} is not a search_event")
        payload = _object(entry.get("payload"), f"entry {offset}.payload")
        if entry.get("payload_hash") != _digest(payload):
            raise SearchReplayError(f"payload hash mismatch at entry {offset}")

        event_id = payload.get("event_id")
        revision = _object(payload.get("revision"), f"entry {offset}.payload.revision")
        revision_id = revision.get("revision_id")
        candidate_id = payload.get("candidate_id")
        state = payload.get("state")
        round_index = payload.get("round_index")
        reason = payload.get("reason")
        data = payload.get("data")

        if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise SearchReplayError(f"unsupported event schema at entry {offset}")
        if payload.get("run_id") != run_id:
            raise SearchReplayError(f"run_id mismatch at entry {offset}")
        if not isinstance(event_id, str) or not event_id or event_id in seen_events:
            raise SearchReplayError(f"invalid or duplicate event_id at entry {offset}")
        if entry.get("stable_id") != event_id:
            raise SearchReplayError(f"stable_id mismatch at entry {offset}")
        if (
            not isinstance(revision_id, str)
            or not revision_id
            or revision_id in seen_revisions
        ):
            raise SearchReplayError(
                f"invalid or duplicate revision_id at entry {offset}"
            )
        if entry.get("revision_id") != revision_id:
            raise SearchReplayError(f"revision_id mismatch at entry {offset}")
        if revision.get("sequence") != 1 or revision.get(
            "supersedes_revision_id"
        ) not in (None, ""):
            raise SearchReplayError(
                f"search event revision must be an initial revision at entry {offset}"
            )
        if not isinstance(candidate_id, str):
            raise SearchReplayError(f"candidate_id must be a string at entry {offset}")
        if state not in _STATES:
            raise SearchReplayError(
                f"unsupported search state at entry {offset}: {state}"
            )
        if (
            not isinstance(round_index, int)
            or isinstance(round_index, bool)
            or round_index < 0
        ):
            raise SearchReplayError(f"invalid round_index at entry {offset}")
        if round_index < last_round:
            raise SearchReplayError(f"round_index moved backwards at entry {offset}")
        if not isinstance(reason, str) or not reason:
            raise SearchReplayError(f"reason is required at entry {offset}")
        _object(data, f"entry {offset}.payload.data")
        if terminal is not None:
            raise SearchReplayError("terminal event must be the final ledger entry")

        if state in _ACTIVATION_STATES:
            if not candidate_id or candidate_id in activated:
                raise SearchReplayError(
                    f"candidate activated more than once at entry {offset}"
                )
            activated[candidate_id] = (offset, json.loads(_canonical(data)))
        elif state == "VALIDATE":
            if (
                candidate_id not in activated
                or candidate_id in evaluated
                or candidate_id in validated
            ):
                raise SearchReplayError(
                    f"invalid VALIDATE transition at entry {offset}"
                )
            validated.add(candidate_id)
        elif state in _COMPLETION_STATES:
            if candidate_id not in validated or candidate_id in evaluated:
                raise SearchReplayError(f"invalid {state} transition at entry {offset}")
            evaluated[candidate_id] = {
                "candidate_id": candidate_id,
                "outcome": state,
                "round_index": round_index,
                "reason": reason,
                "data": json.loads(_canonical(data)),
            }
        elif state == "RANK":
            if (
                candidate_id not in evaluated
                or evaluated[candidate_id]["outcome"] != "EVALUATE"
                or candidate_id in ranked
            ):
                raise SearchReplayError(f"invalid RANK transition at entry {offset}")
            ranked.add(candidate_id)
        else:
            if state == "RESULT" and (
                not candidate_id or candidate_id not in evaluated
            ):
                raise SearchReplayError("RESULT must identify an evaluated candidate")
            if state == "STOP" and candidate_id:
                raise SearchReplayError("STOP must not identify a candidate")
            terminal = {
                "state": state,
                "candidate_id": candidate_id or None,
                "round_index": round_index,
                "reason": reason,
                "data": json.loads(_canonical(data)),
            }

        seen_events.add(event_id)
        seen_revisions.add(revision_id)
        last_round = round_index
        last_state = state

    pending = [
        candidate_id
        for candidate_id, _ in sorted(activated.items(), key=lambda item: item[1][0])
        if candidate_id not in evaluated
    ]
    pending_records = [
        {"candidate_id": candidate_id, **activated[candidate_id][1]}
        for candidate_id in pending
    ]
    evaluated_rows = [evaluated[candidate_id] for candidate_id in sorted(evaluated)]
    return {
        "schema_version": "fermi-search-replay.v1",
        "run_id": run_id,
        "event_count": len(entries),
        "current_state": last_state,
        "terminal": terminal,
        "evaluated_candidates": evaluated_rows,
        "pending_candidates": pending,
        "pending_candidate_records": pending_records,
    }


def create_search_checkpoint(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Create a self-verifying, JSON-serializable checkpoint from a ledger."""

    replay = replay_search_ledger(ledger)
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "ledger": json.loads(_canonical(ledger)),
        "replay": replay,
    }
    return {**body, "checkpoint_hash": _digest(body)}


def verify_search_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Verify checkpoint integrity and return a freshly reconstructed state."""

    source = _object(checkpoint, "checkpoint")
    if source.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SearchReplayError("unsupported checkpoint schema_version")
    supplied_hash = source.get("checkpoint_hash")
    body = {key: value for key, value in source.items() if key != "checkpoint_hash"}
    if not isinstance(supplied_hash, str) or supplied_hash != _digest(body):
        raise SearchReplayError("checkpoint hash mismatch")
    replay = replay_search_ledger(_object(source.get("ledger"), "checkpoint.ledger"))
    if source.get("replay") != replay:
        raise SearchReplayError("checkpoint replay state mismatch")
    return replay
