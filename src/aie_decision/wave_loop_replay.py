"""Ledger replay and checkpoint verification for the wave loop.

This module verifies append-only wave ledgers, reconstructs loop state from
the event stream, and provides self-verifying checkpoints for cold-start
Agent replay.  All verification is fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .models import SCHEMA_VERSION
from .wave_loop_contract import (
    WAVE_ACTIVATION_STATES,
    WAVE_CHECKPOINT_VERSION,
    WAVE_LEDGER_VERSION,
    WAVE_TERMINAL_STATES,
    WaveLoopError,
)


# ---------------------------------------------------------------------------
# Cryptographic helpers
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WaveLoopError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WaveLoopError(f"{name} must be an array")
    return value


# ---------------------------------------------------------------------------
# Ledger replay
# ---------------------------------------------------------------------------


def replay_wave_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a wave ledger and reconstruct the loop state from events."""

    source = _object(ledger, "ledger")
    if source.get("schema_version") not in (WAVE_LEDGER_VERSION, SCHEMA_VERSION):
        raise WaveLoopError("unsupported wave ledger schema_version")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise WaveLoopError("ledger.run_id is required")
    entries = _array(source.get("entries"), "ledger.entries")

    seen_events: set[str] = set()
    seen_revisions: set[str] = set()
    events: list[dict[str, Any]] = []
    surfaces: dict[str, list[str]] = {}
    actions: list[dict[str, Any]] = []
    last_round = 0
    last_state: str | None = None
    accepted = False
    terminal: dict[str, Any] | None = None

    for offset, raw_entry in enumerate(entries, start=1):
        entry = _object(raw_entry, f"ledger.entries[{offset - 1}]")
        if entry.get("sequence") != offset:
            raise WaveLoopError(f"wave sequence must be contiguous at entry {offset}")
        if entry.get("record_type") != "wave_event":
            raise WaveLoopError(f"entry {offset} is not a wave_event")
        payload = _object(entry.get("payload"), f"entry {offset}.payload")
        if entry.get("payload_hash") != _digest(payload):
            raise WaveLoopError(f"payload hash mismatch at entry {offset}")
        event_id = str(payload.get("event_id", ""))
        revision = _object(payload.get("revision"), f"entry {offset}.payload.revision")
        revision_id = str(revision.get("revision_id", ""))
        surface_id = str(payload.get("surface_id", ""))
        state = str(payload.get("state", ""))
        round_index = payload.get("round_index")
        reason = str(payload.get("reason", ""))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise WaveLoopError(f"unsupported event schema at entry {offset}")
        if payload.get("run_id") != run_id:
            raise WaveLoopError(f"run_id mismatch at entry {offset}")
        if not event_id or event_id in seen_events:
            raise WaveLoopError(f"invalid or duplicate event_id at entry {offset}")
        if entry.get("stable_id") != event_id:
            raise WaveLoopError(f"stable_id mismatch at entry {offset}")
        if not revision_id or revision_id in seen_revisions:
            raise WaveLoopError(f"invalid or duplicate revision_id at entry {offset}")
        if entry.get("revision_id") != revision_id:
            raise WaveLoopError(f"revision_id mismatch at entry {offset}")
        if revision.get("sequence") != 1 or revision.get("supersedes_revision_id") not in (
            None,
            "",
        ):
            raise WaveLoopError(
                f"wave event revision must be an initial revision at entry {offset}"
            )
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
            raise WaveLoopError(f"invalid round_index at entry {offset}")
        if round_index < last_round:
            raise WaveLoopError(f"round_index moved backwards at entry {offset}")
        if state not in WAVE_ACTIVATION_STATES | WAVE_TERMINAL_STATES:
            raise WaveLoopError(f"unsupported wave state at entry {offset}: {state}")
        if not reason:
            raise WaveLoopError(f"reason is required at entry {offset}")
        if terminal is not None:
            raise WaveLoopError("terminal event must be the final ledger entry")

        if state == "DRAFT":
            surfaces.setdefault(surface_id, []).append("DRAFT")
        elif state == "EVALUATED":
            surfaces.setdefault(surface_id, []).append("EVALUATED")
        elif state == "REFINING":
            for record_actions in payload.get("data", {}).get("actions_emitted", ()):
                actions.append(
                    {"surface_id": surface_id, "action_kind": str(record_actions)}
                )
            surfaces.setdefault(surface_id, []).append("REFINING")
        elif state == "ACCEPTED":
            accepted = True
            surfaces.setdefault(surface_id, []).append("ACCEPTED")
            terminal = {
                "state": state,
                "surface_id": surface_id,
                "round_index": round_index,
                "reason": reason,
            }
        elif state == "UNRESOLVED":
            surfaces.setdefault(surface_id, []).append("UNRESOLVED")
            terminal = {
                "state": state,
                "surface_id": surface_id,
                "round_index": round_index,
                "reason": reason,
            }
        elif state == "STOP":
            terminal = {
                "state": state,
                "surface_id": surface_id,
                "round_index": round_index,
                "reason": reason,
            }
        else:
            raise WaveLoopError(f"unsupported activation state at entry {offset}: {state}")

        events.append(
            {
                "event_id": event_id,
                "surface_id": surface_id,
                "state": state,
                "round_index": round_index,
                "reason": reason,
                "data": json.loads(_canonical(payload.get("data", {}))),
            }
        )
        seen_events.add(event_id)
        seen_revisions.add(revision_id)
        last_round = round_index
        last_state = state

    return {
        "schema_version": "joint-wave-replay.v1",
        "run_id": run_id,
        "event_count": len(entries),
        "current_state": last_state,
        "terminal": terminal,
        "accepted": accepted,
        "surfaces": {surface_id: list(states) for surface_id, states in surfaces.items()},
        "actions": actions,
        "events": events,
    }


def create_wave_checkpoint(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Build a self-verifying checkpoint from a wave ledger."""

    replay = replay_wave_ledger(ledger)
    body = {
        "schema_version": WAVE_CHECKPOINT_VERSION,
        "ledger": json.loads(_canonical(ledger)),
        "replay": replay,
    }
    return {**body, "checkpoint_hash": _digest(body)}


def verify_wave_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a wave checkpoint and return the reconstructed loop state."""

    source = _object(checkpoint, "checkpoint")
    if source.get("schema_version") != WAVE_CHECKPOINT_VERSION:
        raise WaveLoopError("unsupported wave checkpoint schema_version")
    supplied_hash = source.get("checkpoint_hash")
    body = {key: value for key, value in source.items() if key != "checkpoint_hash"}
    if not isinstance(supplied_hash, str) or supplied_hash != _digest(body):
        raise WaveLoopError("checkpoint hash mismatch")
    replay = replay_wave_ledger(_object(source.get("ledger"), "checkpoint.ledger"))
    if source.get("replay") != replay:
        raise WaveLoopError("checkpoint replay state mismatch")
    return replay


__all__ = [
    "_canonical",
    "_digest",
    "create_wave_checkpoint",
    "replay_wave_ledger",
    "verify_wave_checkpoint",
]
