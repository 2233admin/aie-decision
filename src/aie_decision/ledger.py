"""Append-only in-memory ledger with deterministic revision lineage checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .models import SCHEMA_VERSION, to_dict
from .validation import validate_model


class LedgerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    sequence: int
    record_type: str
    stable_id: str
    revision_id: str
    recorded_at: str
    payload_hash: str
    payload: dict[str, Any]


_ID_FIELDS = (
    # Prefer the record's own leaf identity before provenance/parent identities.
    # Evidence propositions contain both source_id and evidence_atom_id; choosing
    # source_id collapsed every atom from one source into a false revision chain.
    "event_id", "candidate_id", "package_id", "evidence_atom_id", "scene_id", "estimate_id", "factor_id",
    "evaluation_id", "update_id", "graph_id", "source_id", "question_id",
)


class AnalysisLedger:
    """A process-local immutable ledger suitable for later persistence adapters."""

    def __init__(self, run_id: str):
        if not run_id:
            raise LedgerError("run_id is required")
        self.run_id = run_id
        self._entries: list[LedgerEntry] = []

    def append(self, record_type: str, record: Any) -> LedgerEntry:
        payload = to_dict(record)
        if not isinstance(payload, dict):
            raise LedgerError("ledger record must serialize to an object")
        if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise LedgerError(f"unsupported schema version: {payload.get('schema_version')}")
        try:
            validate_model(record)
        except ValueError as exc:
            if "unsupported model type" not in str(exc):
                raise
        stable_id = next((str(payload[name]) for name in _ID_FIELDS if payload.get(name)), "")
        if not stable_id:
            raise LedgerError("record has no stable identifier")
        revision = payload.get("revision")
        if not isinstance(revision, dict) or not revision.get("revision_id"):
            raise LedgerError("record has no revision metadata")
        revision_id = str(revision["revision_id"])
        if any(entry.revision_id == revision_id for entry in self._entries):
            raise LedgerError(f"revision already exists: {revision_id}")
        previous = self.latest(record_type, stable_id)
        if previous is None:
            if revision.get("sequence") != 1 or revision.get("supersedes_revision_id"):
                raise LedgerError("first ledger revision must have sequence 1 and no predecessor")
        else:
            if revision.get("sequence") != previous.payload["revision"]["sequence"] + 1:
                raise LedgerError("revision sequence must increase by exactly one")
            if revision.get("supersedes_revision_id") != previous.revision_id:
                raise LedgerError("revision must supersede the latest ledger revision")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        frozen_payload = json.loads(canonical)
        entry = LedgerEntry(
            sequence=len(self._entries) + 1,
            record_type=record_type,
            stable_id=stable_id,
            revision_id=revision_id,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            payload_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            payload=frozen_payload,
        )
        self._entries.append(entry)
        return entry

    def records(self, record_type: str | None = None) -> tuple[LedgerEntry, ...]:
        if record_type is None:
            return tuple(self._entries)
        return tuple(entry for entry in self._entries if entry.record_type == record_type)

    def latest(self, record_type: str, stable_id: str) -> LedgerEntry | None:
        return next(
            (entry for entry in reversed(self._entries) if entry.record_type == record_type and entry.stable_id == stable_id),
            None,
        )

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "entries": [
                {
                    "sequence": entry.sequence,
                    "record_type": entry.record_type,
                    "stable_id": entry.stable_id,
                    "revision_id": entry.revision_id,
                    "recorded_at": entry.recorded_at,
                    "payload_hash": entry.payload_hash,
                    "payload": entry.payload,
                }
                for entry in self._entries
            ],
        }
