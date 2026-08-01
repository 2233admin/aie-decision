"""Selective recomputation and immutable run comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .exporting import to_primitive


STAGES = ("answer", "decomposition", "evidence", "measurement", "factors", "interval", "package")


def stable_digest(value: Any) -> str:
    payload = json.dumps(to_primitive(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RecomputePlan:
    reasons: tuple[str, ...]
    stages: tuple[str, ...]


def plan_recomputation(
    *,
    answer_changed: bool = False,
    sources_changed: bool = False,
    assumptions_changed: bool = False,
    bounds_changed: bool = False,
) -> RecomputePlan:
    reasons: list[str] = []
    first_stage: str | None = None
    for changed, reason, stage in (
        (answer_changed, "answer_contract_changed", "answer"),
        (sources_changed, "sources_changed", "evidence"),
        (assumptions_changed, "assumptions_changed", "measurement"),
        (bounds_changed, "bounds_changed", "measurement"),
    ):
        if changed:
            reasons.append(reason)
            if first_stage is None or STAGES.index(stage) < STAGES.index(first_stage):
                first_stage = stage
    if first_stage is None:
        return RecomputePlan((), ())
    return RecomputePlan(tuple(reasons), STAGES[STAGES.index(first_stage) :])


def compare_runs(before: Mapping[str, Any], after: Mapping[str, Any], fields: Iterable[str] | None = None) -> dict[str, Any]:
    selected = tuple(fields or sorted(set(before) | set(after)))
    changes: dict[str, Any] = {}
    for field in selected:
        old = to_primitive(before.get(field))
        new = to_primitive(after.get(field))
        if stable_digest(old) != stable_digest(new):
            changes[field] = {"before": old, "after": new}
    return {
        "before_revision": before.get("revision_id") or before.get("revision"),
        "after_revision": after.get("revision_id") or after.get("revision"),
        "changed_fields": changes,
    }
