"""Deterministic projection of factual propositions into event scenes.

The extractor is deliberately conservative: it only supplies fields stated in
the proposition itself.  It does not promote evaluations, rhetoric, forecasts,
or attributed allegations into settled event facts; admissibility remains the
responsibility of :mod:`aie_decision.evidence`.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable, Mapping

from .evidence import reconstruct_event_scene as _settle_event_scene
from .models import EpistemicType, EventScene, EventStatus, EvidenceProposition, SceneEvent


_TIME_PATTERNS = (
    re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}月\d{1,2}日(?:起)?"),
    re.compile(r"(?<!\d)\d{1,2}月(?!\d)"),
    re.compile(r"第[一二三四]季度"),
    re.compile(r"(?:上|下|本|这)月"),
)

_PLACE_AFTER_MOTION = re.compile(
    r"(?:抵达|到达|进入|离开|撤离|疏散(?:至)?|封锁|接走|发生在|位于)"
    r"(?P<place>[\u4e00-\u9fffA-Za-z0-9]*?(?:仓库|重症室|实验室|站|线|片|区|港|闸口|平台|现场))"
)
_PLACE_WITH_PREPOSITION = re.compile(
    r"(?:在|于)(?P<place>[\u4e00-\u9fffA-Za-z0-9]*?(?:仓库|重症室|实验室|站|线|片|区|港|闸口|平台|现场))"
)
_PLACE_AS_SUBJECT = re.compile(
    r"(?P<place>[\u4e00-\u9fffA-Za-z0-9]*?(?:仓库|重症室|实验室|站|线|片|区|港|闸口|平台|现场))"
    r"(?=\d{1,2}:\d{2}|(?:停止|中断|恢复|启动|关闭|开放|封锁|疏散))"
)
_PLACE_BEFORE_MOTION = re.compile(
    r"(?:命令|要求|通知)(?P<place>[\u4e00-\u9fffA-Za-z0-9]*?(?:仓库|重症室|实验室|站|线|片|区|港|闸口|平台|现场))"
    r"(?:疏散|撤离|关闭|停运|封锁)"
)

_ACTION_MARKERS = (
    "通过", "生效", "执行", "公布", "发布", "颁布", "暂停", "恢复", "招标", "授标",
    "签署", "接警", "抵达", "到达", "控制", "停车", "封锁", "批准", "中断", "启动",
    "超", "命令", "接走", "停止", "集合", "取消", "列示", "增加", "减少", "发现", "关闭",
    "离开", "开始", "休庭", "建议", "超标", "列入", "完成",
)
_OMISSION_CUES = ("未提供", "未发布", "未显示", "未填", "尚未", "栏为空", "没有记录", "没有分")
_RHETORIC_CUES = ("显然", "最好", "最糟糕", "无可争辩", "最明智", "粉饰", "演戏")
_FORECAST_CUES = ("预测", "预计")
_ATTRIBUTION = re.compile(r"(?:称|说|表示|认为)")
_STATUS_STANCE = re.compile(
    r"(?P<topic>[\u4e00-\u9fffA-Za-z0-9]{1,12}?)(?:已经|仍|将|会|已)?"
    r"(?P<value>破裂|继续|取消|恢复|解除|撤销|生效|暂停|停止)"
)
_EXPLICIT_DISPUTE = re.compile(r"(?P<topic>日期|时间|原因|数值|金额|数量|责任)(?:存在|有)?争议")


def _first_match(patterns: Iterable[re.Pattern[str]], text: str) -> str | None:
    matches = [match for pattern in patterns if (match := pattern.search(text))]
    if not matches:
        return None
    match = min(matches, key=lambda item: item.start())
    return match.group(0)


def _place(text: str) -> str | None:
    for pattern in (
        _PLACE_AFTER_MOTION,
        _PLACE_WITH_PREPOSITION,
        _PLACE_BEFORE_MOTION,
        _PLACE_AS_SUBJECT,
    ):
        if match := pattern.search(text):
            value = match.group("place")
            # A place noun may be preceded by a temporal adjunct.  That
            # adjunct belongs to the event time, never to the place identity.
            for pattern in _TIME_PATTERNS:
                value = pattern.sub("", value)
            return value.strip(" ，,。；;") or None
    return None


def _actors(text: str) -> tuple[str, ...]:
    """Extract an explicit grammatical actor; never infer one from authorship."""

    candidates: list[tuple[int, str]] = []
    if time := _first_match(_TIME_PATTERNS, text):
        index = text.find(time)
        if index > 0:
            candidates.append((index, text[:index]))
    for marker in _ACTION_MARKERS:
        index = text.find(marker)
        if index > 0:
            candidates.append((index, text[:index]))
    if not candidates:
        return ()
    prefix = min(candidates, key=lambda item: item[0])[1].strip(" ，,。；;：:")
    for pattern in _TIME_PATTERNS:
        prefix = pattern.sub("", prefix)
    prefix = re.sub(r"^(?:并|随后|之后|此前|同时)", "", prefix).strip()
    if not prefix or len(prefix) > 16 or _ATTRIBUTION.search(prefix):
        return ()
    return (prefix,)


def _object(text: str) -> str | None:
    """Return a modest verb complement, only for audit/query convenience."""

    positions = [(text.find(marker), marker) for marker in _ACTION_MARKERS if marker in text]
    if not positions:
        return None
    index, marker = min(positions, key=lambda item: item[0])
    complement = text[index + len(marker) :].strip(" ，,。；;：:")
    if not complement:
        return None
    time = _first_match(_TIME_PATTERNS, complement)
    if time and complement == time:
        return None
    return complement


def infer_scene_fact_fields(
    propositions: Iterable[EvidenceProposition],
) -> dict[str, dict[str, Any]]:
    """Infer literal scene fields from public proposition text.

    The result is keyed by evidence atom id so downstream settlement retains
    exact provenance.  Non-factual proposition types receive no inferred
    fields and therefore can never be laundered into events by this helper.
    """

    inferred: dict[str, dict[str, Any]] = {}
    factual_index = 0
    for atom in propositions:
        if atom.epistemic_type not in {
            EpistemicType.OBSERVED_EVENT,
            EpistemicType.PRIMARY_RECORD,
        }:
            continue
        factual_index += 1
        fields: dict[str, Any] = {
            "action": atom.claim.strip(),
            "sequence": factual_index,
        }
        if actor_values := _actors(atom.claim):
            fields["actors"] = actor_values
        else:
            # An article/source speaker is provenance, not proof that the
            # speaker performed an event with no stated grammatical actor.
            fields["actors"] = ()
        if event_time := (atom.event_time or _first_match(_TIME_PATTERNS, atom.claim)):
            fields["time"] = event_time
        if place := _place(atom.claim):
            fields["place"] = place
        if object_value := _object(atom.claim):
            fields["object"] = object_value
        inferred[atom.evidence_atom_id] = fields
    return inferred


def _safe_scene_type(atom: EvidenceProposition) -> EpistemicType:
    """Fail closed when a coarse upstream type contradicts explicit wording."""

    if atom.epistemic_type not in {EpistemicType.OBSERVED_EVENT, EpistemicType.PRIMARY_RECORD}:
        return atom.epistemic_type
    text = atom.claim
    if any(cue in text for cue in _OMISSION_CUES):
        return EpistemicType.OMISSION
    if any(cue in text for cue in _RHETORIC_CUES):
        return EpistemicType.RHETORIC
    if any(cue in text for cue in _FORECAST_CUES):
        return EpistemicType.FORECAST
    if _ATTRIBUTION.search(text):
        return EpistemicType.ATTRIBUTED_STATEMENT
    return atom.epistemic_type


def _source_disputes(atoms: tuple[EvidenceProposition, ...]) -> tuple[dict[str, Any], ...]:
    """Preserve conflicting attributed statuses without settling either one."""

    stances: dict[str, list[tuple[str, str]]] = {}
    for atom in atoms:
        text = atom.claim
        # Drop an attribution prefix for topic extraction only; the original
        # claim and atom remain unchanged and queryable.
        if _ATTRIBUTION.search(text):
            text = _ATTRIBUTION.split(text, maxsplit=1)[-1]
        if "因" in text:
            text = text.rsplit("因", maxsplit=1)[-1]
        match = _STATUS_STANCE.search(text)
        if not match:
            continue
        topic = match.group("topic").strip(" ，,。；;")
        value = match.group("value")
        stances.setdefault(topic, []).append((atom.evidence_atom_id, value))

    # An explicit dispute over a date/time settles neither attributed value,
    # but the competing reported values themselves remain auditable.
    for atom in atoms:
        match = _EXPLICIT_DISPUTE.search(atom.claim)
        if not match:
            continue
        topic = match.group("topic")
        if topic not in {"日期", "时间"}:
            continue
        values = []
        for candidate in atoms:
            value = candidate.event_time or _first_match(_TIME_PATTERNS, candidate.claim)
            if value and candidate.evidence_atom_id != atom.evidence_atom_id:
                values.append((candidate.evidence_atom_id, value))
        if len({value for _, value in values}) >= 2:
            stances.setdefault(topic, []).extend(values)

    disputes = []
    for topic, values in stances.items():
        if len({value for _, value in values}) < 2:
            continue
        disputes.append(
            {
                "type": "dispute",
                "event_key": topic,
                "field": "time" if topic in {"日期", "时间"} else "status",
                "competing_values": tuple(values),
                "settlement": "unresolved_source_positions",
            }
        )
    return tuple(disputes)


def reconstruct_event_scene(
    *,
    question_id: str,
    propositions: Iterable[EvidenceProposition],
    fact_fields: Mapping[str, Mapping[str, Any]] | None = None,
    scene_id: str | None = None,
) -> EventScene:
    """Project literal scene fields, then apply the existing settlement rules.

    Caller-supplied fields override inferred values one field at a time.  This
    keeps precise external extraction contracts authoritative while making the
    ordinary public-input path useful without fixture-specific bindings.
    """

    atoms = tuple(propositions)
    merged: dict[str, dict[str, Any]] = infer_scene_fact_fields(atoms)
    for atom_id, supplied in (fact_fields or {}).items():
        merged.setdefault(atom_id, {}).update(dict(supplied))
    # A primary record is admissible as evidence that the record makes the
    # assertion even when its real-world truth has not been independently
    # settled.  Feed that minimum record support to the scene settler, then
    # label the resulting event ``claimed_only`` rather than ``confirmed``.
    safe_types = {atom.evidence_atom_id: _safe_scene_type(atom) for atom in atoms}
    claimed_only_atom_ids = {
        atom.evidence_atom_id
        for atom in atoms
        if safe_types[atom.evidence_atom_id] in {EpistemicType.OBSERVED_EVENT, EpistemicType.PRIMARY_RECORD}
        and atom.truth_confidence is None
        and atom.extraction_confidence is not None
        and atom.extraction_confidence >= 0.5
        and atom.provenance
    }
    settlement_atoms = []
    for atom in atoms:
        updates: dict[str, Any] = {}
        safe_type = safe_types[atom.evidence_atom_id]
        if safe_type is not atom.epistemic_type:
            updates["epistemic_type"] = safe_type
            if safe_type in {EpistemicType.RHETORIC, EpistemicType.EVALUATION}:
                updates["source_position"] = atom.source_position or atom.claim
        if atom.evidence_atom_id in claimed_only_atom_ids:
            updates["truth_confidence"] = 0.5
        settlement_atoms.append(replace(atom, **updates) if updates else atom)
    scene = _settle_event_scene(
        question_id=question_id,
        propositions=tuple(settlement_atoms),
        fact_fields=merged,
        scene_id=scene_id,
    )

    # A scene is chronological by evidence order even where no explicit ordinal
    # was printed.  The relation is a projection, not a new factual assertion.
    events = tuple(
        SceneEvent(
            event_id=event.event_id,
            status=(
                EventStatus.CLAIMED_ONLY
                if set(event.supporting_atom_ids) & claimed_only_atom_ids
                and event.status is EventStatus.CONFIRMED
                else event.status
            ),
            actor_ids=event.actor_ids,
            action=event.action,
            supporting_atom_ids=event.supporting_atom_ids,
            counter_atom_ids=event.counter_atom_ids,
            time_window=event.time_window,
            unknown_fields=event.unknown_fields,
        )
        for event in scene.events
    )
    existing_disputes = {
        (relation.get("event_key"), relation.get("field"))
        for relation in scene.relations
        if relation.get("type") == "dispute"
    }
    inferred_disputes = tuple(
        relation
        for relation in _source_disputes(atoms)
        if (relation.get("event_key"), relation.get("field")) not in existing_disputes
    )
    return replace(scene, events=events, relations=tuple(scene.relations) + inferred_disputes)
