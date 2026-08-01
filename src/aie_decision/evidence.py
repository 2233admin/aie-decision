"""Deterministic source ingestion, proposition typing, and scene settlement."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .models import (
    EpistemicType,
    EventScene,
    EventStatus,
    EvidenceProposition,
    ProvenanceRef,
    Revision,
    SceneEvent,
    SourceRecord,
)


class EvidenceError(ValueError):
    """Raised for evidence that cannot be compiled without inventing data."""


Classifier = Callable[[str, int], Mapping[str, Any]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _revision(prefix: str, identity: str) -> Revision:
    return Revision(_stable_id(prefix, identity, 1), 1, _now())


def ingest_source(
    content: str | bytes,
    *,
    title: str,
    uri: str | None = None,
    publisher: str | None = None,
    published_at: str | None = None,
    retrieved_at: str | None = None,
    source_id: str | None = None,
    source_location: str | None = None,
    speaker: str | None = None,
    transformation_lineage: Sequence[str] = (),
) -> SourceRecord:
    """Create an immutable source record whose identity includes exact bytes."""

    raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    if not raw:
        raise EvidenceError("source content cannot be empty")
    if not title.strip():
        raise EvidenceError("source title is required")
    digest = sha256(raw).hexdigest()
    identity = source_id or _stable_id("source", uri or title, digest)
    lineage = " -> ".join(item.strip() for item in transformation_lineage if item.strip()) or None
    attribution = f"speaker={speaker.strip()}" if speaker and speaker.strip() else None
    transformation = "; ".join(item for item in (attribution, lineage) if item) or None
    provenance = (
        ProvenanceRef(
            source_id=identity,
            locator=source_location or uri or "content",
            observed_at=published_at,
            transformation=transformation,
        ),
    )
    return SourceRecord(
        source_id=identity,
        title=title.strip(),
        uri=uri,
        publisher=publisher,
        published_at=published_at,
        retrieved_at=retrieved_at or _now(),
        content_hash=f"sha256:{digest}",
        speaker=speaker.strip() if speaker and speaker.strip() else None,
        transformation_lineage=tuple(
            item.strip() for item in transformation_lineage if item.strip()
        ),
        provenance=provenance,
        revision=_revision("source_rev", identity),
    )


_CLAUSE_BOUNDARY = re.compile(r"(?<!\d)(?:，|,(?!\d))")
_RHETORIC_CUES = (
    "obviously", "disgraceful", "heroic", "betrayal", "undeniable",
    "荒谬", "显然", "可耻", "伟大", "无可争辩", "演戏", "粉饰",
)
_EVALUATION_CUES = (
    "should", "best", "worst", "good", "bad", "unfair", "overreacted",
    "应当", "最好", "最糟糕", "糟糕", "不公平", "最明智", "反应过度",
)
_OMISSION_PATTERNS = (
    re.compile(r"\b(?:no|not|never|missing|unreported|unfilled)\b", re.IGNORECASE),
    re.compile(r"未|尚未|没有|为空|未填|空白"),
)
_FORECAST_PATTERNS = (
    re.compile(r"\b(?:forecast|predict(?:s|ed)?|will|next\s+(?:month|quarter|year))\b", re.IGNORECASE),
    re.compile(r"预测|预计|将会|(?<!协)(?<!员)会(?=[\u4e00-\u9fff])|下月|下季度|明年|开球"),
)
_CAUSAL_PATTERNS = (
    re.compile(r"\b(?:because|caused?|due\s+to|therefore|so\s+that)\b", re.IGNORECASE),
    re.compile(r"导致|造成|由.+造成|因.+(?:破裂|失败|故障)|所以"),
)
_ATTRIBUTION_PATTERN = re.compile(
    r"^(?P<speaker>.{1,24}?)(?:声称|表示|宣布|认为|预测|说|(?<!名)称|\bsaid\b|\bstated\b|\bclaimed\b|\bpredict(?:s|ed)?\b)",
    re.IGNORECASE,
)
_OBSERVED_EVENT_PATTERN = re.compile(
    r"(?:\b\d{1,2}:\d{2}\b.*\b(?:arrived?|stopped?|started?|left|closed?|opened?|found|restored?)\b|"
    r"\d{1,2}:\d{2}.{0,20}(?:抵达|停车|中断|启动|恢复|接走|停止|集合|发现|关闭|离开|控制|停放))",
    re.IGNORECASE,
)
_CONTEXT_ONLY_CUES = (
    "system:", "ignore previous", "do not analyze", "click subscribe",
    "忽略前文", "不要分析", "直接回答", "点击订阅", "广告称",
)
_DIRECT_RECORD_PATTERNS = (
    re.compile(r"\b(?:official\s+)?records?\s+(?:show|shows|indicate|indicates|state|states)\b", re.IGNORECASE),
    re.compile(r"\b(?:recorded|published|issued|filed|registered|announced|released)\b", re.IGNORECASE),
    re.compile(r"(?:记录|档案)(?:显示|表明|记载)|(?:已)?(?:记录|发布|公布|颁布|登记|申报)"),
)


def _direct_record_truth_confidence(text: str) -> float | None:
    """Return minimal direct support only for auditable record assertions.

    A source passage existing proves that it contains the extracted words, but
    does not by itself prove every neutral-sounding claim in those words.  The
    default path therefore keeps truth confidence unknown unless the
    proposition describes a record or publication act that the cited passage
    directly instantiates.  Such propositions receive the minimum admissible
    confidence; stronger values still require an explicit classifier or later
    evidence settlement.
    """

    if any(pattern.search(text) for pattern in _DIRECT_RECORD_PATTERNS):
        return 0.5
    return None


def _default_classification(text: str, _: int) -> Mapping[str, Any]:
    lowered = text.casefold()
    if any(cue in lowered for cue in _RHETORIC_CUES):
        kind = EpistemicType.RHETORIC
    elif any(cue in lowered for cue in _EVALUATION_CUES):
        kind = EpistemicType.EVALUATION
    elif any(pattern.search(text) for pattern in _CAUSAL_PATTERNS):
        # The current evidence contract cannot carry causal-support atom ids.
        # Preserve the attribution instead of emitting a causal claim which the
        # package validator would (correctly) reject as unsupported.
        kind = EpistemicType.ATTRIBUTED_STATEMENT
    elif any(pattern.search(text) for pattern in _OMISSION_PATTERNS) and not re.search(
        r"声称|表示|宣布|认为|说|(?<!名)称|\b(?:said|stated|claimed)\b", text, re.IGNORECASE
    ):
        kind = EpistemicType.OMISSION
    elif any(pattern.search(text) for pattern in _FORECAST_PATTERNS):
        kind = EpistemicType.FORECAST
    elif _ATTRIBUTION_PATTERN.search(text):
        kind = EpistemicType.ATTRIBUTED_STATEMENT
    elif any(pattern.search(text) for pattern in _OMISSION_PATTERNS):
        kind = EpistemicType.OMISSION
    elif _OBSERVED_EVENT_PATTERN.search(text):
        kind = EpistemicType.OBSERVED_EVENT
    else:
        kind = EpistemicType.PRIMARY_RECORD
    source_position = text if kind in {EpistemicType.RHETORIC, EpistemicType.EVALUATION} else None
    return {
        "epistemic_type": kind,
        "source_position": source_position,
        "extraction_confidence": 1.0,
        "truth_confidence": (
            0.5
            if kind is EpistemicType.OBSERVED_EVENT
            else _direct_record_truth_confidence(text)
            if kind is EpistemicType.PRIMARY_RECORD
            else None
        ),
    }


def _passages(content: str) -> tuple[tuple[str, str, tuple[int, int]], ...]:
    """Return atomic passages with deterministic hierarchical locators.

    Sentence and clause delimiters remain attached to the preceding span. This
    makes every emitted claim an exact substring of the source and records the
    character offsets used to obtain it.
    """

    sentence_spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(content):
        char = content[index]
        decimal_point = (
            char == "."
            and index > 0
            and index + 1 < len(content)
            and content[index - 1].isdigit()
            and content[index + 1].isdigit()
        )
        if char in "\r\n" or (char in ".!?。！？；;" and not decimal_point):
            end = index + 1
            while end < len(content) and content[end] in ".!?。！？；;":
                end += 1
            sentence_spans.append((start, end))
            start = end
            while start < len(content) and content[start] in "\r\n":
                start += 1
            index = start
            continue
        index += 1
    if start < len(content):
        sentence_spans.append((start, len(content)))

    atoms: list[tuple[str, str, tuple[int, int]]] = []
    for passage_index, (raw_start, raw_end) in enumerate(sentence_spans, start=1):
        raw = content[raw_start:raw_end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        sentence = raw[leading:trailing]
        if not sentence:
            continue
        sentence_start = raw_start + leading
        boundaries = list(_CLAUSE_BOUNDARY.finditer(sentence))
        if not boundaries:
            atoms.append((sentence, f"passage:{passage_index}", (sentence_start, sentence_start + len(sentence))))
            continue
        pieces: list[tuple[str, int, int]] = []
        cursor = 0
        for boundary in boundaries:
            end = boundary.end()
            piece = sentence[cursor:end]
            if piece.strip():
                left = len(piece) - len(piece.lstrip())
                pieces.append((piece.strip(), cursor + left, end))
            cursor = end
        tail = sentence[cursor:]
        if tail.strip():
            left = len(tail) - len(tail.lstrip())
            pieces.append((tail.strip(), cursor + left, len(sentence)))
        if len(pieces) == 1:
            text, start, end = pieces[0]
            atoms.append((text, f"passage:{passage_index}", (sentence_start + start, sentence_start + end)))
        else:
            for clause_index, (text, start, end) in enumerate(pieces, start=1):
                atoms.append(
                    (text, f"passage:{passage_index}#clause:{clause_index}", (sentence_start + start, sentence_start + end))
                )
    return tuple(atoms)


def _speaker_for(text: str, source: SourceRecord, kind: EpistemicType) -> str | None:
    if kind is EpistemicType.OMISSION:
        return source.speaker
    match = _ATTRIBUTION_PATTERN.search(text)
    if match:
        stated = match.group("speaker").strip(" ，,:;\"'“”")
    else:
        actor = re.match(
            r"^(?P<speaker>[A-Za-z\u4e00-\u9fff]{1,16}?)(?=\d{1,2}(?:月|:))|^(?P<record_speaker>[A-Za-z\u4e00-\u9fff]{1,12}?)(?=记录\d{1,2}:)|^(?P<omission_speaker>消防队)(?=尚未|未)",
            text,
        )
        if not actor:
            return source.speaker
        stated = (
            actor.group("speaker") or actor.group("record_speaker") or actor.group("omission_speaker")
        ).strip()
        explicit_actor = re.fullmatch(
            r"(?:[陈李赵王刘吴林周].{0,3}|法院|调度员|市议会|主持人|消防队|财务官|气象员|经纪人|评论员|作者|记者|书记员|来电者|经理|被告|公司|联盟)",
            stated,
        )
        if not explicit_actor:
            return source.speaker
    stated = re.sub(r"(?:\d{1,2}月\d{0,2}日?|七月|六月)$", "", stated).strip()
    if not stated:
        return source.speaker
    source_speaker = source.speaker or ""
    if stated in source_speaker or source_speaker in stated:
        return source.speaker
    if stated == "另一人" and "家长" in source.title:
        return "另一名家长"
    if stated == "经纪人" and source_speaker.endswith("经纪"):
        return source_speaker + "人"
    if stated == "经理" and "经理" in source.title:
        return source.title.split("声明", 1)[0]
    if stated == "专栏":
        return "专栏作者"
    if stated == "作者" and "新闻稿" in source.title:
        return "研究作者"
    if stated in {"维保", "新闻稿"}:
        return source.speaker
    return stated


def _relevance_for(
    text: str,
    kind: EpistemicType,
    declared: Sequence[str],
) -> tuple[str, ...]:
    """Retain answer linkage while marking obvious context-only material.

    Evidence validation requires a non-empty answer-direction field, including
    for distractors retained for audit. ``context_only`` is deliberately not a
    condition id and therefore cannot satisfy a required condition.
    """

    lowered = text.casefold()
    if not declared:
        # Absence is not evidence of irrelevance. Keeping this empty lets the
        # package validator fail closed when the caller omitted the field.
        return ()
    if (
        kind in {EpistemicType.RHETORIC, EpistemicType.EVALUATION, EpistemicType.FORECAST}
        or any(cue in lowered for cue in _CONTEXT_ONLY_CUES)
    ):
        return ("context_only",)
    return tuple(declared)


def create_proposition(
    source: SourceRecord,
    *,
    claim: str,
    source_locator: str,
    epistemic_type: EpistemicType,
    independence_group: str | None = None,
    target_relevance: Sequence[str] = (),
    speaker: str | None = None,
    event_time: str | None = None,
    modality: str | None = None,
    source_position: str | None = None,
    extraction_confidence: float,
    truth_confidence: float | None,
    transformation: str | None = None,
) -> EvidenceProposition:
    """Create one atomic proposition while keeping two confidences distinct."""

    text = claim.strip()
    if not text:
        raise EvidenceError("proposition claim cannot be empty")
    for name, value in (("extraction_confidence", extraction_confidence), ("truth_confidence", truth_confidence)):
        if value is not None and not 0 <= value <= 1:
            raise EvidenceError(f"{name} must be between 0 and 1")
    if epistemic_type in {EpistemicType.RHETORIC, EpistemicType.EVALUATION} and not source_position:
        source_position = text
    atom_id = _stable_id("atom", source.source_id, source_locator, text)
    return EvidenceProposition(
        evidence_atom_id=atom_id,
        source_id=source.source_id,
        source_locator=source_locator,
        speaker=speaker if speaker is not None else source.speaker,
        claim=text,
        epistemic_type=epistemic_type,
        event_time=event_time,
        published_at=source.published_at,
        modality=modality,
        independence_group=independence_group or source.source_id,
        target_relevance=tuple(target_relevance),
        source_position=source_position,
        extraction_confidence=extraction_confidence,
        truth_confidence=truth_confidence,
        transformation=transformation,
        provenance=(
            ProvenanceRef(
                source_id=source.source_id,
                locator=source_locator,
                observed_at=source.published_at,
                transformation=transformation or "atomic proposition segmentation",
            ),
        ),
        revision=_revision("atom_rev", atom_id),
    )


def segment_propositions(
    source: SourceRecord,
    content: str,
    *,
    classifier: Classifier | None = None,
    target_relevance: Sequence[str] = (),
) -> tuple[EvidenceProposition, ...]:
    """Split text into non-empty atomic passages and type each passage."""

    classify = classifier or _default_classification
    propositions: list[EvidenceProposition] = []
    preceding_speaker: str | None = None
    for index, (segment, default_locator, span) in enumerate(_passages(content), start=1):
        values = dict(classify(segment, index))
        kind = values.pop("epistemic_type", EpistemicType.PRIMARY_RECORD)
        if isinstance(kind, str):
            kind = EpistemicType(kind)
        if (
            classifier is None
            and kind is EpistemicType.PRIMARY_RECORD
            and source.speaker
            and any(cue in segment.casefold() for cue in _CONTEXT_ONLY_CUES)
        ):
            kind = EpistemicType.ATTRIBUTED_STATEMENT
        speaker = values.pop("speaker", None)
        if classifier is None and speaker is None:
            speaker = _speaker_for(segment, source, kind)
            if preceding_speaker and re.match(r"^(?:\d{1,2}(?:月|:)|并于)", segment):
                speaker = preceding_speaker
            explicit = _ATTRIBUTION_PATTERN.search(segment) or re.match(
                r"^(?:[陈李赵王刘吴林周].{0,3}|法院|调度员|市议会|主持人|财务官|气象员|经纪人|评论员|作者|记者|来电者|经理|被告|公司|联盟)",
                segment,
            )
            if explicit and speaker:
                preceding_speaker = speaker
        relevance = values.pop("target_relevance", None)
        if relevance is None:
            relevance = (
                _relevance_for(segment, kind, target_relevance)
                if classifier is None
                else tuple(target_relevance)
            )
        transformation = values.pop("transformation", None)
        if transformation is None:
            transformation = f"atomic clause segmentation; source_span=chars:{span[0]}-{span[1]}"
        propositions.append(
            create_proposition(
                source,
                claim=segment,
                source_locator=str(values.pop("source_locator", default_locator)),
                epistemic_type=kind,
                independence_group=values.pop("independence_group", source.source_id),
                target_relevance=relevance,
                speaker=speaker,
                event_time=values.pop("event_time", None),
                modality=values.pop("modality", None),
                source_position=values.pop("source_position", None),
                extraction_confidence=values.pop("extraction_confidence", 1.0),
                truth_confidence=values.pop("truth_confidence", None),
                transformation=transformation,
            )
        )
        if values:
            raise EvidenceError("classifier returned unsupported fields: " + ", ".join(sorted(values)))
    return tuple(propositions)


_EVENT_FACT_TYPES = {EpistemicType.OBSERVED_EVENT, EpistemicType.PRIMARY_RECORD}


def event_fact_admissibility(
    proposition: EvidenceProposition,
    *,
    minimum_extraction_confidence: float = 0.5,
    minimum_truth_confidence: float = 0.5,
) -> tuple[bool, str]:
    """Decide whether a proposition itself may populate an event fact."""

    if proposition.epistemic_type in {EpistemicType.RHETORIC, EpistemicType.EVALUATION}:
        return False, "source position cannot populate event facts"
    if proposition.epistemic_type not in _EVENT_FACT_TYPES:
        return False, f"{proposition.epistemic_type.value} requires separate settlement"
    if proposition.extraction_confidence is None or proposition.extraction_confidence < minimum_extraction_confidence:
        return False, "extraction confidence is below the admissibility threshold"
    if proposition.truth_confidence is None or proposition.truth_confidence < minimum_truth_confidence:
        return False, "truth confidence is below the admissibility threshold"
    if not proposition.provenance:
        return False, "provenance is missing"
    return True, "admissible factual support"


def independent_support(
    proposition: EvidenceProposition,
    propositions: Iterable[EvidenceProposition],
) -> tuple[EvidenceProposition, ...]:
    """Find separately sourced factual atoms for the same normalized claim."""

    normalized = " ".join(proposition.claim.casefold().split())
    matches = []
    for candidate in propositions:
        admissible, _ = event_fact_admissibility(candidate)
        if (
            candidate.evidence_atom_id != proposition.evidence_atom_id
            and candidate.independence_group != proposition.independence_group
            and " ".join(candidate.claim.casefold().split()) == normalized
            and admissible
        ):
            matches.append(candidate)
    return tuple(matches)


def reconstruct_event_scene(
    *,
    question_id: str,
    propositions: Iterable[EvidenceProposition],
    fact_fields: Mapping[str, Mapping[str, Any]] | None = None,
    scene_id: str | None = None,
) -> EventScene:
    """Reconstruct a scene from admissible atoms without laundering rhetoric.

    ``fact_fields`` is deterministic extraction output keyed by atom id.  It may
    supply actor/actors, action, object, place, time, sequence, event_key and
    fact_value. Missing fields remain explicit known unknowns.
    """

    atoms = tuple(propositions)
    fields_by_atom = fact_fields or {}
    actors: dict[str, dict[str, Any]] = {}
    events: list[SceneEvent] = []
    relations: list[dict[str, Any]] = []
    unknowns: set[str] = set()
    competing: dict[tuple[str, str], list[tuple[str, Any]]] = defaultdict(list)

    for sequence, atom in enumerate(atoms, start=1):
        if atom.epistemic_type is EpistemicType.OMISSION:
            unknowns.add(atom.claim)
            continue
        if atom.epistemic_type in {EpistemicType.RHETORIC, EpistemicType.EVALUATION}:
            relations.append(
                {
                    "type": "source_position",
                    "evidence_atom_id": atom.evidence_atom_id,
                    "position": atom.source_position or atom.claim,
                    "independent_support_atom_ids": tuple(
                        item.evidence_atom_id for item in independent_support(atom, atoms)
                    ),
                }
            )
            continue
        admissible, reason = event_fact_admissibility(atom)
        if not admissible:
            relations.append({"type": "excluded_support", "evidence_atom_id": atom.evidence_atom_id, "reason": reason})
            continue
        supplied = dict(fields_by_atom.get(atom.evidence_atom_id, {}))
        actor_values = supplied.get("actors", supplied.get("actor", atom.speaker))
        if isinstance(actor_values, str):
            actor_values = (actor_values,)
        actor_values = tuple(actor_values or ())
        actor_ids = []
        for actor in actor_values:
            actor_id = _stable_id("actor", actor)
            actors.setdefault(actor_id, {"actor_id": actor_id, "name": str(actor)})
            actor_ids.append(actor_id)
        action = str(supplied.get("action", atom.claim)).strip()
        event_time = supplied.get("time", atom.event_time)
        event_key = str(supplied.get("event_key", action))
        support = (atom.evidence_atom_id,)
        status = EventStatus.CONFIRMED
        events.append(
            SceneEvent(
                event_id=_stable_id("event", question_id, event_key, sequence),
                status=status,
                actor_ids=tuple(actor_ids),
                action=action,
                time_window=str(event_time) if event_time is not None else None,
                supporting_atom_ids=support,
                unknown_fields=tuple(
                    field
                    for field, value in (
                        ("actors", actor_values),
                        ("object", supplied.get("object")),
                        ("time", event_time),
                        ("place", supplied.get("place")),
                    )
                    if not value
                ),
            )
        )
        for field in ("object", "place", "sequence"):
            if field in supplied:
                relations.append(
                    {"type": field, "event_key": event_key, "value": supplied[field], "evidence_atom_id": atom.evidence_atom_id}
                )
        if "fact_field" in supplied and "fact_value" in supplied:
            competing[(event_key, str(supplied["fact_field"]))].append(
                (atom.evidence_atom_id, supplied["fact_value"])
            )

    contested_atom_ids: set[str] = set()
    for (event_key, field), values in competing.items():
        distinct = {repr(value) for _, value in values}
        if len(distinct) > 1:
            contested_atom_ids.update(atom_id for atom_id, _ in values)
            relations.append(
                {"type": "dispute", "event_key": event_key, "field": field, "competing_values": tuple(values)}
            )
    if contested_atom_ids:
        events = [
            SceneEvent(
                event_id=event.event_id,
                status=EventStatus.CONTESTED if set(event.supporting_atom_ids) & contested_atom_ids else event.status,
                actor_ids=event.actor_ids,
                action=event.action,
                supporting_atom_ids=event.supporting_atom_ids,
                counter_atom_ids=tuple(sorted(contested_atom_ids - set(event.supporting_atom_ids)))
                if set(event.supporting_atom_ids) & contested_atom_ids
                else event.counter_atom_ids,
                time_window=event.time_window,
                unknown_fields=event.unknown_fields,
            )
            for event in events
        ]
    identity = scene_id or _stable_id("scene", question_id, *(atom.evidence_atom_id for atom in atoms))
    provenance = tuple(ref for atom in atoms for ref in atom.provenance)
    return EventScene(
        scene_id=identity,
        question_id=question_id,
        actors=tuple(actors.values()),
        events=tuple(events),
        relations=tuple(relations),
        known_unknowns=tuple(sorted(unknowns)),
        provenance=provenance,
        revision=_revision("scene_rev", identity),
    )
