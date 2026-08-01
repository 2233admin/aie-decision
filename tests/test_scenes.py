from __future__ import annotations

from dataclasses import replace

from aie_decision.evidence import create_proposition, ingest_source
from aie_decision.models import EpistemicType, EventStatus
from aie_decision.scenes import infer_scene_fact_fields, reconstruct_event_scene


def _atom(
    claim: str,
    *,
    kind: EpistemicType = EpistemicType.PRIMARY_RECORD,
    speaker: str = "消防队",
):
    source = ingest_source(
        claim,
        title="公开记录",
        source_id=f"source-{abs(hash((claim, speaker)))}",
        speaker=speaker,
        retrieved_at="2026-08-01T00:00:00Z",
    )
    return create_proposition(
        source,
        claim=claim,
        source_locator="passage:1",
        epistemic_type=kind,
        target_relevance=("incident",),
        extraction_confidence=1.0,
        truth_confidence=0.8 if kind is EpistemicType.PRIMARY_RECORD else None,
    )


def test_literal_projection_extracts_actor_time_place_and_sequence():
    first = _atom("消防队22:21抵达3号仓库。")
    second = _atom("23:05控制明火。")

    scene = reconstruct_event_scene(question_id="q-fire", propositions=(first, second))

    assert [actor["name"] for actor in scene.actors] == ["消防队"]
    assert [event.time_window for event in scene.events] == ["22:21", "23:05"]
    assert [relation["value"] for relation in scene.relations if relation["type"] == "place"] == ["3号仓库"]
    assert [relation["value"] for relation in scene.relations if relation["type"] == "sequence"] == [1, 2]


def test_explicit_fields_override_inference_without_erasing_other_fields():
    atom = _atom("消防队22:21抵达3号仓库。")
    scene = reconstruct_event_scene(
        question_id="q-fire",
        propositions=(atom,),
        fact_fields={atom.evidence_atom_id: {"action": "抵达", "place": "东港3号仓库"}},
    )

    assert scene.events[0].action == "抵达"
    assert scene.events[0].time_window == "22:21"
    assert next(item for item in scene.relations if item["type"] == "place")["value"] == "东港3号仓库"


def test_positions_never_receive_inferred_event_fields():
    rhetoric = _atom("作者称这显然是最好的政策。", kind=EpistemicType.RHETORIC, speaker="作者")

    assert infer_scene_fact_fields((rhetoric,)) == {}
    scene = reconstruct_event_scene(question_id="q-policy", propositions=(rhetoric,))
    assert scene.events == ()
    assert [relation["type"] for relation in scene.relations] == ["source_position"]


def test_unsettled_primary_record_projects_as_claimed_only_not_confirmed():
    atom = _atom("消防队22:21抵达3号仓库。")
    atom = replace(atom, truth_confidence=None)

    scene = reconstruct_event_scene(question_id="q-fire", propositions=(atom,))

    assert scene.events[0].status is EventStatus.CLAIMED_ONLY
    assert scene.events[0].time_window == "22:21"


def test_places_are_projected_from_location_subjects_and_evacuation_targets():
    restore = _atom("重症室14:06恢复供电。", speaker="医院值班室")
    evacuate = _atom("指挥部05:42命令东片疏散。", speaker="县应急部")

    scene = reconstruct_event_scene(question_id="q-response", propositions=(restore, evacuate))

    assert [
        relation["value"] for relation in scene.relations if relation["type"] == "place"
    ] == ["重症室", "东片"]


def test_coarsely_typed_attribution_and_omission_fail_closed_in_scene():
    attributed = replace(_atom("经理称警报器未鸣响。"), truth_confidence=None)
    omitted = replace(_atom("消防队尚未确认起火原因。"), truth_confidence=None)

    scene = reconstruct_event_scene(question_id="q-fire", propositions=(attributed, omitted))

    assert scene.events == ()
    assert scene.known_unknowns == ("消防队尚未确认起火原因。",)
    assert any(relation["type"] == "excluded_support" for relation in scene.relations)


def test_competing_source_statuses_are_a_dispute_not_confirmed_events():
    union = replace(_atom("工会称谈判破裂。", speaker="工会"), truth_confidence=None)
    company = replace(_atom("公司称谈判仍继续。", speaker="公司"), truth_confidence=None)

    scene = reconstruct_event_scene(question_id="q-strike", propositions=(union, company))

    assert scene.events == ()
    dispute = next(relation for relation in scene.relations if relation["type"] == "dispute")
    assert dispute["event_key"] == "谈判"
    assert dispute["field"] == "status"
    assert {value for _, value in dispute["competing_values"]} == {"破裂", "继续"}
    assert dispute["settlement"] == "unresolved_source_positions"


def test_authorship_is_not_used_as_actor_when_clause_has_no_actor():
    atom = _atom("23:05控制明火。", speaker="仓储公司")
    scene = reconstruct_event_scene(question_id="q-fire", propositions=(atom,))

    assert scene.actors == ()
    assert scene.events[0].actor_ids == ()


def test_temporal_prefix_is_not_part_of_platform_place():
    atom = _atom("7月5日申领平台开放。", speaker="专栏作者")
    scene = reconstruct_event_scene(question_id="q-policy", propositions=(atom,))

    assert next(item for item in scene.relations if item["type"] == "place")["value"] == "申领平台"


def test_explicit_date_dispute_preserves_competing_reported_dates():
    first = replace(_atom("刘青说3月4日交付样品。", speaker="刘青"), truth_confidence=None)
    second = replace(_atom("被告说3月6日收到。", speaker="被告"), truth_confidence=None)
    disputed = replace(_atom("法官说日期有争议。", speaker="法官"), truth_confidence=None)

    scene = reconstruct_event_scene(question_id="q-court", propositions=(first, second, disputed))

    dispute = next(item for item in scene.relations if item["type"] == "dispute")
    assert dispute["event_key"] == "日期"
    assert {value for _, value in dispute["competing_values"]} == {"3月4日", "3月6日"}
