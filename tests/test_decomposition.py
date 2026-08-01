from dataclasses import replace

import pytest

from aie_decision.decomposition import (
    AnswerContractError,
    ConditionGraphError,
    build_condition_graph,
    create_answer_contract,
    evaluate_answerability,
    invalidate_graph_for_contract_change,
    minimal_sufficient_sets,
    revise_answer_contract,
)
from aie_decision.models import (
    AnswerType,
    AnswerabilityState,
    ConditionEdge,
    ConditionNode,
    ConditionStatus,
    CoverageSemantics,
    Necessity,
)


def contract(**overrides):
    values = {
        "question_id": "q-price",
        "question": "What will the Shanghai copper price be tomorrow?",
        "answer_type": AnswerType.FUTURE_PREDICTION,
        "subject": "Shanghai copper",
        "target": "closing price",
        "unit": "CNY/tonne",
        "observation_cutoff": "2026-08-01T08:00:00Z",
        "prediction_horizon": "2026-08-02T15:00:00+08:00",
        "geography": "Shanghai, CN",
        "decision_use": "decide whether to hedge above 80,000",
        "decision_thresholds": (75_000.0, 80_000.0),
        "uncertainty_semantics": CoverageSemantics.EMPIRICAL_PREDICTION_INTERVAL,
        "requested_coverage": 0.9,
        "acceptable_width": {"absolute": 5_000, "unit": "CNY/tonne"},
    }
    values.update(overrides)
    return create_answer_contract(**values)


def node(
    condition_id: str,
    necessity: Necessity,
    status: ConditionStatus,
    answer_impact: str = "changes the predicted interval",
):
    return ConditionNode(
        condition_id=condition_id,
        name=condition_id.replace("_", " "),
        value_type="number",
        necessity=necessity,
        status=status,
        answer_impact=answer_impact,
    )


def test_answer_contract_preserves_every_decision_field():
    value = contract()

    assert value.target.entity == "Shanghai copper"
    assert value.target.measure == "closing price"
    assert value.target.unit == "CNY/tonne"
    assert value.prediction_horizon.endswith("+08:00")
    assert value.geography == "Shanghai, CN"
    assert value.decision_thresholds == (75_000.0, 80_000.0)
    assert value.uncertainty_semantics is CoverageSemantics.EMPIRICAL_PREDICTION_INTERVAL
    assert value.revision.sequence == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prediction_horizon": None}, "prediction_horizon"),
        ({"uncertainty_semantics": None}, "uncertainty_semantics"),
        ({"requested_coverage": 1.0}, "requested_coverage"),
        ({"decision_thresholds": (80_000, 75_000)}, "increasing"),
    ],
)
def test_ambiguous_or_invalid_contract_is_rejected(overrides, message):
    with pytest.raises(AnswerContractError, match=message):
        contract(**overrides)


def test_graph_supports_all_four_condition_classifications():
    value = contract()
    conditions = (
        node("spot", Necessity.REQUIRED, ConditionStatus.OBSERVED),
        node("inventory", Necessity.OPTIONAL, ConditionStatus.MISSING),
        node("supply_shock", Necessity.COMPETING, ConditionStatus.ESTIMATED),
        node("social_mood", Necessity.EXPLORATORY, ConditionStatus.MISSING),
    )
    graph = build_condition_graph(value, conditions, minimal_sufficient_sets=(("spot",),))

    assert {item.necessity for item in graph.conditions} == set(Necessity)
    assert graph.answer_contract_revision_id == value.revision.revision_id


def test_node_without_stated_answer_path_is_rejected():
    with pytest.raises(ConditionGraphError, match="no_path"):
        build_condition_graph(
            contract(),
            (node("no_path", Necessity.REQUIRED, ConditionStatus.OBSERVED, ""),),
        )


def test_edge_with_unknown_node_is_rejected():
    condition = node("spot", Necessity.REQUIRED, ConditionStatus.OBSERVED)
    edge = ConditionEdge("e1", "spot", "missing", "affects", "supported", "narrows price")

    with pytest.raises(ConditionGraphError, match="unknown condition"):
        build_condition_graph(contract(), (condition,), (edge,))


def test_explicit_minimal_sets_drop_supersets_and_optional_missing_does_not_block():
    value = contract()
    conditions = (
        node("spot", Necessity.REQUIRED, ConditionStatus.OBSERVED),
        node("inventory", Necessity.OPTIONAL, ConditionStatus.MISSING),
    )
    graph = build_condition_graph(
        value,
        conditions,
        minimal_sufficient_sets=(("spot",), ("spot", "inventory")),
    )

    assert minimal_sufficient_sets(graph) == (("spot",),)
    result = evaluate_answerability(value, graph)
    assert result.status is AnswerabilityState.ANSWERABLE_BOUNDED
    assert not result.blocking_condition_ids


def test_unresolved_required_condition_is_explicitly_blocking():
    value = contract()
    graph = build_condition_graph(
        value,
        (node("spot", Necessity.REQUIRED, ConditionStatus.MISSING),),
    )

    result = evaluate_answerability(value, graph)
    assert result.status is AnswerabilityState.INSUFFICIENT_EVIDENCE
    assert result.blocking_condition_ids == ("spot",)


def test_contract_revision_mismatch_is_not_answerable():
    original = contract()
    graph = build_condition_graph(
        original,
        (node("spot", Necessity.REQUIRED, ConditionStatus.OBSERVED),),
    )
    revised = revise_answer_contract(original, prediction_horizon="2026-08-08T15:00:00+08:00")

    assert evaluate_answerability(revised, graph).status is AnswerabilityState.INVALID_CONTRACT


def test_material_contract_change_invalidates_all_descendants_with_lineage():
    original = contract()
    graph = build_condition_graph(
        original,
        (node("spot", Necessity.REQUIRED, ConditionStatus.OBSERVED),),
    )
    revised = revise_answer_contract(original, target=replace(original.target, unit="USD/tonne"))

    invalidated = invalidate_graph_for_contract_change(graph, original, revised)

    assert invalidated.answer_contract_revision_id == revised.revision.revision_id
    assert invalidated.conditions[0].status is ConditionStatus.INVALIDATED
    assert invalidated.revision.supersedes_revision_id == graph.revision.revision_id
