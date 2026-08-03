from __future__ import annotations

import json
from pathlib import Path

from aie_decision.agent_runtime import AgentRuntime
from aie_decision.fermi_kernel import FermiKernel


RAW_QUESTION = "上海地铁乘客在典型工作日支付多少票款？"


def _runtime() -> AgentRuntime:
    return AgentRuntime.start(
        session_id="subway-fares",
        question=RAW_QUESTION,
        kernel=FermiKernel(),
    )


def _apply(runtime: AgentRuntime, action: str, payload: dict) -> None:
    result = runtime.apply(action=action, payload=payload)
    assert result.accepted, result.error


def _define_and_expand(runtime: AgentRuntime) -> None:
    _apply(
        runtime,
        "define_question",
        {
            "target_subject": "上海地铁工作日票款",
            "target_measure": "乘客支付的票款总额",
            "unit": "CNY/day",
            "time_basis": "2026年典型工作日",
            "scope": {
                "population": "上海地铁付费乘客",
                "geography": "上海",
                "time_window": "典型工作日",
            },
            "acceptable_width": 5_000_000,
            "decision_use": "评估票价收入量级",
        },
    )
    _apply(
        runtime,
        "expand",
        {
            "node_id": "n_0001",
            "parent_unit": "CNY/day",
            "expression": "_child_0 * _child_1",
            "rationale": "每日票款等于付费乘客数乘以人均票款",
            "children": [
                {
                    "label": "工作日付费乘客数",
                    "unit": "person/day",
                    "scope": {
                        "population": "上海地铁付费乘客",
                        "geography": "上海",
                        "time_window": "典型工作日",
                    },
                },
                {
                    "label": "每位乘客平均票款",
                    "unit": "CNY/person",
                    "scope": {
                        "population": "上海地铁付费乘客",
                        "geography": "上海",
                        "time_window": "典型工作日",
                    },
                },
            ],
        },
    )


def _atom(
    runtime: AgentRuntime,
    *,
    node_id: str,
    target_object: str,
    unit: str,
    source: str,
    procedure: str,
    marginal: dict,
) -> None:
    _apply(
        runtime,
        "propose_atom",
        {
            "node_id": node_id,
            "target_object": target_object,
            "unit": unit,
            "scope": {
                "population": "上海地铁付费乘客",
                "geography": "上海",
                "time_window": "典型工作日",
            },
            "measurement_kind": "record_lookup",
            "source": source,
            "procedure": procedure,
            "observation_kind": "estimated",
            "marginal": marginal,
        },
    )


def _complete_numeric_branch(runtime: AgentRuntime, *, second_unknown: bool = False) -> None:
    _define_and_expand(runtime)
    _atom(
        runtime,
        node_id="n_0002",
        target_object="在闸机记录中完成付费进站的乘客",
        unit="person/day",
        source="按日汇总的闸机交易记录",
        procedure="汇总指定工作日全部有效付费进站记录",
        marginal={
            "kind": "quantile",
            "p05": 9_800_000,
            "p50": 10_000_000,
            "p95": 10_200_000,
            "family": "normal",
            "rationale": "历史相邻工作日波动",
        },
    )
    _atom(
        runtime,
        node_id="n_0003",
        target_object="每条有效乘车交易实际扣款金额",
        unit="CNY/person",
        source="票务清算交易记录",
        procedure="用当日有效交易总扣款除以付费进站人数",
        marginal=(
            {
                "kind": "unknown",
                "reason": "尚未取得按票种拆分的清算记录",
                "domain": [2.0, 8.0],
            }
            if second_unknown
            else {
                "kind": "quantile",
                "p05": 3.9,
                "p50": 4.0,
                "p95": 4.1,
                "family": "normal",
                "rationale": "历史票种组合波动",
            }
        ),
    )
    _apply(
        runtime,
        "set_dependence",
        {
            "dependence": "independent",
            "sample_count": 4096,
            "seed": 17,
            "rationale": "v1 assumes global independence; no pairwise evidence was supplied",
        },
    )


def test_raw_question_starts_without_formula_variables_or_bounds() -> None:
    runtime = _runtime()
    assert runtime.state["raw_question"] == RAW_QUESTION
    assert runtime.state["question_contract"] is None
    assert runtime.discover()["legal_next_actions"] == ["define_question"]
    assert runtime.kernel.active_frontier(runtime.state) == [
        {
            "node_id": "raw_question",
            "label": RAW_QUESTION,
            "status": "needs_question_definition",
        }
    ]


def test_non_throughput_end_to_end_certifies_executed_frontier() -> None:
    runtime = _runtime()
    _complete_numeric_branch(runtime)
    _apply(runtime, "evaluate", {})
    _apply(
        runtime,
        "test_frontier",
        {"material_degradation": 0.1, "minimum_useful_narrowing": 3_000_000},
    )

    tested = runtime.state["frontier_test"]
    summary = tested["summary"]
    assert summary["probability_interval_valid"] is True
    assert summary["coverage_semantics"] == "monte_carlo_joint_sampling"
    assert summary["correlation"] == 0.0
    assert summary["target_interval_90"]["p05"] < summary["target_interval_90"]["p95"]
    assert summary["interval_width"] > 0
    assert tested["next_measurement"]["leaf_id"] in {"n_0002", "n_0003"}
    assert all(item["deletion"]["new_status"] == "non_computable" for item in tested["necessity"])
    assert all(item["is_necessary"] for item in tested["necessity"])
    assert tested["saturation"]["explored"]
    assert tested["certificate"]["certified"] is True

    final = runtime.finalize()
    assert final["status"] == "certified"
    assert final["evaluation"]["target_interval_90"] == summary["target_interval_90"]
    assert final["evaluation"]["interval_width"] == summary["interval_width"]
    assert final["evaluation"]["next_measurement"] == tested["next_measurement"]
    assert runtime.replay()["verdict"] == "match"


def test_unknown_leaf_exposes_gap_and_cannot_certify() -> None:
    runtime = _runtime()
    _complete_numeric_branch(runtime, second_unknown=True)
    _apply(runtime, "evaluate", {})
    _apply(
        runtime,
        "test_frontier",
        {"material_degradation": 0.1, "minimum_useful_narrowing": 3_000_000},
    )

    tested = runtime.state["frontier_test"]
    summary = tested["summary"]
    assert summary["probability_interval_valid"] is False
    assert summary["target_interval_90"] is None
    assert summary["interval_width"] is None
    assert summary["scenario_bounds"] is not None
    assert summary["unknown_leaves"] == ["n_0003"]
    assert tested["certificate"]["certified"] is False
    assert runtime.kernel.evaluate_frontier(runtime.state)["status"] == "insufficient"


def test_versioned_json_example_executes_without_prefilled_user_formula() -> None:
    example = json.loads(
        Path("examples/fermi-runtime-v1/subway-fares.json").read_text(encoding="utf-8")
    )
    assert set(example["user_input"]) == {"question"}
    runtime = AgentRuntime.start(
        session_id="versioned-example",
        question=example["user_input"]["question"],
        kernel=FermiKernel(),
    )
    for record in example["ai_semantic_actions"]:
        _apply(runtime, record["action"], record["payload"])
    assert runtime.state["frontier_test"]["certificate"]["certified"] is True
