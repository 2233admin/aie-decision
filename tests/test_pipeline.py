import json

from aie_decision.cli import main
from aie_decision.models import IntervalAuditStatus
from aie_decision.pipeline import compile_analysis


def _payload():
    return {
        "answer_contract": {
            "question_id": "q-price",
            "question": "What will product A cost tomorrow?",
            "answer_type": "future_prediction",
            "target": {"entity": "product-a", "measure": "market_price", "unit": "CNY"},
            "observation_cutoff": "2026-08-01T00:00:00Z",
            "prediction_horizon": "2026-08-02T00:00:00Z",
            "decision_use": "choose whether to buy now",
            "decision_thresholds": [490, 510],
            "uncertainty_semantics": "scenario_bound",
            "requested_coverage": 0.9,
        },
        "conditions": [
            {
                "condition_id": "price-target",
                "name": "tomorrow price",
                "value_type": "number",
                "necessity": "required",
                "status": "estimated",
                "answer_impact": "is the target answer",
                "unit": "CNY",
            },
            {
                "condition_id": "supply",
                "name": "available supply",
                "value_type": "number",
                "necessity": "required",
                "status": "observed",
                "answer_impact": "lower supply can increase the target price",
                "unit": "items",
            },
        ],
        "edges": [
            {
                "from_id": "supply",
                "to_id": "price-target",
                "relation": "constrains",
                "answer_impact": "changes the feasible price range",
            }
        ],
        "minimal_sufficient_sets": [["supply"]],
        "sources": [
            {
                "title": "Inventory record",
                "content": "The warehouse recorded 200 available units.",
                "source_location": "line:1",
                "target_relevance": ["supply"],
            }
        ],
        "forecast_interval": {
            "target": "market_price",
            "horizon": "2026-08-02T00:00:00Z",
            "unit": "CNY",
            "population": "product-a",
            "coverage_level": 0.9,
            "generation_method": "declared scenarios",
            "reference_time": "2026-08-01T00:00:00Z",
            "lower": 0,
            "upper": 1000,
            "reference_value": 500,
            "baseline": {"lower": 480, "upper": 520},
        },
    }


def test_end_to_end_compiler_produces_traceable_nonempty_package():
    result = compile_analysis(_payload())
    package = result.package
    assert package.package_state == "complete"
    assert package.condition_graph.conditions
    assert package.evidence_propositions
    assert package.event_scene is not None
    assert package.interval_audit.normalized_width == 2.0
    assert package.interval_audit.status is IntervalAuditStatus.UNCALIBRATED_UNINFORMATIVE
    latest = result.ledger.latest("analysis_package", package.package_id)
    assert latest is not None
    assert latest.stable_id == package.package_id
    assert result.validation_issues == ()


def test_compiler_emits_explicit_partial_package_when_optional_stages_absent():
    payload = _payload()
    payload.pop("sources")
    payload.pop("forecast_interval")
    result = compile_analysis(payload)
    assert result.package.package_state == "partial"
    assert {"sources", "evidence_propositions", "event_scene", "interval_audit"}.issubset(
        result.package.empty_section_reasons
    )


def test_same_source_multiple_propositions_keep_stable_ids_and_bind_scene_fields():
    payload = _payload()
    payload["sources"] = [{
        "source_id": "shared-source",
        "title": "Depot record",
        "content": "The depot recorded opening. The depot recorded closing.",
        "source_location": "record:1",
        "speaker": "Agency",
        "target_relevance": ["supply"],
    }]
    payload["scene_fact_fields"] = [
        {
            "selector": {"source_id": "shared-source", "source_locator": "passage:1"},
            "fields": {"event_key": "opening", "actor": "Agency", "action": "opened", "time": "09:00", "place": "Depot A", "sequence": 1},
        },
        {
            "selector": {"source_id": "shared-source", "source_locator": "passage:2"},
            "fields": {"event_key": "closing", "actor": "Agency", "action": "closed", "time": "17:00", "place": "Depot A", "sequence": 2},
        },
    ]

    first = compile_analysis(payload)
    second = compile_analysis(payload)
    atoms = first.package.evidence_propositions

    assert len(atoms) == 2
    assert len({item.evidence_atom_id for item in atoms}) == 2
    assert [item.evidence_atom_id for item in atoms] == [item.evidence_atom_id for item in second.package.evidence_propositions]
    assert len(first.ledger.records("evidence_proposition")) == 2
    assert [event.action for event in first.package.event_scene.events] == ["opened", "closed"]
    assert [relation["value"] for relation in first.package.event_scene.relations if relation["type"] == "sequence"] == [1, 2]


def test_compile_cli_writes_machine_report_and_ledger(tmp_path, capsys):
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "output"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert main(["compile", str(input_path), "--output-dir", str(output_dir)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "complete"
    assert (output_dir / "analysis-package.json").is_file()
    assert (output_dir / "decision-report.md").is_file()
    assert (output_dir / "analysis-ledger.json").is_file()
