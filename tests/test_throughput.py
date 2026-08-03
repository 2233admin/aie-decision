import json
from pathlib import Path

import pytest

from aie_decision.cli import main
from aie_decision.throughput import estimate_throughput


FIXTURE = Path("fixtures/throughput/v1/cafe-day.json")


def test_question_and_materials_generate_auditable_probability_interval():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "formula" not in payload and "variables" not in payload and "bounds" not in payload

    result = estimate_throughput(payload)

    assert result["status"] == "complete"
    assert len(result["decomposition_candidates"]) == 4
    assert result["selected_decomposition"]["candidate_id"] == "served_throughput"
    assert result["interval_method"] == "joint_monte_carlo_quantiles"
    assert result["coverage"] == 0.9
    assert result["target_90_interval"]["p05"] < result["target_90_interval"]["p95"]
    assert result["absolute_width"] > 0
    assert result["joint_model"]["status"] == "explicit_untested_assumption"
    assert set(result["dependence_stress_cases"]) == {"perfect_positive_rank", "rank_reversed_stress"}
    assert result["calibration"] == "unmeasured"
    assert result["next_measurement"]["status"] == "analytic_probe_not_observation"


def test_minimality_is_verified_by_deleting_every_retained_leaf():
    result = estimate_throughput(json.loads(FIXTURE.read_text(encoding="utf-8")))

    tests = result["minimality_tests"]
    assert result["minimal_variable_set"] == [row["variable"] for row in tests]
    assert all(row["intervention"] == "delete_leaf" for row in tests)
    assert all(row["ablation_executed"] for row in tests)
    assert all(row["deletion_error"].startswith("missing required leaf") for row in tests)
    assert all(row["answerability_after_deletion"] == "not_answerable" for row in tests)
    assert all(row["retained"] for row in tests)
    assert result["minimality_basis"] != "variables_referenced_by_declared_formula"


def test_insufficient_material_exposes_gaps_instead_of_inventing_bounds():
    result = estimate_throughput(
        {
            "question": "How many parcels can the depot process per day?",
            "materials": [{"id": "note", "text": "The depot has 3 service stations."}],
        }
    )

    assert result["status"] == "partial"
    assert result["answerability"] == "not_answerable"
    assert result["target_90_interval"] is None
    assert "rate_per_unit_hour" in result["gaps"]
    assert not any(leaf["evidence_basis"].startswith("assumed") for leaf in result["extracted_leaves"])


def test_plain_range_without_probability_semantics_is_not_used_as_p90():
    result = estimate_throughput(
        {
            "question": "How many parcels can the depot process per day?",
            "materials": [{
                "id": "note",
                "text": "Daily demand is between 100 and 200 parcels. The depot has 3 service stations.",
            }],
        }
    )

    assert result["status"] == "partial"
    assert "daily_demand" in result["gaps"]
    assert result["unqualified_ranges_ignored"] == ["note:sentence:1"]


def test_cli_estimate_executes_public_path(capsys):
    assert main(["estimate", str(FIXTURE)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["input_contract"] == "question_plus_attributed_materials"
    assert output["selected_decomposition"]["generated_expression"].startswith("min(")


def test_duplicate_material_identity_is_rejected():
    with pytest.raises(ValueError, match="duplicate material id"):
        estimate_throughput(
            {
                "question": "How many orders can the depot process per day?",
                "materials": [
                    {"id": "same", "text": "The depot has 3 service stations."},
                    {"id": "same", "text": "The depot has 4 service stations."},
                ],
            }
        )
