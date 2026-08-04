from aie_decision.cli import main
from aie_decision.search import search_fermi

json = __import__("json")


def _loop_payload():
    return {
        "run_id": "loop-golden",
        "question": "How many daily buyers?",
        "target": "daily_buyers",
        "unit": "buyers/day",
        "coverage": 0.9,
        "reference_value": 100,
        "acceptable_width": 40,
        "variables": [
            {"name": "rough_total", "lower": 0, "upper": 1000, "method": "assumed"},
            {"name": "population", "lower": 950, "upper": 1050, "method": "observed"},
            {
                "name": "participation_rate",
                "lower": 0.09,
                "upper": 0.11,
                "method": "observed",
            },
            {"name": "adjustment", "lower": 0, "upper": 0, "method": "assumed"},
        ],
        "candidates": [
            {
                "candidate_id": "rough",
                "formula": "rough_total",
                "mutation_kind": "seed",
            },
            {
                "candidate_id": "expanded",
                "parent_candidate_id": "rough",
                "mutation_kind": "expand",
                "formula": "population * participation_rate + adjustment",
            },
            {
                "candidate_id": "minimal",
                "parent_candidate_id": "expanded",
                "mutation_kind": "ablate",
                "formula": "population * participation_rate",
            },
        ],
        "budget": {
            "max_candidates": 10,
            "max_rounds": 5,
            "max_evaluations": 10,
            "max_seconds": 5,
        },
    }


def test_loop_expands_wide_candidate_then_ablates_redundant_variable():
    result = search_fermi(_loop_payload())

    assert result["status"] == "result-found"
    assert result["selected_candidate"]["candidate_id"] == "minimal"
    assert result["selected_candidate"]["variable_names"] == [
        "population",
        "participation_rate",
    ]
    assert result["selected_candidate"]["target_interval"] == {
        "lower": 85.5,
        "upper": 115.5,
    }
    assert result["minimality_basis"] == "declared_ablation_frontier_exhausted"
    assert result["product_acceptance"] == "provisional_uncalibrated"
    assert result["experimental_usable"] is True
    assert result["usable"] is False
    states = [event["state"] for event in result["search_trace"]]
    assert states == [
        "SEED",
        "VALIDATE",
        "EVALUATE",
        "RANK",
        "EXPAND",
        "VALIDATE",
        "EVALUATE",
        "RANK",
        "MINIMIZE",
        "VALIDATE",
        "EVALUATE",
        "RANK",
        "RESULT",
    ]
    assert len(result["ledger"]["entries"]) == len(result["search_trace"])


def test_loop_stops_when_round_budget_is_exhausted():
    payload = _loop_payload()
    payload["budget"]["max_rounds"] = 1

    result = search_fermi(payload)

    assert result["status"] == "budget-exhausted"
    assert result["selected_candidate"] is None
    assert result["budget"]["evaluations_used"] == 1


def test_loop_uses_bayes_inspired_score_to_rank_equal_minimal_candidates():
    payload = _loop_payload()
    payload["candidates"] = [
        {
            "candidate_id": "low-prior",
            "formula": "population * participation_rate",
            "prior_weight": 0.1,
        },
        {
            "candidate_id": "high-prior",
            "formula": "population * participation_rate",
            "prior_weight": 0.9,
        },
    ]

    result = search_fermi(payload)

    assert result["selected_candidate"]["candidate_id"] == "high-prior"
    assert abs(result["selected_candidate"]["pseudo_posterior"] - 0.9) < 1e-12
    assert result["ranking_method"]["calibrated"] is False
    assert "not a statistical posterior" in result["ranking_method"]["warning"]


def test_mvp_generates_expansion_then_automatically_ablates_declared_variable():
    payload = _loop_payload()
    payload["candidates"] = [
        {"candidate_id": "rough", "formula": "rough_total", "mutation_kind": "seed"},
    ]
    payload["mutation_templates"] = [
        {
            "template_id": "replace-rough-estimate",
            "formula_template": "population * participation_rate + adjustment",
            "diagnostic_reasons": ["interval_too_wide"],
            "mutation_kind": "revise",
            "prior_multiplier": 0.8,
        }
    ]
    for variable in payload["variables"]:
        variable["ablatable"] = variable["name"] == "adjustment"

    result = search_fermi(payload)

    assert result["status"] == "result-found"
    assert result["selected_candidate"]["formula"] == "population * participation_rate"
    assert (
        result["selected_candidate"]["generation"]["removed_variable"] == "adjustment"
    )
    assert result["checkpoint"]["replay"]["terminal"]["reason"] == "result-found"
    assert any(
        row["generation"].get("template_id") == "replace-rough-estimate"
        for row in result["evaluations"]
    )


def test_loop_rejects_candidate_cycles():
    payload = _loop_payload()
    payload["candidates"] = [
        {
            "candidate_id": "a",
            "parent_candidate_id": "b",
            "mutation_kind": "expand",
            "formula": "rough_total",
        },
        {
            "candidate_id": "b",
            "parent_candidate_id": "a",
            "mutation_kind": "expand",
            "formula": "rough_total",
        },
    ]
    try:
        search_fermi(payload)
    except ValueError as exc:
        assert "acyclic" in str(exc)
    else:
        raise AssertionError("candidate cycle was accepted")


def test_search_cli_requires_experimental_flag(tmp_path, capsys):
    path = tmp_path / "search.json"
    path.write_text(json.dumps(_loop_payload()), encoding="utf-8")

    assert main(["search-fermi", str(path)]) == 2
    assert "requires --experimental" in json.loads(capsys.readouterr().err)["message"]

    assert main(["search-fermi", str(path), "--experimental"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected_candidate"]["candidate_id"] == "minimal"
