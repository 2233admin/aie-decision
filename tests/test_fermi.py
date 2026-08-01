import pytest

from aie_decision.fermi import estimate_fermi


def test_fermi_propagates_interval_and_prunes_unused_variables():
    result = estimate_fermi(
        {
            "question": "How many sales?",
            "target": "sales",
            "unit": "orders/day",
            "formula": "visitors * conversion_rate",
            "coverage": 0.9,
            "reference_value": 100,
            "acceptable_width": 100,
            "variables": [
                {"name": "visitors", "lower": 800, "upper": 1200},
                {"name": "conversion_rate", "lower": 0.08, "upper": 0.12},
                {"name": "commentary", "lower": 0, "upper": 1},
            ],
        }
    )

    assert [item["name"] for item in result["minimal_variables"]] == ["visitors", "conversion_rate"]
    assert result["excluded_variables"] == ["commentary"]
    assert result["target_interval"] == {"lower": 64.0, "upper": 144.0}
    assert result["absolute_width"] == 80.0
    assert result["normalized_width"] == 0.8
    assert result["status"] == "uncalibrated_informative"
    assert result["calibration"] == "unmeasured"


def test_fermi_ranks_the_variable_that_most_narrows_the_interval():
    result = estimate_fermi(
        {
            "question": "Tomorrow revenue?",
            "target": "daily_revenue",
            "unit": "CNY/day",
            "formula": "visitors * conversion_rate * average_order_value",
            "reference_value": 10000,
            "acceptable_width": 5000,
            "thresholds": [10000],
            "variables": [
                {"name": "visitors", "lower": 600, "upper": 1400},
                {"name": "conversion_rate", "lower": 0.09, "upper": 0.11},
                {"name": "average_order_value", "lower": 95, "upper": 105},
            ],
        }
    )

    assert result["target_interval"] == {"lower": 5130.0, "upper": 16170.0}
    assert result["absolute_width"] == 11040.0
    assert result["within_acceptable_width"] is False
    assert result["decision_robust"] is False
    assert result["largest_uncertainty_source"] == "visitors"
    assert result["next_measurement"] == "visitors"


def test_fermi_rejects_unsafe_formula_and_zero_crossing_denominator():
    base = {
        "question": "ratio?",
        "target": "ratio",
        "unit": "ratio",
        "variables": [
            {"name": "numerator", "lower": 1, "upper": 2},
            {"name": "denominator", "lower": -1, "upper": 1},
        ],
    }
    with pytest.raises(ValueError, match="denominator interval cannot contain zero"):
        estimate_fermi({**base, "formula": "numerator / denominator"})
    with pytest.raises(ValueError, match="supports only"):
        estimate_fermi({**base, "formula": "__import__('os').system('whoami')"})
