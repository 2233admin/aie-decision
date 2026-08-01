import json

from aie_decision.cli import main


def test_interval_cli_reports_vacuous_width(capsys):
    code = main(
        [
            "audit-interval",
            "--target", "price",
            "--horizon", "tomorrow",
            "--unit", "CNY",
            "--population", "product-a",
            "--coverage", "0.9",
            "--lower", "0",
            "--upper", "1000",
            "--reference", "500",
            "--reference-time", "2026-08-01T00:00:00Z",
            "--method", "scenario bounds",
            "--baseline-lower", "480",
            "--baseline-upper", "520",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["normalized_width"] == 2.0
    assert result["status"] == "uncalibrated_uninformative"


def test_validate_cli_returns_safe_error_for_bad_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert main(["validate", str(path), "--kind", "answer-contract"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "JSONDecodeError"
    assert "Traceback" not in error["message"]
