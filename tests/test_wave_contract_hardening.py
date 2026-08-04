from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_joint_wave_surface_mvp.py"
MATRIX = ROOT / "contracts" / "wave_mvp_scenario_matrix.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("wave_mvp_contract_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    return json.loads((ROOT / "fixtures/golden/joint_wave_surface_mvp.json").read_text(encoding="utf-8"))


def test_scenario_matrix_is_machine_readable_and_points_to_real_tests():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert matrix["schema_version"] == "wave-mvp-scenario-matrix.v1"
    source = (ROOT / "tests/test_joint_wave_surface_mvp.py").read_text(encoding="utf-8")
    source += (ROOT / "tests/test_wave_contract_hardening.py").read_text(encoding="utf-8")
    assert matrix["scenarios"]
    for scenario in matrix["scenarios"]:
        assert scenario["test"] in source


def test_missing_required_adapter_fails_closed(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "_HAS_SEARCH_REPLAY", False)
    with pytest.raises(runner.IntegrationUnavailable, match="integration_unavailable"):
        runner.run_mvp(_fixture())


def test_oracle_mismatch_is_not_accepted():
    runner = _load_runner()
    with pytest.raises(runner.ParityMismatch, match="parity_mismatch"):
        runner.assert_surface_parity(
            {"wave_shape": "bimodal", "bimodal_axes": ["time"], "actions": [], "final_status": "budget-exhausted"},
            {"wave_shape": "unimodal", "bimodal_axes": [], "actions": [], "final_status": "result-found"},
        )


def test_completion_record_requires_quality_evidence():
    validator_path = ROOT / "scripts/validate_wave_mvp_completion.py"
    spec = importlib.util.spec_from_file_location("completion_validator", validator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="completion_evidence_missing"):
        module.validate_completion({"schema_version": "wave-mvp-completion.v1"})


def test_acceptance_source_has_no_skip_or_noop_degradation():
    runner_source = RUNNER.read_text(encoding="utf-8")
    acceptance_source = (ROOT / "tests/test_joint_wave_surface_mvp.py").read_text(encoding="utf-8")
    assert "degrade to no-ops" not in runner_source
    assert "pytest.skip" not in acceptance_source
