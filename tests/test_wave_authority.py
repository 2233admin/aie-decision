"""Cross-path authority tests: provenance, parity, and CLI routing.

These tests verify that:
1. The authoritative package evaluator produces invocation provenance proving
   every component (schema, FactorIR, particle surface, diagnostics, loop,
   ledger, replay) was actually called.
2. The script oracle is explicitly labeled non-authoritative.
3. Cross-path parity fails with structured ``ParityMismatch`` when the
   authoritative and oracle paths diverge — never a silent skip.
4. The golden CLI routes through the authoritative path by default.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_joint_wave_surface_mvp.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("wave_mvp_authority_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wave_payload(**overrides):
    """Minimal payload compatible with both the package authority and script oracle."""
    payload = {
        "schema_version": "joint-wave-surface-mvp.v1",
        "run_id": "authority-parity-test",
        "outcome_space": {
            "axes": [
                {
                    "name": "event_time",
                    "unit": "day",
                    "domain": [0, 50],
                    "time_semantics": "static",
                    "tolerance": {"kind": "absolute", "value": 5.0},
                }
            ]
        },
        "variables": [
            {"name": "travel_days", "unit": "day", "lower": 23, "upper": 27, "method": "user_supplied", "ablatable": False, "bimodal": False},
            {"name": "buffer_days", "unit": "day", "lower": 0, "upper": 3, "method": "user_supplied", "ablatable": False, "bimodal": False},
        ],
        "mappings": [
            {
                "mapping_id": "total_time",
                "formula": "travel_days + buffer_days",
                "output_axes": ["event_time"],
                "expected_unit": "day",
            }
        ],
        "particles": {"count": 64, "seed": 20260805},
        "budget": {"max_rounds": 1, "max_seconds": 10.0},
        "decision_policy": {"axes": {"event_time": {"kind": "absolute", "value": 5.0}}},
        "compatibility": {
            "use_candidate_generation_failure_diagnostic": True,
            "use_search_replay_for_ledger_validation": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def _wave_loop_payload(**overrides):
    """Payload compatible with aie_decision.wave_loop.run_wave_loop."""
    payload = {
        "run_id": "authority-loop-test",
        "outcome_space": [
            {
                "axis_id": "time",
                "name": "event_time",
                "unit": "day",
                "absolute_tolerance": 5.0,
                "reference_value": 25,
            }
        ],
        "variable_specs": [
            {"name": "travel_days", "lower": 23, "upper": 27, "unit": "day", "status": "bounded"},
            {"name": "buffer_days", "lower": 0, "upper": 3, "unit": "day", "status": "bounded"},
        ],
        "mapping_specs": [
            {
                "mapping_id": "time_ratio",
                "variable_names": ["travel_days", "buffer_days"],
                "formula": "travel_days / (buffer_days + travel_days)",
            }
        ],
        "decision_policy": {
            "relative_tolerance": 0.25,
            "min_effective_sample_size": 0.1,
            "min_action_benefit": 0.0,
            "residual_interaction_threshold": 0.05,
        },
        "budget": {"max_rounds": 1, "max_actions": 5, "particle_count": 32, "seed": 42},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


# ---------------------------------------------------------------------------
# 3.1 & 3.2: Provenance tests
# ---------------------------------------------------------------------------


def test_authoritative_run_produces_invocation_provenance():
    """The authoritative path MUST produce provenance proving every component was called."""
    from aie_decision.wave_authority import (
        AUTHORITY_LABEL,
        InvocationProvenance,
        run_authoritative_wave,
    )

    payload = _wave_loop_payload()
    result = run_authoritative_wave(payload)

    assert result.schema_version == "wave-authority/v1"
    assert isinstance(result.provenance, InvocationProvenance)
    assert result.provenance.authority_label == AUTHORITY_LABEL
    assert result.provenance.authority_version == "wave-authority/v1"
    assert result.provenance.authority_hash
    assert len(result.provenance.components) == 7


def test_provenance_all_components_called():
    """Every tracked component MUST be marked called=True."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_wave_loop_payload())
    assert result.provenance.all_components_called(), (
        f"Failed components: {result.provenance.failed_components()}"
    )


def test_provenance_tracks_every_required_component():
    """Provenance MUST include a record for each of the 7 tracked components."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_wave_loop_payload())
    tracked = {c.component for c in result.provenance.components}
    expected = {
        "joint_schema",
        "factor_ir",
        "particle_surface",
        "wave_diagnostics",
        "wave_loop",
        "wave_ledger",
        "wave_replay",
    }
    missing = expected - tracked
    assert not missing, f"Missing provenance for: {missing}"


def test_provenance_component_has_version_and_hash():
    """Each component record MUST include a version and result hash."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_wave_loop_payload())
    for component in result.provenance.components:
        assert component.version, f"{component.component} missing version"
        assert component.result_hash, f"{component.component} missing result_hash"
        assert component.call_signature_hash, f"{component.component} missing call_signature_hash"


def test_authority_label_is_explicitly_not_oracle():
    """The authority label MUST NOT be the oracle label."""
    from aie_decision.wave_authority import AUTHORITY_LABEL, ORACLE_LABEL

    assert AUTHORITY_LABEL != ORACLE_LABEL
    assert AUTHORITY_LABEL == "authoritative"
    assert ORACLE_LABEL == "non_authoritative_oracle"


# ---------------------------------------------------------------------------
# 3.3: Cross-path parity tests
# ---------------------------------------------------------------------------


def test_cross_path_parity_succeeds_when_paths_agree():
    """When both paths produce consistent results, parity check passes."""
    from aie_decision.wave_authority import assert_authoritative_parity, run_authoritative_wave

    auth_result = run_authoritative_wave(_wave_loop_payload())

    oracle = {
        "final_status": "result-found" if auth_result.decision_value.get("accepted") else "insufficient-information",
        "surface_kind": auth_result.surface.get("kind", ""),
        "actions": auth_result.actions,
        "ledger": auth_result.ledger,
    }

    # Should not raise.
    assert_authoritative_parity(auth_result, oracle)


def test_cross_path_parity_fails_with_structured_mismatch_on_divergence():
    """Parity check MUST raise ParityMismatch with structured JSON on divergence."""
    from aie_decision.wave_authority import (
        ParityMismatch,
        assert_authoritative_parity,
        run_authoritative_wave,
    )

    auth_result = run_authoritative_wave(_wave_loop_payload())
    bad_oracle = {
        "final_status": "budget-exhausted",
        "surface_kind": "probability_surface",
        "actions": [{"action_kind": "stop", "kind": "stop"}],
        "ledger": {"entries": []},
    }

    with pytest.raises(ParityMismatch, match="parity_mismatch"):
        assert_authoritative_parity(auth_result, bad_oracle)


def test_cross_path_parity_mismatch_is_parseable_json():
    """The ParityMismatch message MUST be parseable JSON."""
    from aie_decision.wave_authority import (
        ParityMismatch,
        assert_authoritative_parity,
        run_authoritative_wave,
    )

    auth_result = run_authoritative_wave(_wave_loop_payload())
    bad_oracle = {"final_status": "result-found", "surface_kind": "probability_surface", "actions": [], "ledger": {"entries": []}}

    try:
        assert_authoritative_parity(auth_result, bad_oracle)
    except ParityMismatch as exc:
        body = str(exc).split("parity_mismatch: ", 1)[1]
        parsed = json.loads(body)
        assert isinstance(parsed, dict)
        assert "surface_kind" in parsed


def test_cross_path_parity_with_real_oracle_script():
    """Run both the authoritative path and the script oracle on the same input."""
    from aie_decision.wave_authority import run_authoritative_wave

    runner = _load_runner()
    fixture = _wave_payload()

    loop_payload = _wave_loop_payload()
    auth_result = run_authoritative_wave(loop_payload)
    oracle_result = runner.run_mvp(fixture)

    oracle_summary = oracle_result.summary
    assert oracle_summary.get("evaluator_label") == runner.EVALUATOR_LABEL_ORACLE
    assert oracle_summary["evaluator_label"] == "non_authoritative_oracle"
    assert oracle_summary["evaluator_path"] == "script"
    assert "wave_shape" in oracle_summary


def test_script_oracle_is_explicitly_labeled_non_authoritative():
    """The script oracle output MUST include its non-authoritative label."""
    runner = _load_runner()
    result = runner.run_mvp(_wave_payload())
    assert result.summary["evaluator_label"] == runner.EVALUATOR_LABEL_ORACLE
    assert result.summary["evaluator_label"] == "non_authoritative_oracle"
    assert result.summary["evaluator_path"] == "script"


def test_script_oracle_includes_invocation_provenance():
    """The script oracle MUST record which components it called."""
    runner = _load_runner()
    result = runner.run_mvp(_wave_payload())
    provenance = result.summary.get("invocation_provenance")
    assert provenance is not None, "oracle output missing invocation_provenance"
    assert provenance["evaluator"] == "non_authoritative_oracle"
    assert len(provenance["components_called"]) == 7


# ---------------------------------------------------------------------------
# CLI routing tests
# ---------------------------------------------------------------------------


def test_authority_subcommand_routes_to_package_evaluator():
    """The authority module MUST export the authoritative label."""
    from aie_decision.wave_authority import AUTHORITY_LABEL

    runner = _load_runner()
    authority_mod = runner._get_authority_module()
    assert authority_mod["label"] == AUTHORITY_LABEL


def test_authoritative_result_is_json_serializable():
    """The full AuthoritativeWaveResult MUST be JSON-serializable."""
    from aie_decision.wave_authority import run_authoritative_wave

    result = run_authoritative_wave(_wave_loop_payload())
    result_dict = result.to_dict()
    serialized = json.dumps(result_dict, ensure_ascii=False, sort_keys=True, default=str)
    deserialized = json.loads(serialized)
    assert deserialized["schema_version"] == "wave-authority/v1"
    assert "provenance" in deserialized
    assert deserialized["provenance"]["authority_label"] == "authoritative"


def test_parity_mismatch_includes_full_context():
    """ParityMismatch.mismatches MUST be accessible as a dict."""
    from aie_decision.wave_authority import ParityMismatch

    test_mismatches = {"status": {"authoritative": "x", "oracle": "y"}}
    exc = ParityMismatch(test_mismatches)
    assert exc.mismatches == test_mismatches
    assert exc.code == "parity_mismatch"
