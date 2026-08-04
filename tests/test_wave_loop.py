"""Narrow unit tests for the diagnostic-driven wave-loop."""

import copy
import json

import pytest

from aie_decision.wave_loop import (
    JOINT_SCHEMA_VERSION,
    WAVE_CHECKPOINT_VERSION,
    WAVE_LEDGER_VERSION,
    WAVE_LOOP_RESULT_VERSION,
    CompiledFactorIR,
    LoopAction,
    ParticleSurface,
    WaveEvent,
    WaveLoopError,
    compile_factor_ir,
    create_wave_checkpoint,
    evaluate_particle_surface,
    replay_wave_ledger,
    run_wave_loop,
    validate_joint_schema,
    verify_wave_checkpoint,
)


def _base_payload(**overrides):
    payload = {
        "run_id": "wave-loop-tests",
        "schema_version": JOINT_SCHEMA_VERSION,
        "outcome_space": [
            {
                "axis_id": "time",
                "name": "event_time",
                "unit": "day",
                "absolute_tolerance": 2.0,
                "reference_value": 25,
            }
        ],
        "variable_specs": [
            {"name": "travel_days", "lower": 23, "upper": 27, "unit": "day", "status": "bounded"},
            {"name": "buffer_days", "lower": 0, "upper": 2, "unit": "day", "status": "bounded"},
        ],
        "mapping_specs": [
            {
                "mapping_id": "m1",
                "variable_names": ["travel_days", "buffer_days"],
                "formula": "travel_days + buffer_days",
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
    payload.update(overrides)
    return payload


def test_validate_joint_schema_accepts_minimal_payload():
    parsed = validate_joint_schema(_base_payload())

    assert parsed["run_id"] == "wave-loop-tests"
    assert [axis["axis_id"] for axis in parsed["axes"]] == ["time"]
    assert set(parsed["variables"]) == {"travel_days", "buffer_days"}
    assert [mapping["mapping_id"] for mapping in parsed["mappings"]] == ["m1"]
    assert parsed["budget"].particle_count == 32


def test_validate_joint_schema_rejects_missing_outcome_space():
    payload = _base_payload()
    payload["outcome_space"] = []

    with pytest.raises(WaveLoopError, match="outcome_space"):
        validate_joint_schema(payload)


def test_validate_joint_schema_rejects_duplicate_variable_names():
    payload = _base_payload()
    payload["variable_specs"].append(
        {"name": "travel_days", "lower": 0, "upper": 1, "unit": "day", "status": "bounded"}
    )

    with pytest.raises(WaveLoopError, match="must be unique"):
        validate_joint_schema(payload)


def test_validate_joint_schema_rejects_unknown_mapping_variable():
    payload = _base_payload()
    payload["mapping_specs"][0]["variable_names"] = ["travel_days", "missing_var"]

    with pytest.raises(WaveLoopError, match="unknown variable"):
        validate_joint_schema(payload)


def test_compile_factor_ir_records_variables_and_dimensionless_contract():
    mapping = {
        "mapping_id": "m1",
        "variable_names": ["travel_days", "buffer_days"],
        "formula": "travel_days + buffer_days",
    }
    compiled = compile_factor_ir(mapping)

    assert isinstance(compiled, CompiledFactorIR)
    assert compiled.referenced_variables == ("travel_days", "buffer_days")
    assert compiled.log_potential({"travel_days": 23.0, "buffer_days": 2.0}) == 25.0


def test_compile_factor_ir_rejects_unsafe_formula():
    with pytest.raises(WaveLoopError):
        compile_factor_ir(
            {
                "mapping_id": "m_bad",
                "variable_names": ["travel_days"],
                "formula": "__import__('os').system('echo x')",
            }
        )


def test_evaluate_particle_surface_is_deterministic_and_dimensioned():
    from random import Random

    variables = {
        "travel_days": {
            "name": "travel_days",
            "lower": 23.0,
            "upper": 27.0,
            "unit": "day",
            "status": "bounded",
            "method": "observed",
            "evidence_atom_id": None,
        },
        "buffer_days": {
            "name": "buffer_days",
            "lower": 0.0,
            "upper": 2.0,
            "unit": "day",
            "status": "bounded",
            "method": "observed",
            "evidence_atom_id": None,
        },
    }
    axes = (
        {
            "axis_id": "time",
            "name": "event_time",
            "unit": "day",
            "absolute_tolerance": 2.0,
            "reference_value": 25,
            "decision_useful": True,
            "loss_function": None,
        },
    )
    irs = (compile_factor_ir(_base_payload()["mapping_specs"][0]),)
    surface_a = evaluate_particle_surface(axes, variables, irs, Random(42), 32, 1)
    surface_b = evaluate_particle_surface(axes, variables, irs, Random(42), 32, 1)

    assert surface_a.surface_id == surface_b.surface_id
    assert surface_a.values == surface_b.values
    assert surface_a.marginals == surface_b.marginals
    assert surface_a.surface_version == "particle-surface.v1"
    assert surface_a.semantics == "possibility_surface"


def test_run_wave_loop_emits_typed_actions_for_diagnostics():
    payload = _base_payload(
        decision_policy={
            "relative_tolerance": 0.01,
            "min_effective_sample_size": 0.1,
            "min_action_benefit": 0.0,
            "residual_interaction_threshold": 0.05,
        }
    )

    result = run_wave_loop(payload)

    assert result["schema_version"] == WAVE_LOOP_RESULT_VERSION
    kinds = [action["action_kind"] for action in result["actions"]]
    assert "measure" in kinds
    assert "stop" in kinds
    assert result["decision_value"]["accepted"] is False
    assert result["surface"]["diagnostics"]["particle_count"] == 32
    states = [event["state"] for event in result["checkpoint"]["replay"]["events"]]
    assert states[0] == "DRAFT"
    assert states[-1] in {"ACCEPTED", "UNRESOLVED"}


def test_run_wave_loop_emits_split_regime_when_surface_is_bimodal():
    payload = _base_payload(
        outcome_space=[
            {
                "axis_id": "outcome",
                "name": "mix",
                "unit": "usd",
                "absolute_tolerance": 1.0,
                "reference_value": 80,
            }
        ],
        variable_specs=[{"name": "x", "lower": 0, "upper": 10, "unit": "usd", "status": "bounded"}],
        mapping_specs=[
            {
                "mapping_id": "m_mix",
                "variable_names": ["x"],
                "formula": "(x - 2) * (x - 2) * (x - 8) * (x - 8)",
            }
        ],
    )

    result = run_wave_loop(payload)
    kinds = [action["action_kind"] for action in result["actions"]]
    assert "split_regime" in kinds


def test_run_wave_loop_emits_add_interaction_when_axes_correlate():
    payload = _base_payload(
        outcome_space=[
            {
                "axis_id": "time",
                "name": "event_time",
                "unit": "day",
                "absolute_tolerance": 0.1,
                "reference_value": 25,
            },
            {
                "axis_id": "price",
                "name": "event_price",
                "unit": "usd",
                "absolute_tolerance": 5.0,
                "reference_value": 100,
            },
        ],
        variable_specs=[
            {"name": "base_days", "lower": 20, "upper": 30, "unit": "day", "status": "bounded"},
            {"name": "extra_days", "lower": 0, "upper": 5, "unit": "day", "status": "bounded"},
        ],
        mapping_specs=[
            {
                "mapping_id": "m_time",
                "variable_names": ["base_days", "extra_days"],
                "formula": "base_days + extra_days",
            },
            {
                "mapping_id": "m_price",
                "variable_names": ["base_days"],
                "formula": "base_days * 4",
            },
        ],
        decision_policy={
            "relative_tolerance": 0.01,
            "min_effective_sample_size": 0.1,
            "min_action_benefit": 0.0,
            "residual_interaction_threshold": 0.05,
        },
    )

    result = run_wave_loop(payload)
    kinds = [action["action_kind"] for action in result["actions"]]
    assert "add_interaction" in kinds


def test_run_wave_loop_emits_expand_variable_when_status_missing():
    payload = _base_payload(
        variable_specs=[
            {"name": "travel_days", "lower": 23, "upper": 27, "unit": "day", "status": "bounded"},
            {"name": "unknown_factor", "status": "missing"},
        ],
        mapping_specs=[
            {
                "mapping_id": "m1",
                "variable_names": ["travel_days"],
                "formula": "travel_days",
            }
        ],
    )

    result = run_wave_loop(payload)
    kinds = [action["action_kind"] for action in result["actions"]]
    assert "expand_variable" in kinds


def test_run_wave_loop_stops_with_decision_value_when_axis_useful():
    result = run_wave_loop(_base_payload())

    assert result["decision_value"]["accepted"] is True
    assert result["decision_value"]["stop_reason"] == "decision-value-met"
    assert [action["action_kind"] for action in result["actions"]] == ["stop"]


def test_run_wave_loop_is_deterministic_for_identical_inputs():
    payload = _base_payload()
    first = run_wave_loop(payload)
    second = run_wave_loop(payload)

    assert first["surface"]["surface_id"] == second["surface"]["surface_id"]
    assert first["surface"]["marginals"] == second["surface"]["marginals"]
    assert first["checkpoint"]["replay"] == second["checkpoint"]["replay"]


def test_run_wave_loop_writes_deterministic_ledger_events():
    payload = _base_payload(
        decision_policy={
            "relative_tolerance": 0.01,
            "min_effective_sample_size": 0.1,
            "min_action_benefit": 0.0,
            "residual_interaction_threshold": 0.05,
        }
    )
    result = run_wave_loop(payload)
    ledger = result["ledger"]

    assert ledger["run_id"] == "wave-loop-tests"
    sequences = [entry["sequence"] for entry in ledger["entries"]]
    assert sequences == list(range(1, len(sequences) + 1))
    for entry in ledger["entries"]:
        assert entry["record_type"] == "wave_event"
        assert entry["stable_id"] == entry["payload"]["event_id"]
        assert entry["revision_id"] == entry["payload"]["revision"]["revision_id"]


def test_replay_wave_ledger_rebuilds_loop_state():
    payload = _base_payload()
    result = run_wave_loop(payload)

    replay = replay_wave_ledger(result["ledger"])

    assert replay["schema_version"] == "joint-wave-replay.v1"
    assert replay["run_id"] == "wave-loop-tests"
    assert replay["event_count"] == len(result["ledger"]["entries"])
    assert replay["accepted"] is True
    assert replay["terminal"]["state"] in {"ACCEPTED", "UNRESOLVED"}


def test_replay_wave_ledger_detects_payload_tampering():
    ledger = copy.deepcopy(run_wave_loop(_base_payload())["ledger"])
    ledger["entries"][1]["payload"]["reason"] = "rewritten"

    with pytest.raises(WaveLoopError, match="payload hash mismatch"):
        replay_wave_ledger(ledger)


def test_replay_wave_ledger_detects_sequence_gap():
    ledger = copy.deepcopy(run_wave_loop(_base_payload())["ledger"])
    ledger["entries"][1]["sequence"] = 99

    with pytest.raises(WaveLoopError, match="contiguous"):
        replay_wave_ledger(ledger)


def test_replay_wave_ledger_rejects_unknown_record_type():
    ledger = copy.deepcopy(run_wave_loop(_base_payload())["ledger"])
    ledger["entries"][1]["record_type"] = "search_event"

    with pytest.raises(WaveLoopError, match="not a wave_event"):
        replay_wave_ledger(ledger)


def test_create_wave_checkpoint_is_json_serializable_and_verifiable():
    ledger = run_wave_loop(_base_payload())["ledger"]
    checkpoint = create_wave_checkpoint(ledger)
    restored = verify_wave_checkpoint(json.loads(json.dumps(checkpoint)))

    assert restored == replay_wave_ledger(ledger)


def test_verify_wave_checkpoint_detects_hash_tampering():
    ledger = run_wave_loop(_base_payload())["ledger"]
    checkpoint = create_wave_checkpoint(ledger)
    tampered = copy.deepcopy(checkpoint)
    tampered["replay"]["accepted"] = False

    with pytest.raises(WaveLoopError, match="checkpoint hash mismatch"):
        verify_wave_checkpoint(tampered)


def test_verify_wave_checkpoint_detects_replay_state_mismatch():
    ledger = run_wave_loop(_base_payload())["ledger"]
    checkpoint = create_wave_checkpoint(ledger)
    tampered = copy.deepcopy(checkpoint)
    tampered["replay"]["event_count"] = 999
    # recompute the hash so the checkpoint_hash now matches the tampered body
    from aie_decision.wave_loop import _canonical, _digest

    tampered["checkpoint_hash"] = _digest(
        {key: value for key, value in tampered.items() if key != "checkpoint_hash"}
    )

    with pytest.raises(WaveLoopError, match="checkpoint replay state mismatch"):
        verify_wave_checkpoint(tampered)


def test_wave_loop_rejects_unsafe_mapping_formula():
    payload = _base_payload()
    payload["mapping_specs"][0]["formula"] = "open('x','r')"

    with pytest.raises(WaveLoopError):
        run_wave_loop(payload)


def test_wave_loop_emits_minimize_when_some_mappings_are_degenerate():
    payload = _base_payload(
        mapping_specs=[
            {
                "mapping_id": "m_zero_a",
                "variable_names": ["travel_days", "buffer_days"],
                "formula": "0",
            },
            {
                "mapping_id": "m_zero_b",
                "variable_names": ["travel_days", "buffer_days"],
                "formula": "0",
            },
            {
                "mapping_id": "m_sum",
                "variable_names": ["travel_days", "buffer_days"],
                "formula": "travel_days + buffer_days",
            },
        ]
    )

    result = run_wave_loop(payload)
    kinds = [action["action_kind"] for action in result["actions"]]
    assert "minimize" in kinds
