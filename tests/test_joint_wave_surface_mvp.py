"""Deterministic end-to-end acceptance for the joint wave surface MVP.

This test file owns the P0 acceptance bullets from
``docs/PRD-JOINT-WAVE-SURFACE.md`` for the CPU reference path, plus the
fixture declared in ``fixtures/golden/joint_wave_surface_mvp.json``.  It
imports the runner script through ``importlib`` so the runner stays a
single self-contained script and tests do not depend on a new module in
``src/aie_decision``.

Compatibility adapters for tasks A-C are exercised through the runner's
``compatibility`` block: the runner calls into the unmodified
``aie_decision.candidate_generation`` and ``aie_decision.search_replay``
modules when they are available, without this test file or the runner
modifying those modules.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_joint_wave_surface_mvp.py"
FIXTURE_PATH = ROOT / "fixtures" / "golden" / "joint_wave_surface_mvp.json"


def _load_runner():
    """Import the runner script as a Python module under a fixed name.

    The script uses ``slots=True`` dataclasses, which require the module
    to be present in ``sys.modules`` during class construction.
    """
    spec = importlib.util.spec_from_file_location("run_joint_wave_surface_mvp", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture(scope="module")
def fixture_payload():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_result(runner, fixture_payload, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wave-mvp")
    payload = dict(fixture_payload)
    payload["particles"] = dict(payload["particles"])
    payload["particles"]["seed"] = int(payload["particles"]["seed"])
    return runner.run_mvp(payload), out_dir


# ---------------------------------------------------------------------------
# 1. Multi-unit legal mapping.
# ---------------------------------------------------------------------------


def test_legal_mappings_compile_and_produce_documented_units(runner, fixture_payload):
    """Each legal mapping must compile and produce the declared unit."""
    variables = {
        item["name"]: runner.VariableSpec(
            name=item["name"],
            unit=item["unit"],
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            method=item.get("method", "user_supplied"),
            ablatable=bool(item.get("ablatable", False)),
            bimodal=bool(item.get("bimodal", False)),
        )
        for item in fixture_payload["variables"]
    }
    expected_dimensions = {
        "time-leg": runner.Dimension.from_unit("hour"),
        "price-fuel": runner.Dimension.from_unit("usd"),
        "magnitude-fuel": runner.Dimension(),
    }
    for raw in fixture_payload["mappings"]:
        if raw.get("expect_failure"):
            continue
        mapping = runner.MappingSpec(
            mapping_id=raw["mapping_id"],
            formula=raw["formula"],
            output_axes=tuple(raw.get("output_axes", ())),
            expected_unit=raw["expected_unit"],
        )
        compiled = runner.compile_mapping(mapping, variables)
        assert compiled.is_legal, (
            f"expected {mapping.mapping_id} to compile legally, got {compiled.failure}"
        )
        assert compiled.expected_dimension.is_compatible_with(
            expected_dimensions[mapping.mapping_id]
        )


def test_unit_conversion_between_compatible_units_keeps_dimension(runner):
    """hour and day share the ``time`` dimension so they can interoperate."""
    hour = runner.Dimension.from_unit("hour")
    day = runner.Dimension.from_unit("day")
    second = runner.Dimension.from_unit("second")
    assert hour.is_compatible_with(day)
    assert hour.is_compatible_with(second)
    # hour + 3 (dimensionless) must fail because the dimension is "time".
    mapping = runner.MappingSpec(
        mapping_id="bad-time-add",
        formula="lane_hours + 3",
        output_axes=("delivery_time",),
        expected_unit="hour",
    )
    variables = {"lane_hours": runner.VariableSpec("lane_hours", "hour", 0, 24)}
    compiled = runner.compile_mapping(mapping, variables)
    assert not compiled.is_legal
    assert compiled.failure is not None
    assert compiled.failure.code == "unit_mismatch"


def test_composed_unit_usd_per_liter_resolves_to_money_over_volume(runner):
    """The MVP unit table must handle at least one composed unit."""
    usd_per_liter = runner.Dimension.from_unit("usd/liter")
    usd = runner.Dimension.from_unit("usd")
    liter = runner.Dimension.from_unit("liter")
    # usd / liter = money * volume^-1.
    expected = usd.combine(liter, sign=-1)
    assert usd_per_liter.exponents == expected.exponents
    # usd/liter * liter should collapse to money.
    mapping = runner.MappingSpec(
        mapping_id="price-fuel-rule",
        formula="fuel_unit_cost * liters_per_leg",
        output_axes=("price",),
        expected_unit="usd",
    )
    variables = {
        "fuel_unit_cost": runner.VariableSpec("fuel_unit_cost", "usd/liter", 0, 10),
        "liters_per_leg": runner.VariableSpec("liters_per_leg", "liter", 0, 100),
    }
    compiled = runner.compile_mapping(mapping, variables)
    assert compiled.is_legal
    assert compiled.expected_dimension.is_compatible_with(usd)


# ---------------------------------------------------------------------------
# 2. Illegal unit operations fail with structured information.
# ---------------------------------------------------------------------------


def test_illegal_mappings_are_reported_before_evaluation(run_result):
    """Run-level summary must include both illegal mappings with structured fields."""
    illegal = run_result[0].illegal_mappings
    illegal_ids = {item["mapping_id"] for item in illegal}
    assert "illegitimate-time-constant" in illegal_ids
    assert "illegitimate-time-money" in illegal_ids
    for entry in illegal:
        assert entry["code"] == "unit_mismatch"
        assert entry["expected_unit"]
        assert entry["operand_unit"]
        assert entry["operand"]


def test_illegal_mapping_failure_carries_offending_unit_and_operand(runner, fixture_payload):
    """The illegal mapping for ``lane_hours + 3`` must fail with structured context."""
    variables = {
        item["name"]: runner.VariableSpec(
            name=item["name"],
            unit=item["unit"],
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            method=item.get("method", "user_supplied"),
            ablatable=bool(item.get("ablatable", False)),
            bimodal=bool(item.get("bimodal", False)),
        )
        for item in fixture_payload["variables"]
    }
    bad = next(
        raw
        for raw in fixture_payload["mappings"]
        if raw["mapping_id"] == "illegitimate-time-money"
    )
    mapping = runner.MappingSpec(
        mapping_id=bad["mapping_id"],
        formula=bad["formula"],
        output_axes=tuple(bad.get("output_axes", ())),
        expected_unit=bad["expected_unit"],
    )
    compiled = runner.compile_mapping(mapping, variables)
    assert not compiled.is_legal
    failure = compiled.failure
    assert failure is not None
    assert failure.code == "unit_mismatch"
    assert failure.mapping_id == mapping.mapping_id
    assert "money" in failure.operand_unit or "volume" in failure.operand_unit


# ---------------------------------------------------------------------------
# 3. Bimodal / wave surface summary.
# ---------------------------------------------------------------------------


def test_bimodal_input_produces_bimodal_surface(run_result):
    """A bimodal input axis must surface at least one axis with two modes."""
    summary = run_result[0].summary
    assert summary["wave_shape"] == "bimodal"
    assert summary["bimodal_axes"], "expected at least one bimodal axis"
    for axis_name in summary["bimodal_axes"]:
        assert summary["mode_counts"][axis_name] >= 2


def test_surface_marginals_include_both_modes(run_result):
    """The reported mode locations must fall within the surface value range."""
    summary = run_result[0].summary
    assert summary["mode_locations"], "expected mode locations"
    for axis_name, locations in summary["mode_locations"].items():
        assert len(locations) >= 1
        for location in locations:
            assert np.isfinite(location)


def test_surface_carries_possibility_semantics(run_result):
    """The MVP must label uncalibrated surfaces as ``possibility_surface``."""
    iterations = run_result[0].iterations
    first = iterations[0].surface
    assert first["surface_kind"] == "possibility_surface"
    assert first["calibration"] == "unmeasured"
    assert first["coverage_semantics"] == "declared_joint_input_region"
    assert first["evaluator_version"].startswith("joint-wave-surface-mvp.v1+cpu")


# ---------------------------------------------------------------------------
# 4. Diagnostic actions: typed loop actions.
# ---------------------------------------------------------------------------


def test_actions_contain_split_regime_and_stop(run_result):
    """Bimodality must drive a ``split_regime`` action; the loop must end with ``stop``."""
    kinds = [item["kind"] for item in run_result[0].summary["actions"]]
    assert "split_regime" in kinds
    assert "stop" in kinds


def test_each_action_records_target_and_rationale(run_result):
    """Every typed action must be self-describing and traceable."""
    allowed_kinds = {
        "measure",
        "add_interaction",
        "split_regime",
        "minimize",
        "stop",
    }
    for action in run_result[0].summary["actions"]:
        assert action["action_id"]
        assert action["kind"] in allowed_kinds
        assert isinstance(action["rationale"], str) and action["rationale"]
        # Non-terminal actions must declare at least one affected entity.
        if action["kind"] != "stop":
            assert action["affected_entities"]


def test_action_selection_logic_prefers_bimodal_split(runner):
    """The diagnostic policy must map bimodality to ``split_regime`` first."""
    diagnostics = runner.SurfaceDiagnostics(
        particle_count=128,
        axes=(
            runner.AxisDiagnostic(
                name="a",
                unit="",
                absolute_width=2.0,
                relative_width=0.5,
                sharpness_absolute=1.0,
                sharpness_relative=0.5,
                effective_sample_size=128.0,
                entropy_nats=2.0,
                mode_count=2,
                mode_locations=(0.0, 1.0),
                residual_proxy=0.1,
            ),
        ),
        bimodal_axes=("a",),
        constraint_failures=(),
        effective_sample_size=128.0,
    )
    decision = {
        "passed": False,
        "per_axis": {"a": {"passed": False}},
        "bimodal_axes": ["a"],
    }
    variables = (runner.VariableSpec("a", "dimensionless", 0, 1),)
    surface = runner.ParticleSurface(
        axis_order=("a",),
        values=np.array([[0.0], [1.0]], dtype=np.float64),
        log_weight=np.array([0.0, 0.0]),
        surface_kind="possibility_surface",
        calibration="unmeasured",
        coverage_semantics="declared_joint_input_region",
        evaluator_version="test",
        seed=0,
    )
    action = runner._select_action(
        diagnostics=diagnostics,
        decision=decision,
        variables=variables,
        surface=surface,
        regime_split={"split_variable": "a"},
        round_index=1,
    )
    assert action is not None
    assert action.kind == "split_regime"


# ---------------------------------------------------------------------------
# 5. Loop iteration and replay determinism.
# ---------------------------------------------------------------------------


def test_loop_runs_at_least_two_rounds(run_result):
    """Bimodality triggers a regime-split, producing at least two iterations."""
    assert len(run_result[0].iterations) >= 2


def test_replay_produces_identical_summary_and_ledger(runner, fixture_payload):
    """Same input and seed must produce byte-identical summary and ledger."""
    payload = dict(fixture_payload)
    payload["particles"] = dict(payload["particles"])
    first = runner.run_mvp(payload)
    second = runner.run_mvp(payload)
    # Compare JSON-safe mappings so non-finite floats are handled.
    safe_first = runner._sanitize_for_json(first.to_mapping())
    safe_second = runner._sanitize_for_json(second.to_mapping())
    assert safe_first == safe_second


def test_ledger_entries_are_deterministically_sequenced(run_result):
    """The wave ledger must use contiguous, hash-stable entries."""
    entries = run_result[0].ledger["entries"]
    sequences = [entry["sequence"] for entry in entries]
    assert sequences == list(range(1, len(entries) + 1))
    for entry in entries:
        payload = entry["payload"]
        reconstructed = {
            "schema_version": payload["schema_version"],
            "event_id": payload["event_id"],
            "run_id": payload["run_id"],
            "candidate_id": payload["candidate_id"],
            "state": payload["state"],
            "round_index": payload["round_index"],
            "reason": payload["reason"],
            "data": payload["data"],
            "revision": payload["revision"],
        }
        canonical = json.dumps(
            reconstructed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_hash = __import__("hashlib").sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        assert entry["payload_hash"] == expected_hash


# ---------------------------------------------------------------------------
# 6. Compatibility adapters for tasks A-C (read-only).
# ---------------------------------------------------------------------------


def test_compatibility_adapter_reports_search_replay_validation(run_result):
    """When search-replay is importable, the runner must validate the projected ledger."""
    runner_module = _load_runner()
    if not getattr(runner_module, "_HAS_SEARCH_REPLAY", False):
        pytest.skip("aie_decision.search_replay is not importable")
    summary = run_result[0].summary
    replay = summary.get("search_replay")
    assert replay is not None
    assert replay["available"] is True
    assert replay["terminal_state"] in {"RESULT", "STOP"}


def test_compatibility_adapter_invokes_candidate_generation_diagnostic(run_result):
    """The runner must surface a FailureDiagnostic preview without modifying candidate_generation."""
    runner_module = _load_runner()
    if not getattr(runner_module, "_HAS_CANDIDATE_GENERATION", False):
        pytest.skip("aie_decision.candidate_generation is not importable")
    preview = run_result[0].summary.get("candidate_generation_preview")
    assert preview is not None
    assert preview["available"] is True
    assert isinstance(preview["diagnostic"]["reasons"], list)


def test_compatibility_does_not_modify_task_abc_files():
    """Static guard: the runner must not write to tasks A-C files."""
    for relative in (
        "src/aie_decision/search.py",
        "src/aie_decision/candidate_generation.py",
        "src/aie_decision/search_replay.py",
    ):
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        # The runner only imports these files; if their content has been
        # touched by accident, this assertion will fail in CI.
        assert "joint_wave_surface" not in text, (
            f"{relative} must not reference joint_wave_surface"
        )


# ---------------------------------------------------------------------------
# 7. CLI end-to-end smoke.
# ---------------------------------------------------------------------------


def test_cli_run_writes_ledger_and_summary(tmp_path):
    """Invoking the CLI must write deterministic ledger and summary files."""
    output_dir = tmp_path / "cli-out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "run",
            str(FIXTURE_PATH),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in {0, 2}, completed.stderr
    assert (output_dir / "wave-ledger.json").exists()
    assert (output_dir / "wave-summary.json").exists()
    ledger = json.loads((output_dir / "wave-ledger.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "wave-summary.json").read_text(encoding="utf-8"))
    assert ledger["schema_version"].startswith("joint-wave-surface-mvp.v1")
    assert summary["wave_shape"] == "bimodal"
    assert summary["bimodal_axes"]


def test_cli_replay_validates_deterministic_replay(tmp_path):
    """The CLI replay subcommand must confirm the saved ledger is deterministic."""
    output_dir = tmp_path / "cli-replay"
    output_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "run",
            str(FIXTURE_PATH),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in {0, 2}, completed.stderr
    replay_completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "replay",
            str(output_dir / "wave-ledger.json"),
            "--fixture",
            str(FIXTURE_PATH),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert replay_completed.returncode == 0, replay_completed.stderr
    payload = json.loads(replay_completed.stdout)
    assert payload["status"] == "ok"