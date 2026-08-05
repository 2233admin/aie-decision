"""Independent adversarial review tests for the contract-driven wave MVP.

These tests own Task 4.2 (diff review for assertion weakening, skips, no-op
fallbacks, duplicate evaluators) and the requirement side of Task 3.3
(cross-path parity).  They attack **behaviour and call evidence** — they
never repeat implementation constants or mirror existing assertions.

Every test is paired with a scenario in
``contracts/wave_mvp_adversarial_review_matrix.json``.  A test that cannot
be satisfied by the current design is left to **fail honestly** so the
review captures the gap.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "fixtures" / "golden" / "joint_wave_surface_mvp.json"
RUNNER_PATH = ROOT / "scripts" / "run_joint_wave_surface_mvp.py"
MATRIX_PATH = ROOT / "contracts" / "wave_mvp_adversarial_review_matrix.json"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_runner():
    spec = importlib.util.spec_from_file_location("adversarial_wave_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_wave_loop():
    """Load the package path evaluator (wave_loop module)."""
    from aie_decision import wave_loop

    return wave_loop


def _fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _adversarial_matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Matrix self-validation
# ---------------------------------------------------------------------------


def test_adversarial_matrix_is_machine_readable_and_maps_to_these_tests():
    """Every adversarial scenario must reference a test that exists in this file."""
    matrix = _adversarial_matrix()
    assert matrix["schema_version"] == "wave-mvp-adversarial-review-matrix.v1"
    assert matrix["review_tasks"] == ["4.2", "3.3-requirement-side"]
    source = (ROOT / "tests" / "test_wave_mvp_adversarial_review.py").read_text(
        encoding="utf-8"
    )
    for scenario in matrix["scenarios"]:
        assert scenario["test"] in source, (
            f"adversarial scenario {scenario['id']} references "
            f"test {scenario['test']} not found in this file"
        )


# ---------------------------------------------------------------------------
# 1. Illegal unit — structured failure completeness
# ---------------------------------------------------------------------------


def test_adversarial_illegal_unit_carries_complete_structured_error():
    """Attack: an illegal unit failure must report EVERY required field
    with a non-empty value, not just exist.

    The spec requires ``unit_mismatch`` plus mapping id, operand,
    expected unit, and actual unit.  We verify each illegal mapping in
    the fixture produces a failure object where none of these fields
    are missing or empty.
    """
    runner = _load_runner()
    fixture = _fixture()
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
        for item in fixture["variables"]
    }
    required_fields = ("mapping_id", "code", "operand", "operand_unit", "expected_unit")
    failures_found = 0

    for raw in fixture["mappings"]:
        mapping = runner.MappingSpec(
            mapping_id=raw["mapping_id"],
            formula=raw["formula"],
            output_axes=tuple(raw.get("output_axes", ())),
            expected_unit=raw["expected_unit"],
            expect_failure=raw.get("expect_failure"),
        )
        compiled = runner.compile_mapping(mapping, variables)
        if not compiled.is_legal and compiled.failure is not None:
            failures_found += 1
            failure = compiled.failure
            for field in required_fields:
                value = getattr(failure, field, None)
                assert value, (
                    f"illegal mapping {raw['mapping_id']}: "
                    f"failure.{field} is empty or missing"
                )
            assert failure.code == "unit_mismatch", (
                f"expected unit_mismatch for {raw['mapping_id']}, "
                f"got {failure.code}"
            )
            # The actual unit (operand_unit) must differ from expected_unit
            # for this to be a genuine mismatch, not a vacuous report.
            assert failure.operand_unit != failure.expected_unit or "dimensionless" not in (
                failure.operand_unit,
                failure.expected_unit,
            ), (
                f"vacuous unit_mismatch: operand_unit={failure.operand_unit} "
                f"expected_unit={failure.expected_unit} differ only trivially"
            )

    assert failures_found >= 2, (
        f"expected at least 2 illegal mapping failures, found {failures_found}"
    )


# ---------------------------------------------------------------------------
# 2. Missing required adapter — each boundary independently
# ---------------------------------------------------------------------------


def test_adversarial_each_missing_adapter_fails_independently():
    """Attack: remove each required adapter independently and verify the
    failure message names the SPECIFIC missing adapter, not a generic
    message.

    The existing test only monkeypatches search_replay.  We test both
    boundaries separately.
    """
    runner = _load_runner()
    fixture = _fixture()

    # --- candidate_generation missing only ---
    with mock.patch.object(runner, "_HAS_CANDIDATE_GENERATION", False):
        with mock.patch.object(runner, "_HAS_SEARCH_REPLAY", True):
            with pytest.raises(
                runner.IntegrationUnavailable, match="integration_unavailable"
            ) as exc_info:
                runner.run_mvp(dict(fixture))
            msg = str(exc_info.value)
            assert "candidate_generation" in msg, (
                f"missing candidate_generation not named in error: {msg}"
            )

    # --- search_replay missing only ---
    with mock.patch.object(runner, "_HAS_CANDIDATE_GENERATION", True):
        with mock.patch.object(runner, "_HAS_SEARCH_REPLAY", False):
            with pytest.raises(
                runner.IntegrationUnavailable, match="integration_unavailable"
            ) as exc_info:
                runner.run_mvp(dict(fixture))
            msg = str(exc_info.value)
            assert "search_replay" in msg, (
                f"missing search_replay not named in error: {msg}"
            )

    # --- both missing ---
    with mock.patch.object(runner, "_HAS_CANDIDATE_GENERATION", False):
        with mock.patch.object(runner, "_HAS_SEARCH_REPLAY", False):
            with pytest.raises(
                runner.IntegrationUnavailable, match="integration_unavailable"
            ) as exc_info:
                runner.run_mvp(dict(fixture))
            msg = str(exc_info.value)
            assert "candidate_generation" in msg, (
                f"both missing but candidate_generation not named: {msg}"
            )
            assert "search_replay" in msg, (
                f"both missing but search_replay not named: {msg}"
            )


# ---------------------------------------------------------------------------
# 3. Provenance — invocation not just declaration
# ---------------------------------------------------------------------------


def test_adversarial_provenance_invocation_not_just_declaration():
    """Attack: the summary declares ``search_replay_available: True``, but
    we must prove the replay function was actually CALLED, not just that
    the module was importable.

    We instrument the replay function with a spy and verify it received
    a call with a ledger-like payload.
    """
    runner = _load_runner()
    fixture = dict(_fixture())

    # The runner stores the replay function as _replay_search_ledger
    original_replay = runner._replay_search_ledger
    call_args = []

    def spy_replay(ledger):
        call_args.append(ledger)
        return original_replay(ledger)

    with mock.patch.object(runner, "_replay_search_ledger", spy_replay):
        result = runner.run_mvp(fixture)

    assert call_args, (
        "search_replay adapter was NEVER called during run_mvp — "
        "provenance is declared but not invoked"
    )
    assert len(call_args) == 1, (
        f"expected exactly 1 replay call, got {len(call_args)}"
    )
    called_ledger = call_args[0]
    assert isinstance(called_ledger, dict), "replay was not called with a dict"
    assert "entries" in called_ledger, "replay ledger has no entries key"

    # Verify the summary still declares availability
    assert result.summary["search_replay"]["available"] is True

    # Also verify candidate_generation preview is populated with real data
    cg_preview = result.summary.get("candidate_generation_preview")
    assert cg_preview is not None, "candidate_generation_preview missing from summary"
    assert cg_preview["available"] is True
    diagnostic = cg_preview.get("diagnostic")
    assert diagnostic is not None, "diagnostic missing from candidate_generation_preview"
    assert isinstance(diagnostic["reasons"], list), "diagnostic reasons is not a list"


# ---------------------------------------------------------------------------
# 4. Cross-path parity — package evaluator vs fixture runner (Task 3.3)
# ---------------------------------------------------------------------------


def test_adversarial_cross_path_parity_package_vs_runner():
    """Attack: the spec requires ONE authoritative evaluator path, but the
    codebase has TWO independent implementations (wave_loop.run_wave_loop
    in the package, and run_joint_wave_surface_mvp.run_mvp as the runner).

    We construct equivalent minimal inputs for both evaluators, run them,
    and compare surface semantics, terminal status, and action kinds.

    If the implementations produce different results, this test fails
    honestly — documenting the cross-path drift that Task 3.3 targets.

    If the schemas are incompatible (preventing a fair comparison), the
    test fails honestly, documenting the schema gap.
    """
    try:
        wave_loop = _load_wave_loop()
    except ImportError as exc:
        pytest.fail(f"Cannot import wave_loop package evaluator: {exc}")

    runner = _load_runner()

    # --- Build a minimal equivalent input ---
    # Runner format: outcome_space.axes[], variables[], mappings[]
    # Wave-loop format: outcome_space[], variable_specs[], mapping_specs[]

    # Minimal shared semantics: one axis (time), two observed variables,
    # one legal mapping, deterministic seed.
    run_id = "adversarial-cross-path"

    # --- Runner input ---
    runner_payload = {
        "run_id": run_id,
        "schema_version": "joint-wave-surface-mvp.v1",
        "outcome_space": {
            "axes": [
                {
                    "name": "total_time",
                    "unit": "hour",
                    "domain": [0, 48],
                    "time_semantics": "elapsed",
                    "tolerance": {"kind": "absolute", "value": 2.0, "unit": "hour"},
                }
            ]
        },
        "variables": [
            {"name": "drive_hours", "unit": "hour", "lower": 1.0, "upper": 4.0, "method": "observed"},
            {"name": "rest_hours", "unit": "hour", "lower": 0.5, "upper": 2.0, "method": "observed"},
        ],
        "mappings": [
            {
                "mapping_id": "time-sum",
                "formula": "drive_hours + rest_hours",
                "output_axes": ["total_time"],
                "expected_unit": "hour",
            }
        ],
        "particles": {"count": 64, "seed": 42},
        "budget": {"max_rounds": 1, "max_actions_per_round": 1, "max_seconds": 5.0},
        "decision_policy": {
            "axes": {
                "total_time": {"kind": "absolute", "value": 10.0, "unit": "hour"}
            }
        },
        "compatibility": {
            "use_candidate_generation_failure_diagnostic": True,
            "use_search_replay_for_ledger_validation": True,
        },
    }

    runner_result = runner.run_mvp(runner_payload)

    # --- Wave-loop input (package path) ---
    # Map to the schema expected by validate_joint_schema
    wave_loop_payload = {
        "run_id": run_id,
        "schema_version": "joint-wave-schema.v1",
        "outcome_space": [
            {
                "axis_id": "total_time",
                "name": "total_time",
                "unit": "hour",
                "absolute_tolerance": 10.0,
                "decision_useful": True,
            }
        ],
        "variable_specs": [
            {
                "name": "drive_hours",
                "unit": "hour",
                "status": "observed",
                "lower": 1.0,
                "upper": 4.0,
                "method": "observed",
            },
            {
                "name": "rest_hours",
                "unit": "hour",
                "status": "observed",
                "lower": 0.5,
                "upper": 2.0,
                "method": "observed",
            },
        ],
        "mapping_specs": [
            {
                "mapping_id": "time-sum",
                "variable_names": ["drive_hours", "rest_hours"],
                "formula": "drive_hours + rest_hours",
                "direction": "support",
            }
        ],
        "decision_policy": {
            "relative_tolerance": 0.25,
            "min_effective_sample_size": 0.1,
            "min_action_benefit": 0.0,
            "residual_interaction_threshold": 0.05,
        },
        "budget": {
            "max_rounds": 1,
            "max_actions": 1,
            "particle_count": 64,
            "seed": 42,
        },
    }

    try:
        wave_loop_result = wave_loop.run_wave_loop(wave_loop_payload)
    except Exception as exc:
        # Schema incompatibility: the wave_loop module may reject the
        # payload because its narrow schema differs from the runner's.
        # This is itself a finding for Task 3.3.
        pytest.fail(
            f"Cross-path parity blocked by schema gap: wave_loop.run_wave_loop "
            f"rejected the equivalent payload with {type(exc).__name__}: {exc}. "
            f"This documents that the package evaluator and runner use "
            f"incompatible schemas — both must be aligned per Task 3.1/3.2."
        )

    # --- Compare key outputs ---
    mismatches: dict[str, dict[str, object]] = {}

    # Surface semantics
    runner_surface_kind = None
    if runner_result.iterations:
        runner_surface_kind = runner_result.iterations[0].surface.get("surface_kind")
    wave_loop_surface_kind = wave_loop_result["surface"]["semantics"]

    if runner_surface_kind != wave_loop_surface_kind:
        mismatches["surface_kind"] = {
            "runner": runner_surface_kind,
            "wave_loop": wave_loop_surface_kind,
        }

    # Terminal status
    runner_status = runner_result.status
    wave_loop_accepted = wave_loop_result["decision_value"]["accepted"]
    wave_loop_status = "result-found" if wave_loop_accepted else "budget-exhausted"
    if runner_status != wave_loop_status:
        mismatches["terminal_status"] = {
            "runner": runner_status,
            "wave_loop": wave_loop_status,
        }

    # Action kinds
    runner_action_kinds = [
        a.get("kind") if isinstance(a, dict) else getattr(a, "kind", "?")
        for a in runner_result.actions
    ]
    wave_loop_action_kinds = [
        a["action_kind"] for a in wave_loop_result["actions"]
    ]
    if set(runner_action_kinds) != set(wave_loop_action_kinds):
        mismatches["action_kinds"] = {
            "runner": sorted(set(runner_action_kinds)),
            "wave_loop": sorted(set(wave_loop_action_kinds)),
        }

    if mismatches:
        mismatch_report = json.dumps(mismatches, indent=2, sort_keys=True)
        pytest.fail(
            f"Cross-path parity failure (Task 3.3): the package evaluator "
            f"and fixture runner produced different results for equivalent "
            f"inputs. This confirms the two-evaluator drift the spec warns "
            f"against. Mismatches:\n{mismatch_report}"
        )


# ---------------------------------------------------------------------------
# 5. Probability claim without calibration
# ---------------------------------------------------------------------------


def test_adversarial_no_uncalibrated_surface_labeled_as_probability():
    """Attack: verify that NO code path in the runner labels a surface as
    ``probability_surface`` when calibration is ``unmeasured``.

    We enumerate every iteration surface AND the accepted_surface,
    checking that surface_kind is never 'probability_surface'.
    """
    runner = _load_runner()
    fixture = _fixture()
    result = runner.run_mvp(dict(fixture))

    surfaces_to_check: list[dict] = []

    # All iteration surfaces
    for iteration in result.iterations:
        surfaces_to_check.append(iteration.surface)

    # The accepted surface (if any)
    if result.accepted_surface is not None:
        surfaces_to_check.append(result.accepted_surface)

    # The summary itself
    surfaces_to_check.append(result.summary)

    for idx, surface in enumerate(surfaces_to_check):
        kind = surface.get("surface_kind")
        calibration = surface.get("calibration")

        # The contract requires that uncalibrated surfaces are labelled
        # possibility_surface, never probability_surface.
        if kind == "probability_surface":
            pytest.fail(
                f"Surface #{idx} is labelled 'probability_surface' — "
                f"this violates the requirement that uncalibrated "
                f"surfaces MUST be labelled 'possibility_surface'. "
                f"Calibration value: {calibration!r}"
            )

        if calibration and calibration != "unmeasured":
            if kind != "probability_surface":
                # Not a failure per se, but worth flagging: a surface
                # claiming calibration but not calling itself a
                # probability surface is inconsistent.
                pass

    # Additionally: scan the runner source for any literal
    # "probability_surface" that is assigned to surface_kind outside of
    # a test/comment context.  This is a static guard.
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    # The runner explicitly sets surface_kind to "possibility_surface" —
    # verify that string appears and "probability_surface" does not as an
    # assignment target.
    assert '"possibility_surface"' in runner_source, (
        "runner source does not reference possibility_surface"
    )
    assert "probability_surface" not in runner_source, (
        "runner source references probability_surface — "
        "this may indicate a code path that labels uncalibrated "
        "surfaces as probability distributions"
    )


# ---------------------------------------------------------------------------
# 6. Budget exhaustion
# ---------------------------------------------------------------------------


def test_adversarial_budget_exhaustion_is_not_converged():
    """Attack: with a deliberately tight budget (max_rounds=1, non-bimodal
    input), the result MUST be budget-exhausted — never 'converged',
    'accepted', or 'result-found'.

    We also test that the runner correctly handles max_rounds edge cases.
    """
    runner = _load_runner()
    fixture = _fixture()

    # --- Unimodal, tight budget: should exhaust ---
    payload = dict(fixture)
    payload["particles"] = dict(payload.get("particles", {}))
    payload["particles"]["seed"] = int(payload["particles"]["seed"])
    payload["particles"]["count"] = 16
    payload["budget"] = {
        "max_rounds": 1,
        "max_actions_per_round": 1,
        "max_seconds": 1.0,
    }
    # Remove bimodal flags so the input is unimodal
    for var in payload["variables"]:
        var["bimodal"] = False
    # Remove regime_split to prevent split_regime from adding a round
    payload.pop("regime_split", None)
    # Tighten decision policy so it's hard to pass
    payload["decision_policy"] = {
        "axes": {
            "delivery_time": {"kind": "absolute", "value": 0.01, "unit": "hour"},
            "price": {"kind": "absolute", "value": 0.01, "unit": "usd"},
            "magnitude": {"kind": "loss_threshold", "value": 0.01, "unit": "dimensionless"},
        }
    }

    result = runner.run_mvp(payload)

    # The status must not claim convergence or success when budget is
    # exhausted.
    forbidden_statuses = {"converged", "accepted", "result-found"}
    assert result.status not in forbidden_statuses, (
        f"budget-exhausted run reported status '{result.status}' — "
        f"must NOT be one of {forbidden_statuses}"
    )
    # The spec requires unresolved/budget-exhausted status and rejects
    # converged/accepted.  "insufficient-information" is a valid
    # unresolved outcome when the loop has no actionable path (e.g.
    # unimodal input without regime_split).  Both are non-converged.
    valid_exhaust_statuses = {"budget-exhausted", "insufficient-information"}
    assert result.status in valid_exhaust_statuses, (
        f"expected one of {valid_exhaust_statuses}, got {result.status}"
    )

    # The summary must also reflect the budget status
    assert result.summary.get("final_status") != "converged", (
        "summary.final_status must not be 'converged'"
    )

    # --- Edge: max_rounds=0 should still produce a result ---
    edge_payload = dict(payload)
    edge_payload["budget"] = {
        "max_rounds": 0,
        "max_actions_per_round": 0,
        "max_seconds": 0.1,
    }
    edge_result = runner.run_mvp(edge_payload)
    # Must not crash; must produce a status
    assert edge_result.status in {
        "budget-exhausted",
        "insufficient-information",
    }, f"max_rounds=0 produced unexpected status: {edge_result.status}"


# ---------------------------------------------------------------------------
# 7. Action order — determinism and seed sensitivity
# ---------------------------------------------------------------------------


def test_adversarial_action_order_is_deterministic_and_seed_sensitive():
    """Attack: verify that action order is deterministic (same seed →
    same action sequence) and that changing the seed produces a
    materially different surface (proving the seed is not ignored).

    This guards against a "deterministic" implementation that ignores
    the seed and always returns the same result regardless.
    """
    runner = _load_runner()
    fixture = _fixture()

    def action_kinds_from_run(seed: int) -> list[str]:
        payload = dict(fixture)
        payload["particles"] = dict(payload.get("particles", {}))
        payload["particles"]["count"] = 128
        payload["particles"]["seed"] = seed
        result = runner.run_mvp(payload)
        return [
            a.get("kind") if isinstance(a, dict) else getattr(a, "kind", "?")
            for a in result.actions
        ]

    # Same seed → identical action kinds
    run_a = action_kinds_from_run(42)
    run_b = action_kinds_from_run(42)
    assert run_a == run_b, (
        f"same seed (42) produced different action orders: "
        f"{run_a} vs {run_b}"
    )

    # Also verify that same seed produces byte-identical JSON-safe results
    payload_template = dict(fixture)
    payload_template["particles"] = dict(payload_template.get("particles", {}))
    payload_template["particles"]["count"] = 128
    payload_template["particles"]["seed"] = 99

    p1 = dict(payload_template)
    p1["particles"] = dict(p1["particles"])
    p1["particles"]["seed"] = 99

    p2 = dict(payload_template)
    p2["particles"] = dict(p2["particles"])
    p2["particles"]["seed"] = 99

    r1 = runner.run_mvp(p1)
    r2 = runner.run_mvp(p2)

    safe1 = runner._sanitize_for_json(r1.to_mapping())
    safe2 = runner._sanitize_for_json(r2.to_mapping())
    assert safe1 == safe2, "same seed produced non-identical full results"

    # Different seed → at minimum, the surface values should differ.
    # We compare two distant seeds to avoid accidental collision.
    run_x = action_kinds_from_run(1)
    run_y = action_kinds_from_run(99999)

    # The action kinds COULD be the same for two seeds (if both produce
    # bimodal surfaces). But the full surface content should differ.
    p_x = dict(payload_template)
    p_x["particles"] = dict(p_x["particles"])
    p_x["particles"]["seed"] = 1

    p_y = dict(payload_template)
    p_y["particles"] = dict(p_y["particles"])
    p_y["particles"]["seed"] = 99999

    r_x = runner.run_mvp(p_x)
    r_y = runner.run_mvp(p_y)

    safe_x = runner._sanitize_for_json(r_x.to_mapping())
    safe_y = runner._sanitize_for_json(r_y.to_mapping())

    # At least one observable should differ between the two seeds.
    # Compare the mode locations (most seed-sensitive output).
    x_mode_locations = r_x.summary.get("mode_locations", {})
    y_mode_locations = r_y.summary.get("mode_locations", {})

    # If mode_locations are identical for two distant seeds, the seeding
    # may be a no-op.
    assert x_mode_locations != y_mode_locations or safe_x != safe_y, (
        "different seeds (1 vs 99999) produced byte-identical results — "
        "the seed appears to be ignored, which makes 'deterministic' "
        "replay vacuous"
    )


# ---------------------------------------------------------------------------
# 8. Replay identity — tamper detection
# ---------------------------------------------------------------------------


def test_adversarial_replay_identity_detects_tampered_ledger():
    """Attack: take a valid ledger from a real run, tamper with it in
    three different ways, and verify each tamper is detected through a
    structured error (not a silent pass).

    Tamper types:
    1. Change a payload_hash in one entry (integrity check)
    2. Change a round_index in one entry (semantic corruption)
    3. Change a state string in one entry (event forgery)
    """
    runner = _load_runner()
    fixture = dict(_fixture())

    # Produce a valid ledger
    payload = dict(fixture)
    payload["particles"] = dict(payload.get("particles", {}))
    payload["particles"]["seed"] = int(payload["particles"]["seed"])
    result = runner.run_mvp(payload)

    # Project to search-ledger schema (what replay consumes)
    projected = runner._project_wave_ledger_to_search_schema(
        result.ledger["entries"], result.run_id
    )
    projected = runner._sanitize_for_json(projected)

    # Verify the un-tampered ledger replays successfully
    clean = runner._replay_search_ledger(projected)
    assert clean["event_count"] > 0, "clean replay should have events"

    # --- Tamper 1: corrupt a payload_hash ---
    tampered_hash = copy.deepcopy(projected)
    if tampered_hash["entries"]:
        entry = tampered_hash["entries"][0]
        entry["payload_hash"] = entry["payload_hash"][::-1]  # reverse the hash
        with pytest.raises(Exception) as exc_info:
            runner._replay_search_ledger(tampered_hash)
        assert "hash" in str(exc_info.value).lower(), (
            f"payload_hash tamper should trigger hash error, got: {exc_info.value}"
        )

    # --- Tamper 2: change round_index ---
    tampered_round = copy.deepcopy(projected)
    for entry in tampered_round["entries"]:
        if "round_index" in entry.get("payload", {}):
            old_round = entry["payload"]["round_index"]
            entry["payload"]["round_index"] = old_round + 100
            # Recompute hash so it's a semantic corruption not a hash failure
            canonical = json.dumps(
                entry["payload"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            import hashlib
            entry["payload_hash"] = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
            break  # corrupt just one entry

    # The replay may or may not catch round_index corruption depending on
    # validation depth.  If it silently accepts, that's a finding.
    try:
        runner._replay_search_ledger(tampered_round)
        # If we reach here, the tampered round_index was silently accepted.
        # This is a real gap: the replay should validate event ordering.
        pytest.fail(
            "Tampered round_index was silently accepted by replay — "
            "the ledger does not validate round_index integrity"
        )
    except Exception:
        # Expected: tamper was detected
        pass

    # --- Tamper 3: change state string ---
    tampered_state = copy.deepcopy(projected)
    for entry in tampered_state["entries"]:
        payload_obj = entry.get("payload", {})
        if payload_obj.get("state") in ("RESULT", "STOP"):
            payload_obj["state"] = "EVALUATE"  # forge a non-terminal state
            canonical = json.dumps(
                payload_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            import hashlib
            entry["payload_hash"] = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
            break

    # If the tampered state goes undetected, report it.
    try:
        runner._replay_search_ledger(tampered_state)
        pytest.fail(
            "Tampered state string was silently accepted by replay — "
            "the ledger does not validate state transitions"
        )
    except Exception:
        # Expected
        pass


# ---------------------------------------------------------------------------
# Task 4.2: Diff review — assertion weakening, skips, no-op fallbacks
# ---------------------------------------------------------------------------


def test_adversarial_no_assertion_weakening_in_diff():
    """Static guard: the hardening diff must not introduce weaker
    assertions, conditional skips, or no-op fallbacks.

    We inspect the runner source for patterns that would weaken
    the fail-closed contract.
    """
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")

    weakening_patterns = [
        # Conditional skips that hide missing functionality
        ("pytest.skip", "pytest.skip found in runner — must fail closed"),
        ("pytest.mark.skip", "pytest.mark.skip found in runner"),
        # No-op fallbacks
        ("pass  # noqa", "possible no-op exception handler"),
        ("except Exception:\n        pass", "bare-pass exception handler"),
        # Setting flags that weaken fail-closed behavior
        ("_HAS_CANDIDATE_GENERATION = True  #", "hardcoded adapter flag"),
        ("_HAS_SEARCH_REPLAY = True  #", "hardcoded adapter flag"),
    ]

    for pattern, message in weakening_patterns:
        assert pattern not in runner_source, (
            f"Assertion weakening detected: {message}"
        )

    # Also verify the runner uses IntegrationUnavailable (fail-closed)
    # rather than logging a warning and continuing.
    assert "IntegrationUnavailable" in runner_source, (
        "runner must use IntegrationUnavailable for fail-closed semantics"
    )


def test_adversarial_no_duplicate_evaluator_capability():
    """Guard: the runner and wave_loop both implement full evaluators
    with overlapping capability.  This test documents the current state:
    are there two independent evaluator paths?  If so, which capabilities
    overlap?

    This is a documentation test for Task 4.2 — it does not prescribe a
    fix, but records the finding.
    """
    runner = _load_runner()
    try:
        wave_loop = _load_wave_loop()
    except ImportError as exc:
        pytest.fail(
            "wave_loop must be importable for the duplicate-evaluator audit; "
            f"missing capability is a failure, not a skip: {exc}"
        )

    # --- Capability inventory ---
    runner_capabilities = {
        "schema_validation": hasattr(runner, "SCHEMA_VERSION"),
        "unit_analysis": hasattr(runner, "Dimension"),
        "mapping_compilation": hasattr(runner, "compile_mapping"),
        "particle_evaluation": hasattr(runner, "_evaluate_surface"),
        "surface_diagnostics": hasattr(runner, "compute_diagnostics"),
        "decision_value": hasattr(runner, "evaluate_decision_value"),
        "action_selection": hasattr(runner, "_select_action"),
        "loop_orchestration": hasattr(runner, "run_mvp"),
        "ledger_projection": hasattr(runner, "_project_wave_ledger_to_search_schema"),
        "parity_check": hasattr(runner, "assert_surface_parity"),
    }

    wl_capabilities = {
        "schema_validation": hasattr(wave_loop, "validate_joint_schema"),
        "factor_ir": hasattr(wave_loop, "compile_factor_ir"),
        "particle_surface": hasattr(wave_loop, "evaluate_particle_surface"),
        "loop_orchestration": hasattr(wave_loop, "run_wave_loop"),
        "ledger_replay": hasattr(wave_loop, "replay_wave_ledger"),
        "checkpoint": hasattr(wave_loop, "create_wave_checkpoint"),
    }

    # Both have loop orchestration, schema validation, particle evaluation
    overlap = {
        k for k in runner_capabilities
        if runner_capabilities[k]
    } & {
        k for k in wl_capabilities
        if wl_capabilities[k]
    }

    # This is a documentation assertion, not a pass/fail gate.
    # The fact of overlap is the finding for Task 4.2.
    assert overlap, (
        "No overlapping capabilities found between runner and wave_loop — "
        "unexpected: the spec identifies a duplicate evaluator risk"
    )

    # Key finding: both modules can independently evaluate a complete
    # request from schema to surface to action.  This is the "two
    # unlabelled implementations" risk documented in design.md.
    print(f"\n[Task 4.2 finding] Overlapping evaluator capabilities: {sorted(overlap)}")
    print(f"[Task 4.2 finding] Runner-only capabilities: "
          f"{sorted(set(runner_capabilities) - set(wl_capabilities))}")
    print(f"[Task 4.2 finding] Wave-loop-only capabilities: "
          f"{sorted(set(wl_capabilities) - set(runner_capabilities))}")


def test_adversarial_dirty_file_audit():
    """Guard for Task 4.2: enumerate dirty/non-owned files in the diff
    that are not explicitly owned by the hardening change.

    This test reads git status and flags any modified file that is NOT
    in the owned-files list from the hardening tasks.
    """
    import subprocess

    # Files owned by the hardening change per tasks.md
    owned_patterns = [
        "contracts/wave_mvp_scenario_matrix.json",
        "schemas/wave-mvp-completion.schema.json",
        "scripts/run_joint_wave_surface_mvp.py",
        "scripts/validate_wave_mvp_completion.py",
        "tests/test_joint_wave_surface_mvp.py",
        "tests/test_wave_contract_hardening.py",
        "openspec/changes/harden-contract-driven-wave-mvp/",
    ]

    result = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
    )
    modified = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    unowned = []
    for path in modified:
        if not any(path.startswith(p) or p in path for p in owned_patterns):
            # Allow our own adversarial review files
            if "adversarial" in path:
                continue
            unowned.append(path)

    if unowned:
        # This is a documentation finding, not necessarily a failure.
        # But if the hardening diff touches files it shouldn't, flag it.
        print(f"\n[Task 4.2 finding] Un-owned modified files in diff: {unowned}")
