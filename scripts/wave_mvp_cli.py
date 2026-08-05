"""MVP CLI: authority, replay, and argument parsing."""

from __future__ import annotations
import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from wave_mvp_models import _payload_hash, _sanitize_for_json
# CLI.
# ---------------------------------------------------------------------------


# ---- injected by run_joint_wave_surface_mvp.py (avoids circular import) ----
_RUNNER = None  # set to run_mvp
_AUTHORITY = None  # set to _get_authority_module()
_SCHEMA_VERSION = "joint-wave-surface-mvp.v1"
_INTEGRATION_ERROR: type[Exception] = RuntimeError
_ORACLE_LABEL = "non_authoritative_oracle"

def _get_runner():
    if _RUNNER is None:
        raise RuntimeError("runner not injected — call set_runner_deps() first")
    return _RUNNER

def _get_authority():
    if _AUTHORITY is None:
        raise RuntimeError("authority not injected — call set_runner_deps() first")
    return _AUTHORITY()

def set_runner_deps(
    runner, authority, schema_version=None, *, integration_error=RuntimeError,
    oracle_label="non_authoritative_oracle",
):
    global _RUNNER, _AUTHORITY, _SCHEMA_VERSION, _INTEGRATION_ERROR, _ORACLE_LABEL
    _RUNNER = runner
    _AUTHORITY = authority
    if schema_version is not None:
        _SCHEMA_VERSION = schema_version
    _INTEGRATION_ERROR = integration_error
    _ORACLE_LABEL = oracle_label

def _load_payload(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    if document.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses unsupported schema_version {document.get('schema_version')!r}"
        )
    return document


def _ledger_is_authoritative(ledger: Mapping[str, Any]) -> bool:
    """Return True when the ledger was produced by the authoritative package evaluator.

    Authority-ledger entries carry stable_id (from AnalysisLedger.export());
    oracle-ledger entries carry event_id without stable_id.
    """
    entries = ledger.get("entries", ())
    if not isinstance(entries, (list, tuple)) or not entries:
        return False
    first_entry = entries[0]
    if not isinstance(first_entry, Mapping):
        return False
    return "stable_id" in first_entry and "event_id" not in first_entry


def _normalize_ledger_for_comparison(ledger: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with non-deterministic fields removed so two runs compare equal.

    AnalysisLedger.export() includes recorded_at with a wall-clock
    timestamp that differs across invocations.  This normaliser strips those
    fields so the deterministic content of two authority runs can be compared
    without weakening the semantic invariants (payload hashes, sequences,
    state transitions) that replay_wave_ledger already enforces.
    """
    import copy

    normalized: dict[str, Any] = copy.deepcopy(ledger)
    for entry in normalized.get("entries", ()):
        if isinstance(entry, dict):
            entry.pop("recorded_at", None)
    return normalized


def _run_authoritative_and_compare(
    payload: Mapping[str, Any], ledger: Mapping[str, Any],
    authority: Any
) -> dict[str, Any]:
    """Run the authoritative evaluator twice and verify the ledger matches.

    Returns the re-computed ledger on success; returns ledger_mismatch or
    non_deterministic status dicts on failure.

    The ledger_hash is computed from the supplied ledger after canonical
    normalization (recorded_at stripped) so that two CLI replay invocations
    on the same ledger are byte-for-byte identical.
    """
    first_result = authority["run"](payload)
    first_ledger = _sanitize_for_json(first_result.to_dict()["ledger"])
    if _normalize_ledger_for_comparison(first_ledger) != _normalize_ledger_for_comparison(ledger):
        return {
            "status": "ledger_mismatch",
            "ledger_hash_first": _payload_hash(first_ledger),
            "ledger_hash_supplied": _payload_hash(ledger),
        }
    second_result = authority["run"](payload)
    second_ledger = _sanitize_for_json(second_result.to_dict()["ledger"])
    if _normalize_ledger_for_comparison(second_ledger) != _normalize_ledger_for_comparison(first_ledger):
        return {
            "status": "non_deterministic",
            "iterations": [],
        }

    # Build honest iterations from the replay events on the fresh ledger.
    # The replay result is deterministic because it depends only on
    # payload-level fields (hashes, sequences, states) that were already
    # validated as stable across runs.
    replay_events = first_result.replay.get("events", ())
    round_indices = sorted(set(e["round_index"] for e in replay_events if isinstance(e, dict) and "round_index" in e))

    # Compute ledger_hash from the supplied ledger after normalization.
    # The comparison above proved normalized(first_ledger) == normalized(ledger),
    # so the supplied ledger's normalized hash is the canonical output hash.
    canonical = _normalize_ledger_for_comparison(ledger)
    canonical_hash = _payload_hash(canonical)

    return {
        "status": "ok",
        "run_id": first_result.run_id,
        "iterations": round_indices,
        "final_status": "result-found" if first_result.decision_value.get("accepted") else "insufficient-information",
        "ledger_hash": canonical_hash,
    }


def _run_replay(ledger_path: Path, fixture_path: Path | None = None) -> dict[str, Any]:
    document = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("replay document must be a JSON object")
    # Accept either a bare ledger document or a wrapped {ledger, fixture} object.
    if "ledger" in document and isinstance(document["ledger"], Mapping):
        ledger = document["ledger"]
        payload = document.get("fixture")
    else:
        ledger = document
        payload = None
    if not isinstance(ledger, Mapping) or "entries" not in ledger:
        raise ValueError("replay document must contain a ledger")
    if payload is None and fixture_path is not None:
        payload = _load_payload(fixture_path)
    if payload is None:
        raise ValueError(
            "replay of a bare ledger requires a --fixture argument or "
            "a wrapped {ledger, fixture} document"
        )
    payload = dict(payload)
    payload.setdefault("run_id", ledger.get("run_id", "wave-mvp-replay"))
    payload["particles"] = dict(payload.get("particles", {}))
    payload["particles"].setdefault("count", 1)
    payload["particles"].setdefault("seed", 0)

    # Route to the evaluator that produced the ledger.
    if _ledger_is_authoritative(ledger):
        try:
            return _run_authoritative_and_compare(payload, ledger, _get_authority())
        except Exception as exc:
            return {
                "status": "ledger_mismatch",
                "error": str(exc),
            }

    # Oracle path (existing behavior).
    first = _get_runner()(payload)
    if first.ledger != ledger:
        return {
            "status": "ledger_mismatch",
            "ledger_hash_first": _payload_hash(first.ledger),
            "ledger_hash_supplied": _payload_hash(ledger),
        }
    second = _get_runner()(payload)
    if second.ledger != first.ledger:
        return {
            "status": "non_deterministic",
            "iterations": [iteration.round_index for iteration in first.iterations],
        }
    return {
        "status": "ok",
        "run_id": first.run_id,
        "iterations": [iteration.round_index for iteration in first.iterations],
        "final_status": first.status,
        "ledger_hash": _payload_hash(first.ledger),
    }
def _run_authoritative(fixture: Path, output_dir: Path) -> int:
    """Run through the authoritative package evaluator and write evidence."""
    authority = _get_authority()
    payload = _load_payload(fixture)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = authority["run"](payload)
    except authority["error_type"] as exc:
        print(json.dumps({"status": "authority_error", "error": str(exc)}))
        return 2
    ledger_path = output_dir / "wave-ledger.json"
    summary_path = output_dir / "wave-summary.json"
    provenance_path = output_dir / "wave-provenance.json"
    result_dict = result.to_dict()
    safe_result = _sanitize_for_json(result_dict)
    safe_provenance = _sanitize_for_json(result.provenance.to_dict())
    summary = {
        "run_id": result.run_id,
        "evaluator_label": authority["label"],
        "evaluator_path": "package",
        "authority_version": result.provenance.authority_version,
        "all_components_called": result.provenance.all_components_called(),
        "failed_components": list(result.provenance.failed_components()),
        "component_count": len(result.provenance.components),
        "surface": {
            "kind": result.surface.get("kind", ""),
            "coverage_semantics": result.surface.get("coverage_semantics", ""),
            "calibration_basis": result.surface.get("calibration_basis", ""),
            "axis_names": result.surface.get("axis_names", []),
            "particle_count": result.surface.get("particle_count", 0),
            "seed": result.surface.get("seed"),
        },
        "diagnostics": {
            "surface_kind": result.diagnostics.get("surface_kind", ""),
            "calibration_basis": result.diagnostics.get("calibration_basis", ""),
            "multimodal_axes": result.diagnostics.get("multimodal_axes", []),
            "particle_count": result.diagnostics.get("particle_count", 0),
        },
        "actions": [
            {
                "action_kind": a.get("action_kind", a.get("kind", "")),
                "rationale": a.get("rationale", ""),
                "affected_entities": a.get("affected_entities", []),
            }
            for a in result.actions
        ],
        "ledger": {
            "entry_count": len(result.ledger.get("entries", ())),
            "schema_version": result.ledger.get("schema_version", ""),
        },
        "replay": {
            "accepted": result.replay.get("accepted"),
            "event_count": result.replay.get("event_count", 0),
            "current_state": result.replay.get("current_state"),
        },
    }
    safe_summary = _sanitize_for_json(summary)
    summary_path.write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(_sanitize_for_json(result.ledger), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    provenance_path.write_text(
        json.dumps(safe_provenance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    status = "result-found" if result.decision_value.get("accepted") else "insufficient-information"
    print(
        json.dumps(
            {
                "status": status,
                "evaluator": authority["label"],
                "ledger": str(ledger_path),
                "summary": str(summary_path),
                "provenance": str(provenance_path),
                "components_called": result.provenance.all_components_called(),
            }
        )
    )
    # Fail closed: return non-zero when any required component was not
    # called or when the ledger is empty.
    if not result.provenance.all_components_called():
        return 3
    if not result.ledger.get("entries"):
        return 4
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a golden fixture and write evidence")
    run_parser.add_argument("fixture", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--authority",
        choices=("package", "oracle"),
        default="oracle",
        help=(
            "Evaluator path: 'package' routes through the authoritative "
            "package evaluator (aie_decision.wave_authority); 'oracle' uses "
            "the non-authoritative script evaluator for backward compatibility."
        ),
    )

    authority_parser = subparsers.add_parser(
        "authority",
        help="Run through the authoritative package evaluator explicitly",
    )
    authority_parser.add_argument("fixture", type=Path)
    authority_parser.add_argument("--output-dir", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay", help="Replay a saved ledger against its fixture")
    replay_parser.add_argument("ledger", type=Path)
    replay_parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional fixture; required when replaying a bare ledger document",
    )

    args = parser.parse_args(argv)
    if args.command == "authority":
        return _run_authoritative(args.fixture, args.output_dir)
    if args.command == "run":
        if args.authority == "package":
            return _run_authoritative(args.fixture, args.output_dir)
        # Oracle path: existing script evaluator, explicitly labeled non-authoritative.
        payload = _load_payload(args.fixture)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _get_runner()(payload)
        except _INTEGRATION_ERROR as exc:
            print(json.dumps({"status": getattr(_INTEGRATION_ERROR, "code", "integration_unavailable"), "error": str(exc), "evaluator": _ORACLE_LABEL}))
            return 2
        ledger_path = args.output_dir / "wave-ledger.json"
        summary_path = args.output_dir / "wave-summary.json"
        safe_result = result.to_json_safe()
        safe_summary = _sanitize_for_json(result.summary)
        safe_ledger = _sanitize_for_json(result.ledger)
        summary_path.write_text(
            json.dumps(safe_summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        ledger_path.write_text(
            json.dumps(safe_ledger, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "evaluator": _ORACLE_LABEL,
                    "ledger": str(ledger_path),
                    "summary": str(summary_path),
                    "iterations": [iteration.round_index for iteration in result.iterations],
                }
            )
        )
        return 0 if result.status in {"result-found", "budget-exhausted"} else 2
    if args.command == "replay":
        outcome = _run_replay(args.ledger, args.fixture)
        print(json.dumps(_sanitize_for_json(outcome)))
        return 0 if outcome["status"] == "ok" else 2
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
