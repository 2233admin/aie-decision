"""Versioned JSON command-line interface for the agent decomposition runtime.

The CLI is intentionally simple.  Every subcommand reads JSON from a
file (or ``-`` for stdin) and writes JSON to a file (or stdout).  The
protocol is identical to the Python API, so the same payloads round-trip
through a PowerShell process boundary without code changes.

Subcommands:

* ``discover``  — list action specs, legal next actions, and budget status.
* ``start``     — initialise a new session from a raw question.
* ``apply``     — apply one action to an existing session.
* ``inspect``   — read the current projected state of a session.
* ``finalize``  — request a frontier evaluation (delegated to the kernel).
* ``replay``    — replay the trajectory through the kernel and verify digests.

The CLI accepts an injected kernel through a Python entry point (see
``build_default_kernel``) but does NOT embed any model provider, prompt
template, or research subsystem.  Integration adapters are responsible
for wiring in Track A and Track B kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .agent_runtime import (
    AgentRuntime,
    BudgetPolicy,
    KernelProtocol,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from .trajectory import Trajectory


# ---------------------------------------------------------------------------
# Default kernel for CLI usage — pure, no model provider
# ---------------------------------------------------------------------------


def build_default_kernel() -> KernelProtocol:
    """Return a deterministic, model-free kernel suitable for CLI smoke tests.

    The default kernel treats the state as an append-only journal of
    ``recording`` entries.  It validates field shapes, executes pure
    inserts, and refuses to certify the frontier.  This lets the CLI
    run end-to-end without embedding any model provider or prompt — the
    track only exposes the contract.
    """

    class _DefaultKernel:
        def initial_state(self, question: str) -> dict[str, Any]:
            return {
                "question": question,
                "recordings": [],
                "depth": 0,
                "frontier": [{"id": "root", "label": question}],
            }

        def action_specs(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "expand",
                    "category": "structural",
                    "required_fields": ["node_id", "children"],
                    "produces_revision": True,
                },
                {
                    "name": "estimate",
                    "category": "measurement",
                    "required_fields": ["node_id", "value", "unit"],
                    "produces_revision": True,
                },
                {
                    "name": "rollback",
                    "category": "control",
                    "required_fields": ["target_sequence"],
                    "produces_revision": True,
                },
                {
                    "name": "finalize",
                    "category": "frontier",
                    "required_fields": [],
                    "produces_revision": False,
                },
            ]

        def validate(
            self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]
        ) -> list[dict[str, Any]]:
            issues: list[dict[str, Any]] = []
            if not isinstance(payload, Mapping):
                return [{"code": "bad_payload", "path": "$", "message": "payload must be an object"}]
            if action == "expand":
                if "node_id" not in payload:
                    issues.append({"code": "missing_field", "path": "$.node_id", "message": "node_id is required"})
                children = payload.get("children")
                if not isinstance(children, list) or not children:
                    issues.append({"code": "missing_field", "path": "$.children", "message": "children must be a non-empty list"})
            elif action == "estimate":
                for field_name in ("node_id", "value", "unit"):
                    if field_name not in payload:
                        issues.append({"code": "missing_field", "path": f"$.{field_name}", "message": f"{field_name} is required"})
            elif action == "rollback":
                if "target_sequence" not in payload:
                    issues.append({"code": "missing_field", "path": "$.target_sequence", "message": "target_sequence is required"})
            return issues

        def execute(
            self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]
        ) -> dict[str, Any]:
            new_state = {key: value for key, value in state.items()}
            recordings = list(new_state.get("recordings", []))
            depth = int(new_state.get("depth", 0))
            frontier = list(new_state.get("frontier", []))
            if action == "expand":
                node_id = str(payload["node_id"])
                children = list(payload["children"])
                recordings.append({"action": "expand", "node_id": node_id, "children": children})
                new_state["depth"] = depth + 1
                new_state["frontier"] = [
                    {"id": child.get("id", f"{node_id}::{index}"), "label": child.get("label", "")}
                    for index, child in enumerate(children)
                    if isinstance(child, Mapping)
                ]
            elif action == "estimate":
                recordings.append({
                    "action": "estimate",
                    "node_id": str(payload["node_id"]),
                    "value": payload["value"],
                    "unit": str(payload["unit"]),
                })
                new_state["frontier"] = [
                    node for node in frontier if node.get("id") != payload.get("node_id")
                ]
            elif action == "rollback":
                recordings.append({"action": "rollback", "target_sequence": payload.get("target_sequence")})
            elif action == "finalize":
                recordings.append({"action": "finalize"})
            new_state["recordings"] = recordings
            return new_state

        def legal_next_actions(self, state: Mapping[str, Any]) -> list[str]:
            frontier = state.get("frontier", [])
            if not frontier:
                return ["finalize"]
            return ["expand", "estimate", "finalize"]

        def active_frontier(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
            return list(state.get("frontier", []))

        def evaluate_frontier(self, state: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "status": "insufficient",
                "reasons": ["default kernel never certifies; supply an injected kernel for production runs"],
                "blocking_issues": [],
            }

    return _DefaultKernel()


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def _load_session_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("session document must be a JSON object")
    return dict(payload)


def _write_session_document(path: Path | None, document: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rehydrate(document: Mapping[str, Any], kernel: KernelProtocol) -> AgentRuntime:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {document.get('schema_version')!r}")
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol_version: {document.get('protocol_version')!r}")
    trajectory_payload = document.get("trajectory")
    if not isinstance(trajectory_payload, Mapping):
        raise ValueError("session document is missing trajectory")
    trajectory = Trajectory.from_export(trajectory_payload)
    state = document.get("state") or {}
    if not isinstance(state, Mapping):
        raise ValueError("session state must be an object")
    budgets = BudgetPolicy.from_dict(document.get("budget_policy") or {})
    counters_doc = document.get("budget_counters") or {}
    from .agent_runtime import BudgetCounters, SessionStatus

    counters = BudgetCounters(
        actions=int(counters_doc.get("actions", 0)),
        evaluations=int(counters_doc.get("evaluations", 0)),
        compute=int(counters_doc.get("compute", 0)),
        depth=int(counters_doc.get("depth", 0)),
    )
    status_value = str(document.get("status", "active"))
    runtime = AgentRuntime(
        session_id=str(document.get("session_id") or ""),
        kernel=kernel,
        trajectory=trajectory,
        state=dict(state),
        question=str(document.get("question") or ""),
        budgets=budgets,
        counters=counters,
        status=SessionStatus(status_value),
        frontier_evaluation=dict(document["frontier_evaluation"]) if isinstance(document.get("frontier_evaluation"), Mapping) else None,
        created_at=str(document.get("created_at") or ""),
        updated_at=str(document.get("updated_at") or ""),
        metadata=dict(document.get("metadata") or {}),
    )
    return runtime


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _read_input(path: str) -> dict[str, Any]:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("input must be a JSON object")
    return dict(payload)


def _output(result: Mapping[str, Any], output: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None or output == "-":
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        Path(output).write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aie-decision",
        description="Versioned JSON CLI for the agent decomposition runtime",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--output", help="path to write JSON output; '-' for stdout (default)")
        sub.add_argument("--pretty", action="store_true", help="pretty-print JSON to stdout regardless of --output")

    discover = commands.add_parser("discover", help="list action specs and legal next actions")
    add_common(discover)

    start = commands.add_parser("start", help="start a new session from a raw question")
    add_common(start)
    start.add_argument("--session-id", required=True, help="stable session identifier")
    start.add_argument("--question", required=True, help="raw quantitative question")
    start.add_argument("--input", help="JSON file with extra metadata and budgets")
    start.add_argument("--session", dest="session_path", help="path to write the new session document")

    apply = commands.add_parser("apply", help="apply one action to a session")
    add_common(apply)
    apply.add_argument("--session", dest="session_path", required=True, help="path to an existing session document")
    apply.add_argument("--input", help="JSON file with the action payload")
    apply.add_argument("--action", help="action name (overrides the payload's 'action' field)")

    inspect = commands.add_parser("inspect", help="read the current projected state of a session")
    add_common(inspect)
    inspect.add_argument("--session", dest="session_path", required=True, help="path to an existing session document")

    finalize = commands.add_parser("finalize", help="request a frontier evaluation")
    add_common(finalize)
    finalize.add_argument("--session", dest="session_path", required=True, help="path to an existing session document")

    replay = commands.add_parser("replay", help="replay the trajectory through the kernel")
    add_common(replay)
    replay.add_argument("--session", dest="session_path", required=True, help="path to an existing session document")

    return parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def _kernel_from_env() -> KernelProtocol:
    """Resolve a kernel based on the ``AIE_AGENT_KERNEL`` environment variable.

    If the variable is unset, the default model-free kernel is used.  This
    keeps the CLI callable from PowerShell without embedding any model
    provider; production callers wire in their own kernel via the Python
    API or by setting the variable to a fully-qualified factory path.
    """

    factory_path = os.environ.get("AIE_AGENT_KERNEL")
    if not factory_path:
        return build_default_kernel()
    module_name, _, attr = factory_path.partition(":")
    if not attr:
        attr = "build"
    module = __import__(module_name, fromlist=[attr])
    factory = getattr(module, attr)
    return factory()


def _do_discover(args: argparse.Namespace) -> int:
    kernel = _kernel_from_env()
    runtime = AgentRuntime(
        session_id="__discover__",
        kernel=kernel,
        trajectory=Trajectory("__discover__"),
        state=kernel.initial_state(""),
    )
    payload = runtime.discover()
    _output(payload, args.output if not args.pretty else "-")
    return 0


def _do_start(args: argparse.Namespace) -> int:
    if not args.question:
        raise ValueError("start requires --question")
    extra = _read_input(args.input) if args.input else {}
    metadata = dict(extra.get("metadata") or {})
    budgets_doc = extra.get("budgets") or {}
    budgets = BudgetPolicy.from_dict(budgets_doc) if isinstance(budgets_doc, Mapping) else BudgetPolicy()
    kernel = _kernel_from_env()
    runtime = AgentRuntime.start(
        session_id=args.session_id,
        question=args.question,
        kernel=kernel,
        budgets=budgets,
        metadata=metadata,
    )
    document = runtime.export()
    session_path = _write_session_document(Path(args.session_path) if args.session_path else None, document)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "session_id": args.session_id,
        "session_path": str(session_path) if session_path else None,
        "inspect": runtime.inspect(),
    }
    _output(payload, args.output if not args.pretty else "-")
    return 0


def _do_apply(args: argparse.Namespace) -> int:
    document = _load_session_document(Path(args.session_path))
    kernel = _kernel_from_env()
    runtime = _rehydrate(document, kernel)
    extra = _read_input(args.input) if args.input else {}
    action = args.action or extra.get("action")
    if not action:
        # Report the missing-action error as a structured JSON response
        # on stdout so callers (Python or PowerShell) only have to read
        # one channel.  No event is appended to the trajectory.
        _output(
            {
                "error": {
                    "code": "missing_field",
                    "message": "apply requires --action or an 'action' field in --input",
                    "issues": [
                        {
                            "code": "missing_field",
                            "path": "$.action",
                            "message": "action name is required",
                        }
                    ],
                }
            },
            args.output if not args.pretty else "-",
        )
        return 2
    # Accept both protocol envelopes and the natural flat CLI form:
    # {"action": "expand", "payload": {...}} or
    # {"action": "expand", ...payload fields...}.  ``--action`` may also
    # supply the name while the input file contains either payload shape.
    if isinstance(extra.get("payload"), Mapping):
        payload = dict(extra["payload"])
    else:
        control_fields = {
            "action",
            "prior_revision",
            "rollback_target_sequence",
            "compute_cost",
        }
        payload = {
            key: value for key, value in extra.items() if key not in control_fields
        }
    prior_revision = extra.get("prior_revision")
    rollback_target = extra.get("rollback_target_sequence")
    compute_cost = extra.get("compute_cost")
    result = runtime.apply(
        action=action,
        payload=payload,
        prior_revision=prior_revision,
        rollback_target_sequence=rollback_target,
        compute_cost=compute_cost,
    )
    new_document = runtime.export()
    _write_session_document(Path(args.session_path), new_document)
    _output(result.to_dict(), args.output if not args.pretty else "-")
    return 0 if result.accepted else 2


def _do_inspect(args: argparse.Namespace) -> int:
    document = _load_session_document(Path(args.session_path))
    kernel = _kernel_from_env()
    runtime = _rehydrate(document, kernel)
    payload = runtime.inspect()
    _output(payload, args.output if not args.pretty else "-")
    return 0


def _do_finalize(args: argparse.Namespace) -> int:
    document = _load_session_document(Path(args.session_path))
    kernel = _kernel_from_env()
    runtime = _rehydrate(document, kernel)
    verdict = runtime.finalize()
    new_document = runtime.export()
    _write_session_document(Path(args.session_path), new_document)
    _output(verdict, args.output if not args.pretty else "-")
    return 0 if verdict["status"] == "certified" else 2


def _do_replay(args: argparse.Namespace) -> int:
    document = _load_session_document(Path(args.session_path))
    kernel = _kernel_from_env()
    runtime = _rehydrate(document, kernel)
    verdict = runtime.replay()
    _output(verdict, args.output if not args.pretty else "-")
    return 0 if verdict["verdict"] == "match" else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            return _do_discover(args)
        if args.command == "start":
            return _do_start(args)
        if args.command == "apply":
            return _do_apply(args)
        if args.command == "inspect":
            return _do_inspect(args)
        if args.command == "finalize":
            return _do_finalize(args)
        if args.command == "replay":
            return _do_replay(args)
        parser.error(f"unsupported command: {args.command}")
    except (ValueError, json.JSONDecodeError) as exc:
        # Errors are surfaced on BOTH stdout and stderr as a single JSON
        # document.  Stdout lets callers parse the protocol uniformly;
        # stderr preserves the original logging channel.
        document = json.dumps(
            {"error": type(exc).__name__, "message": str(exc)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        sys.stdout.write(document)
        sys.stdout.flush()
        sys.stderr.write(document)
        sys.stderr.flush()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
