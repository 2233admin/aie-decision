"""Command-line surface for the standalone evidence compiler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .exporting import human_report, package_json, to_primitive, write_human_report, write_package_json
from .fermi import estimate_fermi
from .intervals import ForecastInterval, IntervalKind, audit_interval
from .pipeline import compile_analysis
from .validation import ContractValidationError, load_and_validate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aie-decision", description="Standalone answer-oriented evidence compiler")
    commands = parser.add_subparsers(dest="command", required=True)

    compile_command = commands.add_parser("compile", help="Compile a supplied question and materials into an analysis package")
    compile_command.add_argument("path", type=Path, help="JSON input; no external acquisition is performed")
    compile_command.add_argument("--output-dir", type=Path, required=True)

    fermi = commands.add_parser(
        "fermi",
        help="Run the legacy deterministic interval engine with supplied variables",
    )
    fermi.add_argument("path", type=Path, help="JSON question, formula, and variable ranges")

    search = commands.add_parser(
        "search-fermi",
        help="Run the experimental bounded Fermi candidate loop",
    )
    search.add_argument("path", type=Path, help="JSON question, variables, candidate graph, and search budget")
    search.add_argument(
        "--experimental",
        action="store_true",
        help="Acknowledge that the loop is an uncalibrated experimental surface",
    )
    validate = commands.add_parser("validate", help="Validate a versioned AIE JSON object")
    validate.add_argument("path", type=Path)
    validate.add_argument("--kind", help="Schema kind when it cannot be inferred")

    report = commands.add_parser("render-report", help="Render a human report from a machine package")
    report.add_argument("path", type=Path)
    report.add_argument("--output", type=Path)

    audit = commands.add_parser("audit-interval", help="Audit one declared future-value interval")
    audit.add_argument("--target", required=True)
    audit.add_argument("--horizon", required=True)
    audit.add_argument("--unit", required=True)
    audit.add_argument("--population", required=True)
    audit.add_argument("--coverage", required=True, type=float)
    audit.add_argument("--lower", required=True, type=float)
    audit.add_argument("--upper", required=True, type=float)
    audit.add_argument("--reference", required=True, type=float)
    audit.add_argument("--reference-time", required=True)
    audit.add_argument("--method", required=True)
    audit.add_argument("--baseline-lower", type=float)
    audit.add_argument("--baseline-upper", type=float)
    audit.add_argument("--threshold", action="append", type=float, default=[])
    return parser


def _compile(args: argparse.Namespace) -> int:
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("compiler input must be a JSON object")
    result = compile_analysis(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package_path = write_package_json(result.package, args.output_dir / "analysis-package.json")
    report_path = write_human_report(result.package, args.output_dir / "decision-report.md")
    ledger_path = args.output_dir / "analysis-ledger.json"
    ledger_path.write_text(package_json(result.ledger.export()), encoding="utf-8", newline="\n")
    response = {
        "status": result.package.package_state,
        "package": str(package_path),
        "report": str(report_path),
        "ledger": str(ledger_path),
        "validation_issues": [to_primitive(issue) for issue in result.validation_issues],
    }
    print(package_json(response), end="")
    return 0 if not result.validation_issues else 2


def _fermi(args: argparse.Namespace) -> int:
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    result = estimate_fermi(payload)
    print(package_json(result), end="")
    return 0


def _search_fermi(args: argparse.Namespace) -> int:
    if not args.experimental:
        raise ValueError("search-fermi requires --experimental")
    from .search import search_fermi

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    result = search_fermi(payload)
    print(package_json(result), end="")
    return 0 if result["status"] == "result-found" else 2
def _validate(args: argparse.Namespace) -> int:
    document, issues = load_and_validate(args.path, args.kind, raise_on_error=False)
    result = {
        "valid": not issues,
        "kind": args.kind or document.get("kind") or document.get("schema_kind"),
        "issues": [to_primitive(issue) for issue in issues],
    }
    print(package_json(result), end="")
    return 0 if not issues else 2


def _render_report(args: argparse.Namespace) -> int:
    package = json.loads(args.path.read_text(encoding="utf-8"))
    report = human_report(package)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
    else:
        print(report, end="")
    return 0


def _audit_interval(args: argparse.Namespace) -> int:
    interval = ForecastInterval(
        target=args.target,
        horizon=args.horizon,
        unit=args.unit,
        population=args.population,
        coverage_level=args.coverage,
        conditional_assumptions=(),
        generation_method=args.method,
        reference_time=args.reference_time,
        lower=args.lower,
        upper=args.upper,
        kind=IntervalKind.PREDICTION,
    )
    baseline = None
    if (args.baseline_lower is None) != (args.baseline_upper is None):
        raise ValueError("baseline lower and upper must be provided together")
    if args.baseline_lower is not None:
        baseline = ForecastInterval(
            target=args.target,
            horizon=args.horizon,
            unit=args.unit,
            population=args.population,
            coverage_level=args.coverage,
            conditional_assumptions=(),
            generation_method="declared_baseline",
            reference_time=args.reference_time,
            lower=args.baseline_lower,
            upper=args.baseline_upper,
            kind=IntervalKind.PREDICTION,
        )
    audit = audit_interval(
        interval,
        scale=args.reference,
        baseline_width=(baseline.upper - baseline.lower) if baseline else None,
        thresholds=args.threshold,
        baseline=baseline,
    )
    result = to_primitive(audit)
    result["status"] = "uncalibrated_informative" if audit.informative else "uncalibrated_uninformative"
    result["empirical_coverage"] = None
    result["calibration"] = "unmeasured"
    print(package_json(result), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            return _compile(args)
        if args.command == "fermi":
            return _fermi(args)
        if args.command == "search-fermi":
            return _search_fermi(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "render-report":
            return _render_report(args)
        if args.command == "audit-interval":
            return _audit_interval(args)
        parser.error(f"unsupported command: {args.command}")
    except (ContractValidationError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
