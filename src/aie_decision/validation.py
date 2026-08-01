"""Deterministic validators for AIE v1 boundary records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import SCHEMA_VERSION, AnalysisPackage, ConditionGraph, to_dict

_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-_:][a-z0-9]+)*$")
_TERMINAL_STATES = {
    "answerable_bounded",
    "not_answerable",
    "insufficient_evidence",
    "invalid_contract",
    "failed_validation",
}
_FACT_TYPES = {"observed_event", "primary_record"}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str


class ContractValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.code} at {i.path}: {i.message}" for i in self.issues))


def _issue(issues: list[ValidationIssue], code: str, path: str, message: str) -> None:
    issues.append(ValidationIssue(code, path, message))


def _required(doc: Mapping[str, Any], names: Iterable[str], issues: list[ValidationIssue], path: str = "$") -> None:
    for name in names:
        if name not in doc or doc[name] is None or doc[name] == "":
            _issue(issues, "required_field", f"{path}.{name}", "field is required")


def _schema_version(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    if doc.get("schema_version") != SCHEMA_VERSION:
        _issue(issues, "schema_version", "$.schema_version", f"expected {SCHEMA_VERSION}")


def _stable_id(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _issue(issues, "unstable_identifier", path, "must be a lowercase stable identifier")


def _revision(doc: Mapping[str, Any], issues: list[ValidationIssue], path: str = "$") -> None:
    rev = doc.get("revision")
    if not isinstance(rev, Mapping):
        _issue(issues, "revision_missing", f"{path}.revision", "revision metadata is required")
        return
    _required(rev, ("revision_id", "sequence", "created_at"), issues, f"{path}.revision")
    _stable_id(rev.get("revision_id"), f"{path}.revision.revision_id", issues)
    sequence = rev.get("sequence")
    predecessor = rev.get("supersedes_revision_id")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        _issue(issues, "revision_sequence", f"{path}.revision.sequence", "must be a positive integer")
    elif sequence == 1 and predecessor:
        _issue(issues, "revision_lineage", f"{path}.revision.supersedes_revision_id", "first revision cannot supersede another")
    elif sequence > 1 and not predecessor:
        _issue(issues, "revision_lineage", f"{path}.revision.supersedes_revision_id", "later revision must name its predecessor")
    if predecessor:
        _stable_id(predecessor, f"{path}.revision.supersedes_revision_id", issues)


def _provenance(items: Any, issues: list[ValidationIssue], path: str, required: bool = True) -> None:
    if not isinstance(items, list) or (required and not items):
        _issue(issues, "provenance_incomplete", path, "at least one provenance reference is required")
        return
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            _issue(issues, "provenance_incomplete", item_path, "must be an object")
            continue
        _required(item, ("source_id", "locator"), issues, item_path)
        _stable_id(item.get("source_id"), f"{item_path}.source_id", issues)


def _validate_answer_contract(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("question_id", "question", "answer_type", "target", "observation_cutoff"), issues)
    _stable_id(doc.get("question_id"), "$.question_id", issues)
    if doc.get("answer_type") not in {
        "current_observation", "historical_reconstruction", "future_prediction",
        "causal_explanation", "decision_comparison",
    }:
        _issue(issues, "answer_type", "$.answer_type", "unknown answer type")
    target = doc.get("target")
    if isinstance(target, Mapping):
        _required(target, ("entity", "measure", "unit"), issues, "$.target")
    else:
        _issue(issues, "required_field", "$.target", "must be an object")
    coverage = doc.get("requested_coverage")
    if coverage is not None and (not isinstance(coverage, (int, float)) or not 0 < coverage < 1):
        _issue(issues, "coverage_range", "$.requested_coverage", "must be between zero and one")
    if doc.get("answer_type") == "future_prediction" and not doc.get("prediction_horizon"):
        _issue(issues, "incomplete_answer_contract", "$.prediction_horizon", "future prediction requires a horizon")
    if doc.get("answer_type") == "future_prediction" and not doc.get("uncertainty_semantics"):
        _issue(issues, "incomplete_answer_contract", "$.uncertainty_semantics", "future prediction requires declared uncertainty semantics")
    _revision(doc, issues)


def _validate_graph(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("graph_id", "question_id", "answer_contract_revision_id", "conditions", "edges"), issues)
    _stable_id(doc.get("graph_id"), "$.graph_id", issues)
    _stable_id(doc.get("question_id"), "$.question_id", issues)
    nodes = doc.get("conditions", [])
    edges = doc.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        _issue(issues, "graph_shape", "$", "conditions and edges must be arrays")
        return
    node_ids: list[str] = []
    for index, node in enumerate(nodes):
        path = f"$.conditions[{index}]"
        if not isinstance(node, Mapping):
            _issue(issues, "graph_shape", path, "node must be an object")
            continue
        _required(node, ("condition_id", "name", "value_type", "necessity", "status", "answer_impact"), issues, path)
        _stable_id(node.get("condition_id"), f"{path}.condition_id", issues)
        if not node.get("answer_impact"):
            _issue(issues, "answer_impact_missing", f"{path}.answer_impact", "node must state how it changes the answer")
        node_ids.append(node.get("condition_id"))
        _revision(node, issues, path)
        _provenance(node.get("provenance"), issues, f"{path}.provenance")
    if len(node_ids) != len(set(node_ids)):
        _issue(issues, "duplicate_identifier", "$.conditions", "condition identifiers must be unique")
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids if isinstance(node_id, str)}
    edge_ids: list[str] = []
    for index, edge in enumerate(edges):
        path = f"$.edges[{index}]"
        if not isinstance(edge, Mapping):
            _issue(issues, "graph_shape", path, "edge must be an object")
            continue
        _required(edge, ("edge_id", "from_id", "to_id", "relation", "evidence_status", "answer_impact"), issues, path)
        _stable_id(edge.get("edge_id"), f"{path}.edge_id", issues)
        edge_ids.append(edge.get("edge_id"))
        source, target = edge.get("from_id"), edge.get("to_id")
        if source not in adjacency:
            _issue(issues, "dangling_edge", f"{path}.from_id", "source node does not exist")
        if target not in adjacency and not (isinstance(target, str) and target.startswith("target:")):
            _issue(issues, "dangling_edge", f"{path}.to_id", "target node does not exist")
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)
        _revision(edge, issues, path)
        _provenance(edge.get("provenance"), issues, f"{path}.provenance")
    if len(edge_ids) != len(set(edge_ids)):
        _issue(issues, "duplicate_identifier", "$.edges", "edge identifiers must be unique")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        if any(visit(child) for child in adjacency.get(node_id, ())):
            return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    if any(visit(node_id) for node_id in adjacency if node_id not in visited):
        _issue(issues, "graph_cycle", "$.edges", "condition graph must be acyclic")
    for set_index, sufficient in enumerate(doc.get("minimal_sufficient_sets", [])):
        if not sufficient:
            _issue(issues, "empty_sufficient_set", f"$.minimal_sufficient_sets[{set_index}]", "set cannot be empty")
        for condition_id in sufficient:
            if condition_id not in adjacency:
                _issue(issues, "dangling_sufficient_set", f"$.minimal_sufficient_sets[{set_index}]", "condition does not exist")
    _revision(doc, issues)


def _validate_evidence(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("evidence_atom_id", "source_id", "source_locator", "claim", "epistemic_type", "independence_group", "target_relevance"), issues)
    _stable_id(doc.get("evidence_atom_id"), "$.evidence_atom_id", issues)
    _stable_id(doc.get("source_id"), "$.source_id", issues)
    epistemic = doc.get("epistemic_type")
    if epistemic == "attributed_statement" and not doc.get("speaker"):
        _issue(issues, "attribution_loss", "$.speaker", "attributed statement requires a speaker")
    if epistemic == "rhetoric" and doc.get("admitted_as_fact"):
        _issue(issues, "rhetorical_leakage", "$.admitted_as_fact", "rhetoric cannot enter the fact layer")
    if epistemic == "causal_claim" and not doc.get("causal_support_atom_ids"):
        _issue(issues, "unsupported_causal_claim", "$.causal_support_atom_ids", "causal claim requires explicit support")
    if not doc.get("target_relevance"):
        _issue(issues, "target_relevance_missing", "$.target_relevance", "proposition must be answer-directed")
    extraction = doc.get("extraction_confidence")
    if not isinstance(extraction, (int, float)) or isinstance(extraction, bool) or not 0 <= extraction <= 1:
        _issue(issues, "confidence_missing_or_invalid", "$.extraction_confidence", "extraction confidence must be between zero and one")
    if "truth_confidence" not in doc:
        _issue(issues, "confidence_missing_or_invalid", "$.truth_confidence", "truth confidence must be explicit, including null when unknown")
    else:
        truth = doc["truth_confidence"]
        if truth is not None and (not isinstance(truth, (int, float)) or isinstance(truth, bool) or not 0 <= truth <= 1):
            _issue(issues, "confidence_missing_or_invalid", "$.truth_confidence", "truth confidence must be null or between zero and one")
    _provenance(doc.get("provenance"), issues, "$.provenance")
    _revision(doc, issues)


def _validate_scene(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("scene_id", "question_id", "actors", "events"), issues)
    _stable_id(doc.get("scene_id"), "$.scene_id", issues)
    for index, event in enumerate(doc.get("events", [])):
        path = f"$.events[{index}]"
        if isinstance(event, Mapping):
            _required(event, ("event_id", "status", "actor_ids", "action", "supporting_atom_ids"), issues, path)
            if event.get("status") == "confirmed" and not event.get("supporting_atom_ids"):
                _issue(issues, "unsupported_scene_fact", f"{path}.supporting_atom_ids", "confirmed event requires admissible support")
    _provenance(doc.get("provenance"), issues, "$.provenance")
    _revision(doc, issues)


def _validate_missing(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("estimate_id", "condition_id", "estimate_type", "lower", "upper", "unit", "coverage_semantics", "coverage", "method", "dependence_case"), issues)
    _stable_id(doc.get("estimate_id"), "$.estimate_id", issues)
    lower, upper = doc.get("lower"), doc.get("upper")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)) or lower > upper:
        _issue(issues, "invalid_bounds", "$", "lower and upper must be ordered numbers")
    elif lower == upper:
        _issue(issues, "hidden_point_imputation", "$", "missing condition cannot be silently point-filled")
    _provenance(doc.get("bound_provenance"), issues, "$.bound_provenance")
    if doc.get("dependence_case") not in {"independent", "dependent", "conditionally_dependent", "unknown"}:
        _issue(issues, "dependence_error", "$.dependence_case", "dependence must be declared")
    _revision(doc, issues)


def _validate_derived(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("factor_id", "name", "status", "input_condition_ids", "composition", "target_paths", "hypothesis", "falsification_conditions", "validation_status"), issues)
    _stable_id(doc.get("factor_id"), "$.factor_id", issues)
    if len(doc.get("input_condition_ids", [])) < 2:
        _issue(issues, "factor_not_composite", "$.input_condition_ids", "derived factor requires at least two inputs")
    if not doc.get("falsification_conditions"):
        _issue(issues, "not_falsifiable", "$.falsification_conditions", "candidate must state a falsification test")
    _revision(doc, issues)


def _validate_interval(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    _required(doc, ("evaluation_id", "question_id", "forecast", "reference_value", "normalized_width", "baseline_interval", "information_gain", "status"), issues)
    _stable_id(doc.get("evaluation_id"), "$.evaluation_id", issues)
    forecast = doc.get("forecast", {})
    if not isinstance(forecast, Mapping) or not all(key in forecast for key in ("p05", "p50", "p95", "unit")):
        _issue(issues, "interval_shape", "$.forecast", "p05, p50, p95 and unit are required")
    elif not forecast["p05"] <= forecast["p50"] <= forecast["p95"]:
        _issue(issues, "invalid_bounds", "$.forecast", "quantiles must be ordered")
    if doc.get("normalized_width", 0) >= 1 and doc.get("status") in {"calibrated_informative", "uncalibrated_informative"}:
        _issue(issues, "vacuous_wide_interval", "$.normalized_width", "full-scale interval cannot be called informative")
    _revision(doc, issues)


def _validate_package(doc: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _schema_version(doc, issues)
    required_sections = (
        "package_id", "run_id", "package_state", "answer_contract", "condition_graph", "sources",
        "evidence_propositions", "event_scene", "missing_conditions", "derived_factors", "calculations",
        "interval_audit", "conclusion", "answerability", "empty_section_reasons",
    )
    # Partial packages deliberately carry null/empty sections.  Presence is
    # mandatory, while an explicit reason below distinguishes absence from an
    # omitted contract field.
    for name in required_sections:
        if name not in doc:
            _issue(issues, "required_field", f"$.{name}", "field is required")
    _stable_id(doc.get("package_id"), "$.package_id", issues)
    _stable_id(doc.get("run_id"), "$.run_id", issues)
    state = doc.get("package_state")
    if state not in {"complete", "partial"}:
        _issue(issues, "package_state", "$.package_state", "must be complete or partial")
    if state == "complete":
        for section in ("event_scene", "interval_audit", "conclusion"):
            if doc.get(section) is None:
                _issue(issues, "complete_section_missing", f"$.{section}", "complete package requires a non-null section")
    answerability = doc.get("answerability", {})
    terminal = answerability.get("status") if isinstance(answerability, Mapping) else None
    if terminal not in _TERMINAL_STATES:
        _issue(issues, "terminal_answerability_missing", "$.answerability.status", "explicit terminal state is required")
    if terminal != "answerable_bounded" and not answerability.get("reasons"):
        _issue(issues, "answerability_reason_missing", "$.answerability.reasons", "non-answerable result requires reasons")
    empty_reasons = doc.get("empty_section_reasons", {})
    for section in ("sources", "evidence_propositions", "event_scene", "missing_conditions", "derived_factors", "calculations", "interval_audit", "conclusion"):
        if doc.get(section) in (None, [], {}) and not empty_reasons.get(section):
            _issue(issues, "empty_section_reason_missing", f"$.{section}", "empty section requires an explicit reason")
    if isinstance(doc.get("answer_contract"), Mapping):
        nested: list[ValidationIssue] = []
        _validate_answer_contract(doc["answer_contract"], nested)
        issues.extend(ValidationIssue(i.code, f"$.answer_contract{i.path[1:]}", i.message) for i in nested)
    if isinstance(doc.get("condition_graph"), Mapping):
        nested = []
        _validate_graph(doc["condition_graph"], nested)
        issues.extend(ValidationIssue(i.code, f"$.condition_graph{i.path[1:]}", i.message) for i in nested)
    evidence = doc.get("evidence_propositions", [])
    evidence_by_id = {item.get("evidence_atom_id"): item for item in evidence if isinstance(item, Mapping)}
    excluded_source_ids = {
        item.get("source_id")
        for item in doc.get("sources", [])
        if isinstance(item, Mapping) and item.get("evidence_disposition") == "excluded"
    }
    for index, item in enumerate(evidence):
        nested = []
        _validate_evidence(item, nested)
        issues.extend(ValidationIssue(i.code, f"$.evidence_propositions[{index}]{i.path[1:]}", i.message) for i in nested)
        if isinstance(item, Mapping) and item.get("source_id") in excluded_source_ids:
            _issue(
                issues,
                "excluded_source_leakage",
                f"$.evidence_propositions[{index}].source_id",
                "an explicitly irrelevant source cannot enter target evidence",
            )
    scene = doc.get("event_scene")
    if isinstance(scene, Mapping):
        nested = []
        _validate_scene(scene, nested)
        issues.extend(ValidationIssue(i.code, f"$.event_scene{i.path[1:]}", i.message) for i in nested)
        for event_index, event in enumerate(scene.get("events", [])):
            for atom_id in event.get("supporting_atom_ids", []):
                atom = evidence_by_id.get(atom_id)
                if atom and atom.get("epistemic_type") not in _FACT_TYPES:
                    _issue(issues, "rhetorical_leakage", f"$.event_scene.events[{event_index}].supporting_atom_ids", "only admissible fact atoms support scene facts")
    for index, item in enumerate(doc.get("missing_conditions", [])):
        nested = []
        _validate_missing(item, nested)
        issues.extend(ValidationIssue(i.code, f"$.missing_conditions[{index}]{i.path[1:]}", i.message) for i in nested)
    for index, item in enumerate(doc.get("derived_factors", [])):
        nested = []
        _validate_derived(item, nested)
        issues.extend(ValidationIssue(i.code, f"$.derived_factors[{index}]{i.path[1:]}", i.message) for i in nested)
    interval = doc.get("interval_audit")
    if isinstance(interval, Mapping):
        nested = []
        _validate_interval(interval, nested)
        issues.extend(ValidationIssue(i.code, f"$.interval_audit{i.path[1:]}", i.message) for i in nested)
    _revision(doc, issues)


_VALIDATORS = {
    "answer_contract": _validate_answer_contract,
    "condition_graph": _validate_graph,
    "evidence_proposition": _validate_evidence,
    "evidence_atom": _validate_evidence,
    "event_scene": _validate_scene,
    "reconstructed_scene": _validate_scene,
    "missing_condition": _validate_missing,
    "condition_estimate": _validate_missing,
    "derived_factor": _validate_derived,
    "derived_factor_candidate": _validate_derived,
    "interval_audit": _validate_interval,
    "forecast_interval_evaluation": _validate_interval,
    "analysis_package": _validate_package,
    "complete_analysis_package": _validate_package,
    "partial_analysis_package": _validate_package,
}


def validate_document(kind: str, document: Mapping[str, Any], *, raise_on_error: bool = True) -> tuple[ValidationIssue, ...]:
    """Validate a JSON-compatible domain document by stable contract name."""

    normalized = kind.removesuffix(".schema.json").replace("-", "_")
    validator = _VALIDATORS.get(normalized)
    if validator is None:
        raise ValueError(f"unknown contract kind: {kind}")
    issues: list[ValidationIssue] = []
    validator(document, issues)
    result = tuple(issues)
    if result and raise_on_error:
        raise ContractValidationError(result)
    return result


def validate_model(model: Any, *, raise_on_error: bool = True) -> tuple[ValidationIssue, ...]:
    names = {
        "AnswerContract": "answer_contract", "ConditionGraph": "condition_graph",
        "EvidenceProposition": "evidence_proposition", "EventScene": "event_scene",
        "MissingCondition": "missing_condition", "DerivedFactor": "derived_factor",
        "IntervalAudit": "interval_audit", "AnalysisPackage": "analysis_package",
    }
    kind = names.get(type(model).__name__)
    if not kind:
        raise ValueError(f"unsupported model type: {type(model).__name__}")
    return validate_document(kind, to_dict(model), raise_on_error=raise_on_error)


def validate_graph(graph: ConditionGraph | Mapping[str, Any], *, raise_on_error: bool = True) -> tuple[ValidationIssue, ...]:
    return validate_document("condition_graph", to_dict(graph), raise_on_error=raise_on_error)


def validate_package(package: AnalysisPackage | Mapping[str, Any], *, raise_on_error: bool = True) -> tuple[ValidationIssue, ...]:
    return validate_document("analysis_package", to_dict(package), raise_on_error=raise_on_error)


def load_and_validate(path: str | Path, kind: str | None = None, *, raise_on_error: bool = True) -> tuple[dict[str, Any], tuple[ValidationIssue, ...]]:
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    contract_kind = kind or document.get("contract_kind")
    if not contract_kind:
        raise ValueError("contract kind is required")
    issues = validate_document(contract_kind, document, raise_on_error=raise_on_error)
    return document, issues
