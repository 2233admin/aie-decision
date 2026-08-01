"""Standalone orchestration from an answer contract to an auditable package."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .decomposition import build_condition_graph, create_answer_contract, evaluate_answerability
from .evidence import ingest_source, segment_propositions
from .factors import generate_candidate
from .intervals import ForecastInterval, IntervalKind, audit_interval
from .ledger import AnalysisLedger
from .models import (
    AnalysisPackage,
    AnswerType,
    ConditionEdge,
    ConditionNode,
    ConditionStatus,
    CoverageSemantics,
    DerivedFactor,
    IntervalAudit,
    IntervalAuditStatus,
    MissingCondition,
    Necessity,
    ProvenanceRef,
    Revision,
    ValidationStatus,
)
from .scenes import reconstruct_event_scene
from .validation import validate_package


def _stable_id(prefix: str, value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]}"


def _normalized_claim(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _proposition_matches_selector(proposition: Any, selector: Mapping[str, Any]) -> bool:
    for name, actual in (
        ("source_id", proposition.source_id),
        ("source_locator", proposition.source_locator),
        ("evidence_atom_id", proposition.evidence_atom_id),
    ):
        expected = selector.get(name)
        if expected and str(expected) != actual:
            return False
    expected_claim = selector.get("normalized_claim", selector.get("claim"))
    if expected_claim and _normalized_claim(expected_claim) != _normalized_claim(proposition.claim):
        return False
    return bool(
        selector.get("source_id")
        and any(selector.get(name) for name in ("source_locator", "evidence_atom_id", "normalized_claim", "claim"))
    )


def _required(data: Mapping[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return value


@dataclass(frozen=True, slots=True)
class CompilationResult:
    package: AnalysisPackage
    ledger: AnalysisLedger
    validation_issues: tuple[Any, ...]


def compile_analysis(payload: Mapping[str, Any]) -> CompilationResult:
    """Compile supplied materials only; this function never acquires external data."""
    answer_data = _required(payload, "answer_contract")
    if not isinstance(answer_data, Mapping):
        raise ValueError("answer_contract must be an object")
    target_data = _required(answer_data, "target")
    if not isinstance(target_data, Mapping):
        raise ValueError("answer_contract.target must be an object")
    contract = create_answer_contract(
        question_id=str(_required(answer_data, "question_id")),
        question=str(_required(answer_data, "question")),
        answer_type=AnswerType(str(_required(answer_data, "answer_type"))),
        subject=str(_required(target_data, "entity")),
        target=str(_required(target_data, "measure")),
        unit=str(_required(target_data, "unit")),
        observation_cutoff=str(_required(answer_data, "observation_cutoff")),
        prediction_horizon=answer_data.get("prediction_horizon"),
        geography=answer_data.get("geography"),
        decision_use=answer_data.get("decision_use"),
        decision_thresholds=tuple(float(item) for item in answer_data.get("decision_thresholds", ())),
        uncertainty_semantics=(
            CoverageSemantics(str(answer_data["uncertainty_semantics"]))
            if answer_data.get("uncertainty_semantics")
            else None
        ),
        requested_coverage=(float(answer_data["requested_coverage"]) if answer_data.get("requested_coverage") is not None else None),
        acceptable_width=answer_data.get("acceptable_width"),
    )

    contract_provenance = ProvenanceRef(
        source_id=contract.question_id,
        locator="answer_contract",
        observed_at=contract.observation_cutoff,
        transformation="answer-oriented decomposition",
    )

    def stage_revision(kind: str, stable_id: str) -> Revision:
        return Revision(
            revision_id=_stable_id(f"{kind}-rev", {"id": stable_id, "contract": contract.question_id}),
            sequence=1,
            created_at=contract.observation_cutoff,
        )

    conditions = tuple(
        ConditionNode(
            condition_id=str(_required(item, "condition_id")),
            name=str(_required(item, "name")),
            value_type=str(_required(item, "value_type")),
            necessity=Necessity(str(_required(item, "necessity"))),
            status=ConditionStatus(str(_required(item, "status"))),
            answer_impact=str(_required(item, "answer_impact")),
            unit=item.get("unit"),
            provenance=(contract_provenance,),
            revision=stage_revision("condition", str(item["condition_id"])),
        )
        for item in payload.get("conditions", ())
    )
    if not conditions:
        raise ValueError("at least one answer-relevant condition is required")
    edges = tuple(
        ConditionEdge(
            edge_id=str(item.get("edge_id") or _stable_id("edge", item)),
            from_id=str(_required(item, "from_id")),
            to_id=str(_required(item, "to_id")),
            relation=str(_required(item, "relation")),
            evidence_status=str(item.get("evidence_status", "proposed")),
            answer_impact=str(_required(item, "answer_impact")),
            direction=item.get("direction"),
            provenance=(contract_provenance,),
            revision=stage_revision("edge", str(item.get("edge_id") or _stable_id("edge", item))),
        )
        for item in payload.get("edges", ())
    )
    graph = build_condition_graph(
        contract,
        conditions,
        edges,
        payload.get("minimal_sufficient_sets", ()),
        graph_id=payload.get("graph_id"),
    )
    answerability = evaluate_answerability(contract, graph)

    sources = []
    propositions = []
    for source_data in payload.get("sources", ()):
        content = str(_required(source_data, "content"))
        source = ingest_source(
            content,
            title=str(_required(source_data, "title")),
            uri=source_data.get("uri"),
            publisher=source_data.get("publisher"),
            published_at=source_data.get("published_at"),
            retrieved_at=source_data.get("retrieved_at"),
            source_id=source_data.get("source_id"),
            source_location=source_data.get("source_location"),
            speaker=source_data.get("speaker"),
            transformation_lineage=tuple(source_data.get("transformation_lineage", ())),
        )
        relevance_declared = "target_relevance" in source_data
        target_relevance = tuple(source_data.get("target_relevance", ()))
        source = replace(
            source,
            target_relevance=target_relevance,
            evidence_disposition=(
                "admitted" if target_relevance else "excluded" if relevance_declared else "unassessed"
            ),
            exclusion_reason=(
                "source explicitly declared irrelevant to the answer target"
                if relevance_declared and not target_relevance
                else None
            ),
        )
        sources.append(source)
        if relevance_declared and not target_relevance:
            continue
        propositions.extend(
            segment_propositions(source, content, target_relevance=target_relevance)
        )
    fact_fields_by_atom: dict[str, Mapping[str, Any]] = {}
    for index, binding in enumerate(payload.get("scene_fact_fields", ())):
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("selector"), Mapping)
            or not isinstance(binding.get("fields"), Mapping)
        ):
            raise ValueError(f"scene_fact_fields[{index}] requires selector and fields objects")
        matches = [item for item in propositions if _proposition_matches_selector(item, binding["selector"])]
        if len(matches) != 1:
            raise ValueError(f"scene_fact_fields[{index}] selector resolved {len(matches)} propositions")
        atom_id = matches[0].evidence_atom_id
        if atom_id in fact_fields_by_atom:
            raise ValueError(f"scene_fact_fields[{index}] duplicates proposition {atom_id}")
        fact_fields_by_atom[atom_id] = binding["fields"]
    scene = (
        reconstruct_event_scene(
            question_id=contract.question_id,
            propositions=propositions,
            fact_fields=fact_fields_by_atom,
        )
        if propositions else None
    )

    missing_conditions = tuple(
        MissingCondition(
            estimate_id=str(item.get("estimate_id") or _stable_id("estimate", item)),
            condition_id=str(_required(item, "condition_id")),
            estimate_type=str(_required(item, "estimate_type")),
            lower=float(_required(item, "lower")),
            upper=float(_required(item, "upper")),
            unit=str(_required(item, "unit")),
            coverage_semantics=CoverageSemantics(str(_required(item, "coverage_semantics"))),
            coverage=float(_required(item, "coverage")),
            method=str(_required(item, "method")),
            input_atom_ids=tuple(item.get("input_atom_ids", ())),
            assumptions=tuple(item.get("assumptions", ())),
            bound_provenance=(contract_provenance,),
            dependence_case=str(_required(item, "dependence_case")),
            calibration_profile_id=item.get("calibration_profile_id"),
            valid_until=item.get("valid_until"),
            revision=stage_revision("estimate", str(item.get("estimate_id") or _stable_id("estimate", item))),
        )
        for item in payload.get("missing_conditions", ())
    )

    derived_factors = []
    for item in payload.get("derived_factors", ()):
        candidate = generate_candidate(
            str(_required(item, "factor_id")),
            str(_required(item, "name")),
            tuple(_required(item, "input_condition_ids")),
            str(_required(item, "hypothesis")),
            tuple(_required(item, "observable_implications")),
            tuple(_required(item, "falsification_conditions")),
        )
        derived_factors.append(
            DerivedFactor(
                factor_id=candidate.factor_id,
                name=candidate.label,
                status=candidate.status.value,
                input_condition_ids=candidate.contributing_conditions,
                composition=dict(item.get("composition", {"mechanism": candidate.mechanism})),
                unit=str(_required(item, "unit")),
                time_window=str(_required(item, "time_window")),
                target_paths=tuple(_required(item, "target_paths")),
                hypothesis=candidate.mechanism,
                falsification_conditions=candidate.rejection_conditions,
                validation_status=ValidationStatus.UNVALIDATED,
                provenance=(contract_provenance,),
                revision=stage_revision("factor", candidate.factor_id),
            )
        )

    interval_model = None
    interval_data = payload.get("forecast_interval")
    if interval_data:
        forecast = ForecastInterval(
            target=str(_required(interval_data, "target")),
            horizon=str(_required(interval_data, "horizon")),
            unit=str(_required(interval_data, "unit")),
            population=str(_required(interval_data, "population")),
            coverage_level=float(_required(interval_data, "coverage_level")),
            conditional_assumptions=tuple(interval_data.get("conditional_assumptions", ())),
            generation_method=str(_required(interval_data, "generation_method")),
            reference_time=str(_required(interval_data, "reference_time")),
            lower=float(_required(interval_data, "lower")),
            upper=float(_required(interval_data, "upper")),
            kind=IntervalKind(str(interval_data.get("kind", IntervalKind.PREDICTION.value))),
        )
        baseline_data = interval_data.get("baseline")
        baseline = None
        if baseline_data:
            baseline = ForecastInterval(
                target=forecast.target,
                horizon=forecast.horizon,
                unit=forecast.unit,
                population=forecast.population,
                coverage_level=forecast.coverage_level,
                conditional_assumptions=(),
                generation_method="declared_baseline",
                reference_time=forecast.reference_time,
                lower=float(_required(baseline_data, "lower")),
                upper=float(_required(baseline_data, "upper")),
                kind=forecast.kind,
            )
        audit = audit_interval(
            forecast,
            scale=float(_required(interval_data, "reference_value")),
            baseline_width=(baseline.upper - baseline.lower) if baseline else None,
            thresholds=contract.decision_thresholds,
            baseline=baseline,
        )
        status = IntervalAuditStatus.UNCALIBRATED_INFORMATIVE if audit.informative else IntervalAuditStatus.UNCALIBRATED_UNINFORMATIVE
        interval_model = IntervalAudit(
            evaluation_id=str(interval_data.get("evaluation_id") or _stable_id("interval", interval_data)),
            question_id=contract.question_id,
            forecast={
                "target": forecast.target,
                "horizon": forecast.horizon,
                "unit": forecast.unit,
                "coverage_level": forecast.coverage_level,
                "p05": forecast.lower,
                "p50": (forecast.lower + forecast.upper) / 2,
                "p95": forecast.upper,
                "lower": forecast.lower,
                "upper": forecast.upper,
                "coverage_semantics": contract.uncertainty_semantics.value if contract.uncertainty_semantics else "unknown",
            },
            reference_value=float(interval_data["reference_value"]),
            normalized_width=audit.normalized_width,
            empirical_coverage=None,
            baseline_interval={"lower": baseline.lower, "upper": baseline.upper} if baseline else {},
            information_gain=audit.baseline_improvement or 0.0,
            status=status,
            next_information_actions=tuple(audit.flags),
            provenance=(contract_provenance,),
            revision=stage_revision("interval", str(interval_data.get("evaluation_id") or _stable_id("interval", interval_data))),
        )

    package_state = "complete" if scene is not None and interval_model is not None else "partial"
    empty_reasons = {}
    if not sources:
        empty_reasons["sources"] = "no source material supplied"
    if not propositions:
        empty_reasons["evidence_propositions"] = "no propositions compiled from supplied material"
    if scene is None:
        empty_reasons["event_scene"] = "no admissible source propositions supplied"
    if not missing_conditions:
        empty_reasons["missing_conditions"] = "no bounded missing-condition estimate supplied"
    if not derived_factors:
        empty_reasons["derived_factors"] = "no derived-factor hypothesis supplied"
    empty_reasons["calculations"] = "no additional calculation records supplied"
    if interval_model is None:
        empty_reasons["interval_audit"] = "no forecast interval supplied"
    package = AnalysisPackage(
        package_id=str(payload.get("package_id") or _stable_id("package", payload)),
        run_id=str(payload.get("run_id") or _stable_id("run", payload)),
        package_state=package_state,
        answer_contract=contract,
        condition_graph=graph,
        sources=tuple(sources),
        evidence_propositions=tuple(propositions),
        event_scene=scene,
        missing_conditions=missing_conditions,
        derived_factors=tuple(derived_factors),
        calculations=(),
        interval_audit=interval_model,
        conclusion={"answerability": answerability.status.value, "reasons": list(answerability.reasons)},
        answerability=answerability,
        empty_section_reasons=empty_reasons,
        revision=stage_revision("package", str(payload.get("package_id") or _stable_id("package", payload))),
        provenance=(contract_provenance,),
    )
    issues = validate_package(package, raise_on_error=False)
    ledger = AnalysisLedger(package.run_id)
    for record_type, records in (
        ("answer_contract", (contract,)),
        ("condition_graph", (graph,)),
        ("source", tuple(sources)),
        ("evidence_proposition", tuple(propositions)),
        ("event_scene", (scene,) if scene else ()),
        ("missing_condition", missing_conditions),
        ("derived_factor", tuple(derived_factors)),
        ("interval_audit", (interval_model,) if interval_model else ()),
        ("analysis_package", (package,)),
    ):
        for record in records:
            ledger.append(record_type, record)
    return CompilationResult(package, ledger, issues)
