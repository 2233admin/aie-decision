"""Experimental closed-loop search over declared Fermi candidates.

The controller is deliberately independent from candidate generation and numeric
execution.  The first generator is a deterministic candidate graph so the full
route can be tested before an LLM or GPU is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Mapping, Protocol, Sequence

from .fermi import estimate_fermi
from .ledger import AnalysisLedger
from .models import SCHEMA_VERSION, Revision

_MUTATION_KINDS = {"seed", "expand", "revise", "ablate"}


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    formula: str
    parent_candidate_id: str | None = None
    mutation_kind: str = "seed"
    prior_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class SearchBudget:
    max_candidates: int = 100
    max_rounds: int = 10
    max_evaluations: int = 100
    max_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class SearchEvent:
    schema_version: str
    event_id: str
    run_id: str
    candidate_id: str
    state: str
    round_index: int
    reason: str
    data: dict[str, Any]
    revision: Revision


class CandidateGenerator(Protocol):
    def seeds(self) -> tuple[CandidateSpec, ...]: ...

    def children(
        self, parent: CandidateSpec, passed: bool
    ) -> tuple[CandidateSpec, ...]: ...


class CandidateEvaluator(Protocol):
    def evaluate(
        self, payload: Mapping[str, Any], candidate: CandidateSpec
    ) -> dict[str, Any]: ...


class DeclaredCandidateGenerator:
    """Deterministic generator seam backed by a validated candidate graph."""

    def __init__(self, candidates: Sequence[CandidateSpec]):
        self._items = tuple(candidates)
        self._by_id = {candidate.candidate_id: candidate for candidate in self._items}
        if len(self._by_id) != len(self._items):
            raise ValueError("candidate_id values must be unique")
        for candidate in self._items:
            if candidate.mutation_kind not in _MUTATION_KINDS:
                raise ValueError(
                    f"unsupported mutation_kind: {candidate.mutation_kind}"
                )
            if (
                candidate.mutation_kind == "seed"
                and candidate.parent_candidate_id is not None
            ):
                raise ValueError("seed candidate cannot have a parent")
            if (
                candidate.mutation_kind != "seed"
                and candidate.parent_candidate_id is None
            ):
                raise ValueError("non-seed candidate requires a parent")
            if (
                candidate.parent_candidate_id is not None
                and candidate.parent_candidate_id not in self._by_id
            ):
                raise ValueError(
                    f"candidate parent does not exist: {candidate.parent_candidate_id}"
                )
            if candidate.parent_candidate_id == candidate.candidate_id:
                raise ValueError("candidate cannot be its own parent")
        self._reject_cycles()

    def _reject_cycles(self) -> None:
        for candidate in self._items:
            seen: set[str] = set()
            current = candidate
            while current.parent_candidate_id is not None:
                if current.candidate_id in seen:
                    raise ValueError("candidate graph must be acyclic")
                seen.add(current.candidate_id)
                current = self._by_id[current.parent_candidate_id]

    def seeds(self) -> tuple[CandidateSpec, ...]:
        return tuple(
            candidate
            for candidate in self._items
            if candidate.parent_candidate_id is None
            or candidate.mutation_kind == "seed"
        )

    def children(
        self, parent: CandidateSpec, passed: bool
    ) -> tuple[CandidateSpec, ...]:
        allowed = {"ablate"} if passed else {"expand", "revise"}
        return tuple(
            candidate
            for candidate in self._items
            if candidate.parent_candidate_id == parent.candidate_id
            and candidate.mutation_kind in allowed
        )


class FermiCandidateEvaluator:
    """CPU reference adapter around the existing interval evaluator."""

    def evaluate(
        self, payload: Mapping[str, Any], candidate: CandidateSpec
    ) -> dict[str, Any]:
        request = dict(payload)
        request.pop("candidates", None)
        request.pop("budget", None)
        request.pop("run_id", None)
        request["formula"] = candidate.formula
        return estimate_fermi(request)


def _positive_int(value: Any, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _budget(payload: Mapping[str, Any]) -> SearchBudget:
    raw = payload.get("budget", {})
    if not isinstance(raw, Mapping):
        raise ValueError("budget must be an object")
    seconds = raw.get("max_seconds", 30.0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("budget.max_seconds must be a positive number")
    seconds = float(seconds)
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds <= 0:
        raise ValueError("budget.max_seconds must be a positive number")
    return SearchBudget(
        max_candidates=_positive_int(
            raw.get("max_candidates"), "budget.max_candidates", 100
        ),
        max_rounds=_positive_int(raw.get("max_rounds"), "budget.max_rounds", 10),
        max_evaluations=_positive_int(
            raw.get("max_evaluations"), "budget.max_evaluations", 100
        ),
        max_seconds=seconds,
    )


def _candidates(payload: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    raw = payload.get("candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("candidates must be a non-empty array")
    result: list[CandidateSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"candidates[{index}] must be an object")
        candidate_id = item.get("candidate_id")
        formula = item.get("formula")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"candidates[{index}].candidate_id is required")
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError(f"candidates[{index}].formula is required")
        parent = item.get("parent_candidate_id")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise ValueError(
                f"candidates[{index}].parent_candidate_id must be a non-empty string"
            )
        mutation = item.get("mutation_kind", "seed" if parent is None else "revise")
        if not isinstance(mutation, str):
            raise ValueError(f"candidates[{index}].mutation_kind must be a string")
        prior = item.get("prior_weight", 1.0)
        if isinstance(prior, bool) or not isinstance(prior, (int, float)):
            raise ValueError(
                f"candidates[{index}].prior_weight must be a positive number"
            )
        prior = float(prior)
        if prior != prior or prior in (float("inf"), float("-inf")) or prior <= 0:
            raise ValueError(
                f"candidates[{index}].prior_weight must be a positive number"
            )
        result.append(
            CandidateSpec(
                candidate_id.strip(), formula.strip(), parent, mutation, prior
            )
        )
    return tuple(result)


def _mutation_templates(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = payload.get("mutation_templates", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("mutation_templates must be an array")
    module = __import__(
        "aie_decision.candidate_generation", fromlist=["MutationTemplate"]
    )
    templates = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"mutation_templates[{index}] must be an object")
        reasons = item.get("diagnostic_reasons", ())
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            raise ValueError(
                f"mutation_templates[{index}].diagnostic_reasons must be an array"
            )
        templates.append(
            module.MutationTemplate(
                template_id=str(item.get("template_id", "")),
                formula_template=str(item.get("formula_template", "")),
                diagnostic_reasons=tuple(str(reason) for reason in reasons),
                mutation_kind=str(item.get("mutation_kind", "revise")),
                prior_multiplier=item.get("prior_multiplier", 1.0),
                rationale_template=str(
                    item.get(
                        "rationale_template",
                        "apply {template_id} for {diagnostic_reason}",
                    )
                ),
            )
        )
    return tuple(templates)


def _ablatable_variables(payload: Mapping[str, Any]) -> frozenset[str]:
    raw = payload.get("variables", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return frozenset()
    return frozenset(
        str(item.get("name"))
        for item in raw
        if isinstance(item, Mapping)
        and item.get("ablatable") is True
        and item.get("name")
    )


def _run_id(payload: Mapping[str, Any]) -> str:
    supplied = payload.get("run_id")
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied.strip():
            raise ValueError("run_id must be a non-empty string")
        return supplied.strip()
    return "search-loop"


def _heuristic_likelihood(result: Mapping[str, Any]) -> float:
    """Return an explicitly uncalibrated Bayes-inspired likelihood proxy."""
    width = float(result["absolute_width"])
    acceptable = result.get("acceptable_width")
    if (
        isinstance(acceptable, (int, float))
        and not isinstance(acceptable, bool)
        and acceptable > 0
    ):
        scale = float(acceptable)
    else:
        reference = result.get("reference_value")
        scale = (
            abs(float(reference))
            if isinstance(reference, (int, float)) and reference
            else 1.0
        )
    width_fit = 1.0 / (1.0 + width / scale)
    variables = result.get("minimal_variables", ())
    supported = sum(
        str(item.get("method", "")).casefold()
        in {"observed", "measured", "primary_record"}
        for item in variables
        if isinstance(item, Mapping)
    )
    support_fraction = supported / len(variables) if variables else 0.0
    evidence_fit = 0.25 + 0.75 * support_fraction
    robustness_fit = 1.0 if result.get("decision_robust") else 0.5
    return max(1e-12, width_fit * evidence_fit * robustness_fit)


def search_fermi(
    payload: Mapping[str, Any],
    *,
    generator: CandidateGenerator | None = None,
    evaluator: CandidateEvaluator | None = None,
    proposer: Any | None = None,
) -> dict[str, Any]:
    """Run a bounded expand/evaluate/minimize loop on the CPU reference path."""
    if not isinstance(payload, Mapping):
        raise ValueError("search input must be an object")
    budget = _budget(payload)
    declared = _candidates(payload)
    mutation_templates = _mutation_templates(payload)
    ablatable_variables = _ablatable_variables(payload)
    candidate_generator = generator or DeclaredCandidateGenerator(declared)
    candidate_evaluator = evaluator or FermiCandidateEvaluator()
    run_id = _run_id(payload)
    started_at = str(payload.get("started_at") or "1970-01-01T00:00:00Z")
    started = monotonic()
    ledger = AnalysisLedger(run_id)
    events: list[SearchEvent] = []
    evaluations: dict[str, dict[str, Any]] = {}
    candidate_metadata: dict[str, dict[str, Any]] = {
        candidate.candidate_id: {
            "generation_method": "declared",
            "rationale": "declared candidate",
        }
        for candidate in declared
    }
    queue = list(candidate_generator.seeds())
    queued = {candidate.candidate_id for candidate in queue}
    known_formulas = {candidate.formula for candidate in declared}
    stop_reason = ""

    def record(
        candidate_id: str,
        state: str,
        round_index: int,
        reason: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        sequence = len(events) + 1
        event_id = f"{run_id}-event-{sequence:06d}"
        event = SearchEvent(
            schema_version=SCHEMA_VERSION,
            event_id=event_id,
            run_id=run_id,
            candidate_id=candidate_id,
            state=state,
            round_index=round_index,
            reason=reason,
            data=dict(data or {}),
            revision=Revision(
                revision_id=f"{event_id}-r1",
                sequence=1,
                created_at=started_at,
            ),
        )
        events.append(event)
        ledger.append("search_event", event)

    for candidate in queue:
        record(
            candidate.candidate_id,
            "SEED",
            0,
            "declared_seed",
            {
                "formula": candidate.formula,
                "parent_candidate_id": candidate.parent_candidate_id,
                "mutation_kind": candidate.mutation_kind,
                "prior_weight": candidate.prior_weight,
            },
        )

    round_index = 0
    while queue:
        if round_index >= budget.max_rounds:
            stop_reason = "budget-exhausted"
            break
        round_index += 1
        current, queue = queue, []
        for candidate in current:
            if len(evaluations) >= min(budget.max_candidates, budget.max_evaluations):
                queue.append(candidate)
                stop_reason = "budget-exhausted"
                break
            if monotonic() - started >= budget.max_seconds:
                queue.append(candidate)
                stop_reason = "budget-exhausted"
                break
            record(
                candidate.candidate_id, "VALIDATE", round_index, "candidate_selected"
            )
            try:
                result = candidate_evaluator.evaluate(payload, candidate)
            except (TypeError, ValueError) as exc:
                evaluations[candidate.candidate_id] = {
                    "candidate_id": candidate.candidate_id,
                    "parent_candidate_id": candidate.parent_candidate_id,
                    "mutation_kind": candidate.mutation_kind,
                    "formula": candidate.formula,
                    "passed": False,
                    "status": "invalid",
                    "error": str(exc),
                }
                record(
                    candidate.candidate_id,
                    "REJECT",
                    round_index,
                    "invalid_candidate",
                    {"error": str(exc)},
                )
                continue

            passed = bool(
                result["within_acceptable_width"] and result["decision_robust"]
            )
            heuristic_likelihood = _heuristic_likelihood(result)
            row = {
                "candidate_id": candidate.candidate_id,
                "parent_candidate_id": candidate.parent_candidate_id,
                "mutation_kind": candidate.mutation_kind,
                "formula": candidate.formula,
                "generation": candidate_metadata.get(candidate.candidate_id, {}),
                "prior_weight": candidate.prior_weight,
                "heuristic_likelihood": heuristic_likelihood,
                "unnormalized_pseudo_posterior": candidate.prior_weight
                * heuristic_likelihood,
                "passed": passed,
                "status": result["status"],
                "variable_names": [
                    item["name"] for item in result["minimal_variables"]
                ],
                "variable_count": result["minimal_variable_count"],
                "target_interval": result["target_interval"],
                "absolute_width": result["absolute_width"],
                "next_measurement": result["next_measurement"],
                "calibration": result["calibration"],
                "evaluation": result,
            }
            evaluations[candidate.candidate_id] = row
            record(
                candidate.candidate_id,
                "EVALUATE",
                round_index,
                "quality_gate_passed" if passed else "quality_gate_failed",
                {"absolute_width": result["absolute_width"], "passed": passed},
            )
            record(
                candidate.candidate_id,
                "RANK",
                round_index,
                "candidate_ranked",
                {
                    "variable_count": result["minimal_variable_count"],
                    "absolute_width": result["absolute_width"],
                    "prior_weight": candidate.prior_weight,
                    "heuristic_likelihood": heuristic_likelihood,
                },
            )

            children = list(candidate_generator.children(candidate, passed))
            dynamic_metadata: dict[str, dict[str, Any]] = {}
            if passed and ablatable_variables:
                ablation_module = __import__(
                    "aie_decision.ablation", fromlist=["plan_variable_ablations"]
                )
                plan = ablation_module.plan_variable_ablations(
                    candidate.formula,
                    parent_candidate_id=candidate.candidate_id,
                )
                for proposed in plan.candidates:
                    if proposed.removed_variable not in ablatable_variables:
                        continue
                    child = CandidateSpec(
                        proposed.candidate_id,
                        proposed.formula,
                        proposed.parent_candidate_id,
                        "ablate",
                        candidate.prior_weight,
                    )
                    children.append(child)
                    dynamic_metadata[child.candidate_id] = {
                        "generation_method": "mechanical_ast_ablation",
                        "rationale": (
                            "remove declared ablatable variable: "
                            f"{proposed.removed_variable}"
                        ),
                        "removed_variable": proposed.removed_variable,
                        "restore_formula": proposed.restore_formula,
                    }
            elif not passed and mutation_templates:
                generation_module = __import__(
                    "aie_decision.candidate_generation",
                    fromlist=["FailureDiagnostic", "generate_candidates"],
                )
                diagnostic = generation_module.FailureDiagnostic.from_evaluation(result)
                generated = generation_module.generate_candidates(
                    candidate,
                    diagnostic,
                    mutation_templates,
                    proposer=proposer,
                    existing_formulas=tuple(known_formulas),
                )
                for proposed in generated:
                    child = CandidateSpec(
                        proposed.candidate_id,
                        proposed.formula,
                        proposed.parent_candidate_id,
                        proposed.mutation_kind,
                        proposed.prior_weight,
                    )
                    children.append(child)
                    dynamic_metadata[child.candidate_id] = {
                        "generation_method": proposed.generation_method,
                        "rationale": proposed.rationale,
                        "template_id": proposed.template_id,
                        "heuristic": proposed.heuristic,
                        "calibrated": proposed.calibrated,
                    }
            next_state = "MINIMIZE" if passed else "EXPAND"
            for child in children:
                if child.candidate_id in queued or child.candidate_id in evaluations:
                    continue
                if child.formula in known_formulas and child.candidate_id not in {
                    item.candidate_id for item in declared
                }:
                    continue
                if len(queued) >= budget.max_candidates:
                    stop_reason = "budget-exhausted"
                    break
                queue.append(child)
                queued.add(child.candidate_id)
                known_formulas.add(child.formula)
                candidate_metadata.setdefault(
                    child.candidate_id,
                    dynamic_metadata.get(
                        child.candidate_id,
                        {
                            "generation_method": "declared",
                            "rationale": "declared child candidate",
                        },
                    ),
                )
                record(
                    child.candidate_id,
                    next_state,
                    round_index,
                    f"activated_by:{candidate.candidate_id}",
                    {
                        "formula": child.formula,
                        "parent_candidate_id": child.parent_candidate_id,
                        "mutation_kind": child.mutation_kind,
                        "prior_weight": child.prior_weight,
                        "generation": candidate_metadata[child.candidate_id],
                    },
                )
        if stop_reason:
            break

    posterior_total = sum(
        float(row.get("unnormalized_pseudo_posterior", 0.0))
        for row in evaluations.values()
    )
    for row in evaluations.values():
        weight = float(row.get("unnormalized_pseudo_posterior", 0.0))
        row["pseudo_posterior"] = weight / posterior_total if posterior_total else 0.0

    passed_rows = [row for row in evaluations.values() if row.get("passed")]
    passed_ablation_parents = {
        row["parent_candidate_id"]
        for row in passed_rows
        if row["mutation_kind"] == "ablate" and row["parent_candidate_id"]
    }
    minimal_rows = [
        row for row in passed_rows if row["candidate_id"] not in passed_ablation_parents
    ]
    minimal_rows.sort(
        key=lambda row: (
            row["variable_count"],
            -row["pseudo_posterior"],
            row["absolute_width"],
            row["candidate_id"],
        )
    )

    if not stop_reason:
        stop_reason = "result-found" if minimal_rows else "insufficient-information"
    selected = (
        minimal_rows[0] if stop_reason == "result-found" and minimal_rows else None
    )
    terminal_candidate = selected["candidate_id"] if selected else ""
    record(
        terminal_candidate, "RESULT" if selected else "STOP", round_index, stop_reason
    )

    ledger_export = ledger.export()
    replay_module = __import__(
        "aie_decision.search_replay", fromlist=["create_search_checkpoint"]
    )
    checkpoint = replay_module.create_search_checkpoint(ledger_export)
    return {
        "schema_version": "fermi-search-loop.v1",
        "run_id": run_id,
        "status": stop_reason,
        "product_acceptance": "provisional_uncalibrated"
        if selected
        else "not_accepted",
        "experimental_usable": selected is not None,
        "usable": False,
        "usable_reason": "calibration is unmeasured"
        if selected
        else "no minimal candidate passed the loop",
        "ranking_method": {
            "name": "bayes_inspired_heuristic_v1",
            "formula": (
                "pseudo_posterior = prior_weight * heuristic_likelihood / sum(weights)"
            ),
            "likelihood_basis": (
                "interval width fit * declared measurement support * "
                "threshold robustness"
            ),
            "calibrated": False,
            "warning": (
                "This score ranks search hypotheses; it is not a statistical "
                "posterior probability."
            ),
        },
        "selected_candidate": selected,
        "evaluations": [evaluations[key] for key in sorted(evaluations)],
        "search_trace": [
            {
                "event_id": event.event_id,
                "candidate_id": event.candidate_id,
                "state": event.state,
                "round_index": event.round_index,
                "reason": event.reason,
                "data": event.data,
            }
            for event in events
        ],
        "budget": {
            "max_candidates": budget.max_candidates,
            "max_rounds": budget.max_rounds,
            "max_evaluations": budget.max_evaluations,
            "max_seconds": budget.max_seconds,
            "rounds_used": round_index,
            "evaluations_used": len(evaluations),
        },
        "ledger": ledger_export,
        "checkpoint": checkpoint,
        "minimality_basis": "declared_ablation_frontier_exhausted"
        if selected
        else None,
    }
