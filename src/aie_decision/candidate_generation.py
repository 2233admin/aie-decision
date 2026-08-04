"""Heuristic candidate mutation for the experimental Fermi search loop.

This module does not infer truth and does not produce calibrated probabilities.
It turns explicit failure diagnostics and caller-declared mutation templates into
deterministic, structured candidate proposals.  A future LLM proposer can replace
the template proposer through :class:`CandidateProposer`, while the validation,
labelling, and deduplication boundary remains local and deterministic.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from string import Formatter
from typing import Any, Protocol

_DIAGNOSTIC_REASONS = {
    "interval_too_wide",
    "next_measurement",
    "missing_variables",
    "invalid_candidate",
}
_MUTATION_KINDS = {"expand", "revise", "ablate"}
_TEMPLATE_FIELDS = {
    "parent_formula",
    "selected_variable",
    "next_measurement",
    "missing_variable",
}


class CandidateLike(Protocol):
    """Minimal structural interface accepted from the search layer."""

    candidate_id: str
    formula: str
    prior_weight: float


@dataclass(frozen=True, slots=True)
class FailureDiagnostic:
    """Mechanical signals emitted by evaluation; no model judgement is implied."""

    reasons: tuple[str, ...]
    next_measurement: str | None = None
    missing_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("diagnostic requires at least one reason")
        unsupported = set(self.reasons) - _DIAGNOSTIC_REASONS
        if unsupported:
            raise ValueError(f"unsupported diagnostic reason: {min(unsupported)}")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("diagnostic reasons must be unique")
        if (
            self.next_measurement is not None
            and not self.next_measurement.isidentifier()
        ):
            raise ValueError("next_measurement must be a valid identifier")
        if any(not item.isidentifier() for item in self.missing_variables):
            raise ValueError("missing_variables must contain valid identifiers")
        if len(set(self.missing_variables)) != len(self.missing_variables):
            raise ValueError("missing_variables must be unique")

    @classmethod
    def from_evaluation(
        cls,
        evaluation: Mapping[str, Any],
        *,
        missing_variables: Sequence[str] = (),
    ) -> FailureDiagnostic:
        """Extract explicit loop signals from an evaluator result."""

        reasons: list[str] = []
        if evaluation.get("within_acceptable_width") is False:
            reasons.append("interval_too_wide")
        next_measurement = evaluation.get("next_measurement")
        if next_measurement is not None:
            if not isinstance(next_measurement, str):
                raise ValueError("evaluation.next_measurement must be a string or null")
            reasons.append("next_measurement")
        missing = tuple(missing_variables)
        if missing:
            reasons.append("missing_variables")
        if not reasons:
            raise ValueError("evaluation contains no supported failure diagnostic")
        return cls(tuple(reasons), next_measurement, missing)


@dataclass(frozen=True, slots=True)
class MutationTemplate:
    """A declared formula rewrite; templates are configuration, not learned facts."""

    template_id: str
    formula_template: str
    diagnostic_reasons: tuple[str, ...]
    mutation_kind: str = "revise"
    prior_multiplier: float = 1.0
    rationale_template: str = "apply {template_id} for {diagnostic_reason}"

    def __post_init__(self) -> None:
        if not self.template_id.strip():
            raise ValueError("template_id is required")
        if not self.formula_template.strip():
            raise ValueError("formula_template is required")
        if self.mutation_kind not in _MUTATION_KINDS:
            raise ValueError(f"unsupported mutation_kind: {self.mutation_kind}")
        if not self.diagnostic_reasons:
            raise ValueError("template requires at least one diagnostic reason")
        unsupported = set(self.diagnostic_reasons) - _DIAGNOSTIC_REASONS
        if unsupported:
            raise ValueError(f"unsupported template diagnostic: {min(unsupported)}")
        if not isfinite(self.prior_multiplier) or self.prior_multiplier <= 0:
            raise ValueError("prior_multiplier must be a positive finite number")
        fields = _format_fields(self.formula_template)
        unknown = fields - _TEMPLATE_FIELDS
        if unknown:
            raise ValueError(f"unsupported formula template field: {min(unknown)}")


@dataclass(frozen=True, slots=True)
class ProposedMutation:
    """Untrusted proposer output before deterministic validation."""

    formula: str
    mutation_kind: str
    template_id: str
    rationale: str
    prior_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    parent_candidate_id: str
    parent_formula: str
    parent_prior_weight: float
    diagnostic: FailureDiagnostic
    templates: tuple[MutationTemplate, ...]


class CandidateProposer(Protocol):
    """Replaceable proposal seam; implementations have proposal power only."""

    def propose(self, request: ProposalRequest) -> Sequence[ProposedMutation]: ...


@dataclass(frozen=True, slots=True)
class GeneratedCandidate:
    """Validated CandidateSpec-like mutation ready for a search-layer adapter."""

    candidate_id: str
    formula: str
    parent_candidate_id: str
    mutation_kind: str
    prior_weight: float
    variable_names: tuple[str, ...]
    template_id: str
    rationale: str
    generation_method: str = "declared_template_heuristic"
    heuristic: bool = True
    calibrated: bool = False

    def as_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "formula": self.formula,
            "parent_candidate_id": self.parent_candidate_id,
            "mutation_kind": self.mutation_kind,
            "prior_weight": self.prior_weight,
            "variable_names": list(self.variable_names),
            "template_id": self.template_id,
            "rationale": self.rationale,
            "generation_method": self.generation_method,
            "heuristic": self.heuristic,
            "calibrated": self.calibrated,
        }


def _format_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field, _, _ in Formatter().parse(template):
        if field:
            fields.add(field)
    return fields


def _formula_identity(formula: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("proposed formula is required")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError("proposed formula must be a valid expression") from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in names:
            names.append(node.id)
    if not names:
        raise ValueError("proposed formula must reference at least one variable")
    return ast.dump(tree, annotate_fields=True, include_attributes=False), tuple(names)


class DeclaredTemplateProposer:
    """Expand only caller-declared templates from mechanical diagnostic slots."""

    def propose(self, request: ProposalRequest) -> tuple[ProposedMutation, ...]:
        proposals: list[ProposedMutation] = []
        diagnostic_reasons = set(request.diagnostic.reasons)
        for template in request.templates:
            matching = [
                reason
                for reason in template.diagnostic_reasons
                if reason in diagnostic_reasons
            ]
            if not matching:
                continue
            for slots in _slot_contexts(template, request):
                formula = template.formula_template.format_map(slots)
                rationale = template.rationale_template.format_map(
                    {
                        **slots,
                        "template_id": template.template_id,
                        "diagnostic_reason": matching[0],
                    }
                )
                proposals.append(
                    ProposedMutation(
                        formula=formula,
                        mutation_kind=template.mutation_kind,
                        template_id=template.template_id,
                        rationale=rationale,
                        prior_multiplier=template.prior_multiplier,
                    )
                )
        return tuple(proposals)


def _slot_contexts(
    template: MutationTemplate, request: ProposalRequest
) -> tuple[dict[str, str], ...]:
    fields = _format_fields(template.formula_template)
    diagnostic = request.diagnostic
    base = {
        "parent_formula": request.parent_formula,
        "next_measurement": diagnostic.next_measurement or "",
        "missing_variable": "",
        "selected_variable": "",
    }
    if "next_measurement" in fields and diagnostic.next_measurement is None:
        return ()
    if "missing_variable" in fields:
        return tuple(
            {**base, "missing_variable": item, "selected_variable": item}
            for item in diagnostic.missing_variables
        )
    if "selected_variable" in fields:
        selected = list(diagnostic.missing_variables)
        if diagnostic.next_measurement and diagnostic.next_measurement not in selected:
            selected.append(diagnostic.next_measurement)
        return tuple({**base, "selected_variable": item} for item in selected)
    return (base,)


def generate_candidates(
    parent: CandidateLike,
    diagnostic: FailureDiagnostic,
    templates: Sequence[MutationTemplate],
    *,
    proposer: CandidateProposer | None = None,
    existing_formulas: Sequence[str] = (),
) -> tuple[GeneratedCandidate, ...]:
    """Generate, validate, label, and semantically deduplicate mutations.

    Even an LLM-backed proposer remains behind this function: its proposals are
    syntax checked and always returned as heuristic, uncalibrated hypotheses.
    """

    if not parent.candidate_id.strip() or not parent.formula.strip():
        raise ValueError("parent candidate id and formula are required")
    if not isfinite(parent.prior_weight) or parent.prior_weight <= 0:
        raise ValueError("parent prior_weight must be a positive finite number")
    request = ProposalRequest(
        parent_candidate_id=parent.candidate_id,
        parent_formula=parent.formula,
        parent_prior_weight=float(parent.prior_weight),
        diagnostic=diagnostic,
        templates=tuple(templates),
    )
    raw = (proposer or DeclaredTemplateProposer()).propose(request)
    seen = {_formula_identity(formula)[0] for formula in existing_formulas}
    generated: list[GeneratedCandidate] = []
    for proposal in raw:
        if proposal.mutation_kind not in _MUTATION_KINDS:
            raise ValueError(
                f"unsupported proposed mutation_kind: {proposal.mutation_kind}"
            )
        if not proposal.template_id.strip():
            raise ValueError("proposed template_id is required")
        if not isfinite(proposal.prior_multiplier) or proposal.prior_multiplier <= 0:
            raise ValueError(
                "proposed prior_multiplier must be a positive finite number"
            )
        identity, variables = _formula_identity(proposal.formula)
        if identity in seen:
            continue
        seen.add(identity)
        signature = (
            f"{parent.candidate_id}\n{proposal.mutation_kind}\n"
            f"{identity}\n{proposal.template_id}"
        )
        digest = sha256(signature.encode("utf-8")).hexdigest()[:16]
        generated.append(
            GeneratedCandidate(
                candidate_id=f"candidate-{digest}",
                formula=proposal.formula.strip(),
                parent_candidate_id=parent.candidate_id,
                mutation_kind=proposal.mutation_kind,
                prior_weight=parent.prior_weight * proposal.prior_multiplier,
                variable_names=variables,
                template_id=proposal.template_id.strip(),
                rationale=proposal.rationale.strip(),
                generation_method=(
                    "declared_template_heuristic"
                    if proposer is None
                    else "external_proposer_heuristic"
                ),
            )
        )
    return tuple(generated)
