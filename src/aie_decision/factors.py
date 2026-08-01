"""Derived-factor candidate lifecycle, utility ranking, and falsification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence


class FactorStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    RETIRED = "retired"


@dataclass(frozen=True)
class FactorCandidate:
    factor_id: str
    label: str
    contributing_conditions: tuple[str, ...]
    mechanism: str
    observable_implications: tuple[str, ...]
    rejection_conditions: tuple[str, ...]
    status: FactorStatus = FactorStatus.PROPOSED
    revision: int = 1
    independent_evidence: tuple[str, ...] = ()
    latent_variable_treatment: str | None = None
    identifiable: bool = True
    lifecycle_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.factor_id.strip() or not self.label.strip():
            raise ValueError("factor_id and label are required")
        if not self.contributing_conditions or len(set(self.contributing_conditions)) != len(self.contributing_conditions):
            raise ValueError("distinct contributing conditions are required")
        if not self.mechanism.strip():
            raise ValueError("an explicit mechanism or composition rule is required")
        if not self.observable_implications or not self.rejection_conditions:
            raise ValueError("observable implications and rejection conditions are required")


@dataclass(frozen=True)
class FactorScores:
    explanatory_gain: float
    incremental_predictive_value: float
    stability: float
    observability: float
    redundancy: float
    uncertainty: float

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if not 0.0 <= value <= 1.0:
                raise ValueError("factor scores must be between zero and one")


@dataclass(frozen=True)
class RankedFactor:
    candidate: FactorCandidate
    scores: FactorScores
    utility: float


def generate_candidate(
    factor_id: str,
    label: str,
    contributing_conditions: Sequence[str],
    mechanism: str,
    observable_implications: Sequence[str],
    rejection_conditions: Sequence[str],
) -> FactorCandidate:
    return FactorCandidate(
        factor_id,
        label,
        tuple(contributing_conditions),
        mechanism,
        tuple(observable_implications),
        tuple(rejection_conditions),
    )


def assess_identifiability(
    candidate: FactorCandidate, existing_condition_implications: Iterable[str]
) -> FactorCandidate:
    existing = set(existing_condition_implications)
    identifiable = bool(set(candidate.observable_implications) - existing)
    note = "identifiable" if identifiable else "non_identifiable_no_distinguishing_implication"
    return replace(
        candidate,
        identifiable=identifiable,
        revision=candidate.revision + 1,
        lifecycle_notes=candidate.lifecycle_notes + (note,),
    )


def transition_factor(
    candidate: FactorCandidate,
    status: FactorStatus,
    *,
    evidence: Sequence[str] = (),
    latent_variable_treatment: str | None = None,
    note: str,
) -> FactorCandidate:
    if candidate.status is FactorStatus.RETIRED:
        raise ValueError("retired factors cannot transition")
    if status is FactorStatus.SUPPORTED:
        if not candidate.identifiable:
            raise ValueError("a non-identifiable factor cannot be promoted")
        if not evidence and not latent_variable_treatment:
            raise ValueError("support requires independent evidence or declared latent-variable treatment")
    if not note.strip():
        raise ValueError("lifecycle transition requires a note")
    return replace(
        candidate,
        status=status,
        independent_evidence=candidate.independent_evidence + tuple(evidence),
        latent_variable_treatment=latent_variable_treatment or candidate.latent_variable_treatment,
        revision=candidate.revision + 1,
        lifecycle_notes=candidate.lifecycle_notes + (note,),
    )


def rank_candidates(
    candidates: Sequence[tuple[FactorCandidate, FactorScores]],
) -> tuple[RankedFactor, ...]:
    """Rank with exposed components; redundancy and uncertainty are penalties."""
    ranked = []
    for candidate, scores in candidates:
        utility = (
            scores.explanatory_gain * 0.25
            + scores.incremental_predictive_value * 0.25
            + scores.stability * 0.20
            + scores.observability * 0.15
            + (1.0 - scores.redundancy) * 0.10
            + (1.0 - scores.uncertainty) * 0.05
        )
        if not candidate.identifiable:
            utility = 0.0
        ranked.append(RankedFactor(candidate, scores, utility))
    return tuple(sorted(ranked, key=lambda item: (-item.utility, item.candidate.factor_id)))
