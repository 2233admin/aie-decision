"""Auditable question-to-interval estimator for operational throughput.

This is intentionally a narrow product slice.  It accepts prose, extracts only
measurements supported by the grammar below, compares generated decompositions,
and produces a conditional probability interval from an explicit joint model.
Unsupported or weakly specified quantities remain gaps.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from math import isfinite
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence


_NUMBER = r"(?:\d+(?:\.\d+)?)"
_DOMAIN_TERMS = re.compile(r"\b(?:serve|served|process|processed|produce|capacity|throughput)\b", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


@dataclass(frozen=True, slots=True)
class Leaf:
    key: str
    label: str
    unit: str
    p05: float
    p50: float
    p95: float
    distribution: str
    source_id: str
    locator: str
    excerpt: str
    evidence_basis: str

    @property
    def uncertain(self) -> bool:
        return self.p05 != self.p95

    def sample(self, probability: float) -> float:
        if not self.uncertain:
            return self.p50
        # The source explicitly supplies P05/P95. A normal family is the
        # declared interpolation assumption, not additional evidence.
        z90 = NormalDist().inv_cdf(0.95)
        value = self.p50 + ((self.p95 - self.p05) / (2 * z90)) * NormalDist().inv_cdf(probability)
        if self.key == "utilization":
            return min(1.0, max(0.0, value))
        return max(0.0, value)

    def export(self) -> dict[str, Any]:
        return {
            "variable": self.key,
            "label": self.label,
            "unit": self.unit,
            "distribution": self.distribution,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "evidence_basis": self.evidence_basis,
            "provenance": {
                "source_id": self.source_id,
                "locator": self.locator,
                "excerpt": self.excerpt,
            },
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    name: str
    target_kind: str
    expression: str
    variables: tuple[str, ...]
    evaluator: Callable[[Mapping[str, float]], float]
    priority: int


def _as_number(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("extracted measurement must be finite")
    return result


def _point_leaf(
    key: str,
    label: str,
    unit: str,
    value: float,
    source_id: str,
    locator: str,
    excerpt: str,
) -> Leaf:
    return Leaf(key, label, unit, value, value, value, "observed_constant", source_id, locator, excerpt, "stated_observation")


def _interval_leaf(
    key: str,
    label: str,
    unit: str,
    lower: float,
    upper: float,
    source_id: str,
    locator: str,
    excerpt: str,
) -> Leaf:
    if lower > upper:
        lower, upper = upper, lower
    return Leaf(
        key,
        label,
        unit,
        lower,
        (lower + upper) / 2,
        upper,
        "normal_from_stated_p05_p95",
        source_id,
        locator,
        excerpt,
        "source_stated_90_percent_interval",
    )


def _extract(payload: Mapping[str, Any]) -> tuple[dict[str, Leaf], list[dict[str, Any]], list[str]]:
    raw_materials = payload.get("materials")
    if not isinstance(raw_materials, Sequence) or isinstance(raw_materials, (str, bytes)) or not raw_materials:
        raise ValueError("materials must be a non-empty array")
    leaves: dict[str, Leaf] = {}
    conflicts: list[dict[str, Any]] = []
    conflicted_keys: set[str] = set()
    ignored_ranges: list[str] = []
    source_ids: set[str] = set()

    def admit(leaf: Leaf) -> None:
        if leaf.key in conflicted_keys:
            return
        current = leaves.get(leaf.key)
        if current is None:
            leaves[leaf.key] = leaf
        elif (current.p05, current.p95) != (leaf.p05, leaf.p95):
            conflicts.append({
                "variable": leaf.key,
                "first_source": current.source_id,
                "conflicting_source": leaf.source_id,
                "status": "unresolved_conflict",
            })
            leaves.pop(leaf.key, None)
            conflicted_keys.add(leaf.key)

    for material_index, material in enumerate(raw_materials):
        if not isinstance(material, Mapping):
            raise ValueError(f"materials[{material_index}] must be an object")
        source_id = material.get("id")
        text = material.get("text")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"materials[{material_index}].id is required")
        if source_id in source_ids:
            raise ValueError(f"duplicate material id: {source_id}")
        source_ids.add(source_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"materials[{material_index}].text is required")
        for sentence_index, sentence in enumerate(filter(None, (part.strip() for part in _SENTENCE_SPLIT.split(text)))):
            locator = f"sentence:{sentence_index + 1}"
            point_patterns = (
                ("operating_units", "operating units", "units", rf"\b(?:has|uses|operates)\s+({_NUMBER})\s+(?:service\s+)?(?:stations?|machines?|servers?|workers?)\b"),
                ("rate_per_unit_hour", "per-unit hourly rate", "items/unit/hour", rf"\b(?:each\s+(?:station|machine|server|worker)\s+(?:can\s+)?(?:serve|process|handle|produce)s?|rate\s+(?:is|of))\s+({_NUMBER})\s+(?:customers?|orders?|items?|units?|drinks?)\s+per\s+hour\b"),
                ("operating_hours", "operating hours", "hours/day", rf"\b(?:operates?|open)\s+(?:for\s+)?({_NUMBER})\s+hours?\s+per\s+day\b"),
                ("orders_per_customer", "orders per customer", "orders/customer", rf"\b(?:each\s+customer\s+(?:places|makes)|orders?\s+per\s+customer\s+(?:is|of))\s+(?:exactly\s+)?({_NUMBER})\s+orders?\b"),
            )
            for key, label, unit, pattern in point_patterns:
                match = re.search(pattern, sentence, re.I)
                if match:
                    admit(_point_leaf(key, label, unit, _as_number(match.group(1)), source_id, locator, sentence))

            utilization = re.search(
                rf"\butili[sz]ation\b.*?\b90\s*%\s+(?:probability\s+)?interval\b.*?({_NUMBER})\s*%\s*(?:to|-)\s*({_NUMBER})\s*%",
                sentence,
                re.I,
            )
            if utilization:
                admit(_interval_leaf(
                    "utilization", "utilization", "ratio", _as_number(utilization.group(1)) / 100,
                    _as_number(utilization.group(2)) / 100, source_id, locator, sentence,
                ))

            demand = re.search(
                rf"\b(?:daily\s+demand|footfall|visitors?|arrivals?)\b.*?\b90\s*%\s+(?:probability\s+)?interval\b.*?({_NUMBER})\s*(?:to|-)\s*({_NUMBER})\s+(?:customers?|orders?|items?|units?|drinks?)\s+per\s+day",
                sentence,
                re.I,
            )
            if demand:
                admit(_interval_leaf(
                    "daily_demand", "daily demand", "items/day", _as_number(demand.group(1)),
                    _as_number(demand.group(2)), source_id, locator, sentence,
                ))
            elif re.search(r"\b(?:range|between)\b", sentence, re.I) and re.search(rf"{_NUMBER}\s*(?:to|-|and)\s*{_NUMBER}", sentence, re.I):
                ignored_ranges.append(f"{source_id}:{locator}")
    return leaves, conflicts, ignored_ranges


def _candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(
            "served_throughput", "capacity-demand bottleneck", "actual_throughput",
            "min(operating_units * rate_per_unit_hour * operating_hours * utilization, daily_demand)",
            ("operating_units", "rate_per_unit_hour", "operating_hours", "utilization", "daily_demand"),
            lambda v: min(
                v["operating_units"] * v["rate_per_unit_hour"] * v["operating_hours"] * v["utilization"],
                v["daily_demand"],
            ),
            40,
        ),
        Candidate(
            "effective_capacity", "utilization-adjusted capacity", "effective_capacity",
            "operating_units * rate_per_unit_hour * operating_hours * utilization",
            ("operating_units", "rate_per_unit_hour", "operating_hours", "utilization"),
            lambda v: v["operating_units"] * v["rate_per_unit_hour"] * v["operating_hours"] * v["utilization"],
            30,
        ),
        Candidate(
            "demand_volume", "demand-side volume", "demand_volume",
            "daily_demand",
            ("daily_demand",),
            lambda v: v["daily_demand"],
            20,
        ),
        Candidate(
            "theoretical_capacity", "theoretical capacity", "maximum_capacity",
            "operating_units * rate_per_unit_hour * operating_hours",
            ("operating_units", "rate_per_unit_hour", "operating_hours"),
            lambda v: v["operating_units"] * v["rate_per_unit_hour"] * v["operating_hours"],
            10,
        ),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _simulate(
    candidate: Candidate,
    leaves: Mapping[str, Leaf],
    samples: int,
    seed: int,
    dependence: str,
    resolved: str | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    values: list[float] = []
    uncertain = [name for name in candidate.variables if leaves[name].uncertain and name != resolved]
    uncertain_positions = {name: index for index, name in enumerate(uncertain)}
    for _ in range(samples):
        shared = min(1 - 1e-12, max(1e-12, rng.random()))
        assignment: dict[str, float] = {}
        for index, name in enumerate(candidate.variables):
            leaf = leaves[name]
            if name == resolved:
                assignment[name] = leaf.p50
            elif not leaf.uncertain:
                assignment[name] = leaf.p50
            elif dependence == "perfect_positive_rank":
                assignment[name] = leaf.sample(shared)
            elif dependence == "rank_reversed_stress":
                assignment[name] = leaf.sample(shared if uncertain_positions[name] % 2 == 0 else 1 - shared)
            else:
                assignment[name] = leaf.sample(min(1 - 1e-12, max(1e-12, rng.random())))
        values.append(float(candidate.evaluator(assignment)))
    return {
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "width": _quantile(values, 0.95) - _quantile(values, 0.05),
        "sample_count": samples,
        "uncertain_leaf_count": len(uncertain),
    }


def estimate_throughput(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a raw throughput question and source materials into an audit."""
    if not isinstance(payload, Mapping):
        raise ValueError("estimator input must be an object")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    if not _DOMAIN_TERMS.search(question):
        return {
            "schema_version": "throughput-estimate.v1",
            "status": "not_answerable",
            "answerability": "unsupported_domain",
            "question": question.strip(),
            "supported_domain": "operational throughput",
            "gaps": ["question is outside the supported operational-throughput domain"],
        }
    coverage = payload.get("coverage", 0.9)
    if coverage != 0.9:
        raise ValueError("v1 supports only a 0.9 target coverage")
    samples = payload.get("samples", 20000)
    seed = payload.get("seed", 1729)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1000 <= samples <= 100000:
        raise ValueError("samples must be an integer between 1000 and 100000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    leaves, conflicts, ignored_ranges = _extract(payload)
    candidates = _candidates()
    candidate_rows: list[dict[str, Any]] = []
    complete: list[Candidate] = []
    for candidate in candidates:
        missing = [name for name in candidate.variables if name not in leaves]
        row = {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "target_kind": candidate.target_kind,
            "generated_expression": candidate.expression,
            "required_variables": list(candidate.variables),
            "missing_variables": missing,
            "complete": not missing,
            "comparison_score": candidate.priority + (100 if not missing else 0) - 10 * len(missing),
        }
        candidate_rows.append(row)
        if not missing:
            complete.append(candidate)

    selected = next((item for item in complete if item.target_kind == "actual_throughput"), None)
    if selected is None:
        gaps = sorted({name for row in candidate_rows for name in row["missing_variables"]})
        return {
            "schema_version": "throughput-estimate.v1",
            "status": "partial",
            "answerability": "not_answerable",
            "question": question.strip(),
            "input_contract": "question_plus_attributed_materials",
            "extracted_leaves": [leaf.export() for leaf in leaves.values()],
            "decomposition_candidates": candidate_rows,
            "selected_decomposition": None,
            "gaps": gaps,
            "conflicts": conflicts,
            "unqualified_ranges_ignored": ignored_ranges,
            "target_90_interval": None,
        }

    main = _simulate(selected, leaves, samples, seed, "independent")
    stresses = {
        mode: _simulate(selected, leaves, samples, seed, mode)
        for mode in ("perfect_positive_rank", "rank_reversed_stress")
    }

    minimality: list[dict[str, Any]] = []
    for variable in selected.variables:
        ablated_assignment = {
            name: leaves[name].p50 for name in selected.variables if name != variable
        }
        try:
            selected.evaluator(ablated_assignment)
        except KeyError as exc:
            deletion_result = "not_answerable"
            deletion_error = f"missing required leaf: {exc.args[0]}"
        else:
            deletion_result = "answerable"
            deletion_error = None
        substitutes = [
            item.candidate_id for item in complete
            if item.target_kind == selected.target_kind and variable not in item.variables and item.candidate_id != selected.candidate_id
        ]
        minimality.append({
            "variable": variable,
            "intervention": "delete_leaf",
            "ablation_executed": True,
            "answerability_after_deletion": "answerable_by_substitute" if substitutes else deletion_result,
            "deletion_error": deletion_error,
            "same_target_substitutes": substitutes,
            "retained": deletion_result == "not_answerable" and not substitutes,
            "basis": "executed evaluator ablation failed closed" if not substitutes else "same-target substitute available",
        })
    minimal_variables = [row["variable"] for row in minimality if row["retained"]]

    measurement_rows: list[dict[str, Any]] = []
    for variable in selected.variables:
        if not leaves[variable].uncertain:
            continue
        resolved = _simulate(selected, leaves, samples, seed, "independent", resolved=variable)
        narrowing = max(0.0, main["width"] - resolved["width"])
        measurement_rows.append({
            "variable": variable,
            "counterfactual": "resolve_to_current_median",
            "status": "analytic_probe_not_observation",
            "width_if_measured": resolved["width"],
            "expected_narrowing": narrowing,
            "expected_narrowing_fraction": narrowing / main["width"] if main["width"] else 0.0,
        })
    measurement_rows.sort(key=lambda row: (-row["expected_narrowing"], row["variable"]))

    return {
        "schema_version": "throughput-estimate.v1",
        "status": "complete",
        "answerability": "conditionally_answerable",
        "question": question.strip(),
        "supported_domain": "operational throughput",
        "input_contract": "question_plus_attributed_materials",
        "coverage": 0.9,
        "coverage_semantics": "conditional_subjective_probability_interval",
        "calibration": "unmeasured",
        "extracted_leaves": [leaves[name].export() for name in sorted(leaves)],
        "decomposition_candidates": candidate_rows,
        "selected_decomposition": {
            "candidate_id": selected.candidate_id,
            "name": selected.name,
            "target_kind": selected.target_kind,
            "generated_expression": selected.expression,
        },
        "minimal_variable_set": minimal_variables,
        "minimality_basis": "per_leaf_deletion_answerability_test",
        "minimality_tests": minimality,
        "joint_model": {
            "primary_dependence": "independent",
            "status": "explicit_untested_assumption",
            "variables": [name for name in selected.variables if leaves[name].uncertain],
            "marginal_model": "normal interpolated from source-stated P05/P95; observed constants remain fixed",
            "simulation_seed": seed,
            "sample_count": samples,
        },
        "target_90_interval": {"p05": main["p05"], "p50": main["p50"], "p95": main["p95"]},
        "absolute_width": main["width"],
        "interval_method": "joint_monte_carlo_quantiles",
        "dependence_stress_cases": stresses,
        "measurement_priorities": measurement_rows,
        "next_measurement": measurement_rows[0] if measurement_rows else None,
        "assumptions": [
            "the generated bottleneck equation is structurally adequate for the stated operating period",
            "source-stated 90% leaf intervals are represented by normal marginals",
            "uncertain leaves are independent in the primary result",
        ],
        "gaps": [
            "no historical outcomes were supplied, so empirical calibration is unmeasured",
            "the primary independence assumption is not established by the materials",
        ],
        "conflicts": conflicts,
        "unqualified_ranges_ignored": ignored_ranges,
    }
