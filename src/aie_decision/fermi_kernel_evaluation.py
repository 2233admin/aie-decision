"""Probability evaluation and frontier certification for the Fermi kernel."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .fermi_contracts import MeasurementKind, NodeStatus, ObservationKind
from .fermi_kernel_tree import _mapping, _plain, _rebuild_tree, _required_text
from .frontier import (
    RefinementProposal,
    SaturationEvidence,
    SufficiencyEvidence,
    Tolerance,
    certify_frontier,
    evaluate_necessity,
    evaluate_saturation,
    evaluate_sufficiency,
)
from .probability import (
    ConstantMarginal,
    DependenceCase,
    DistributionFamily,
    JointModel,
    LeafSpec,
    QuantileFittedMarginal,
    TargetSummary,
    UnknownMarginal,
    compile_expression,
    joint_sample,
    rank_width_reduction,
    reducible_uncertainty,
)


def _marginal(document: Mapping[str, Any]):
    kind = _required_text(document, "kind")
    rationale = str(document.get("rationale") or "")
    if kind == "constant":
        return ConstantMarginal(value=float(document["value"]), rationale=rationale)
    if kind == "quantile":
        if not rationale.strip():
            raise ValueError("quantile marginal requires a non-empty evidence/assumption rationale")
        return QuantileFittedMarginal(
            p05=float(document["p05"]),
            p50=float(document["p50"]),
            p95=float(document["p95"]),
            family=DistributionFamily(str(document.get("family") or "normal")),
            rationale=rationale,
        )
    if kind == "unknown":
        domain = document.get("domain") or [0.0, 1.0]
        if not isinstance(domain, (list, tuple)) or len(domain) != 2:
            raise ValueError("unknown marginal domain must contain two endpoints")
        return UnknownMarginal(
            reason=_required_text(document, "reason"),
            domain=(float(domain[0]), float(domain[1])),
            rationale=rationale,
        )
    raise ValueError(f"unsupported marginal kind: {kind}")


def _joint_model(document: Mapping[str, Any] | None) -> JointModel:
    source = dict(document or {})
    return JointModel(
        dependence=DependenceCase(str(source.get("dependence") or "unknown")),
        sample_count=int(source.get("sample_count", 4096)),
        seed=int(source.get("seed", 0)),
        correlation=float(source.get("correlation", 0.7)),
    )


def _active_expression(tree: DecompositionState):
    branch = tree.current_branch()
    active = set(branch.expansion_ids)
    expansion_by_node = {
        item.target_node_id: item
        for item in tree.expansions
        if item.expansion_id in active and not item.is_redundant
    }
    relationship_by_id = {item.relationship_id: item for item in tree.relationships}
    roots = [item for item in tree.nodes if item.parent_id is None]
    if len(roots) != 1:
        raise ValueError("active branch must have exactly one root node")

    visiting: set[str] = set()

    def render(node_id: str) -> str:
        if node_id in visiting:
            raise ValueError(f"cycle detected at node {node_id}")
        expansion = expansion_by_node.get(node_id)
        if expansion is None:
            return node_id
        visiting.add(node_id)
        relationship = relationship_by_id[expansion.relationship_id]
        source = relationship.expression
        for index, child_id in enumerate(relationship.child_node_ids):
            source = re.sub(
                rf"\b_child_{index}\b",
                f"({render(child_id)})",
                source,
            )
        visiting.remove(node_id)
        return f"({source})"

    return compile_expression(render(roots[0].node_id))


def _leaf_specs(tree: DecompositionState, atoms: Mapping[str, Mapping[str, Any]], expression) -> tuple[LeafSpec, ...]:
    result: list[LeafSpec] = []
    for leaf_id in expression.variables:
        node = tree.node(leaf_id)
        if node.status is not NodeStatus.ATOMIC_LEAF:
            raise ValueError(f"leaf {leaf_id} is not operationally atomic")
        atom = atoms.get(leaf_id)
        if atom is None:
            raise ValueError(f"leaf {leaf_id} has no atomic measurement record")
        marginal_doc = _mapping(atom.get("marginal"), f"{leaf_id}.marginal")
        result.append(
            LeafSpec(
                leaf_id=leaf_id,
                marginal=_marginal(marginal_doc),
                unit=str(node.unit or ""),
                measurement_procedure=str(atom.get("procedure") or ""),
            )
        )
    return tuple(result)


def _midpoint(leaf: LeafSpec) -> float:
    marginal = leaf.marginal
    if isinstance(marginal, ConstantMarginal):
        return marginal.value
    if isinstance(marginal, QuantileFittedMarginal):
        return marginal.p50
    lower, upper = marginal.domain
    return (lower + upper) / 2.0


def _tree_depth(tree_document: Mapping[str, Any]) -> int:
    nodes = {str(item["node_id"]): item for item in tree_document.get("nodes", [])}
    maximum = 0
    for node_id in nodes:
        depth = 0
        cursor = nodes[node_id].get("parent_id")
        seen: set[str] = set()
        while cursor and cursor not in seen:
            seen.add(str(cursor))
            depth += 1
            cursor = nodes.get(str(cursor), {}).get("parent_id")
        maximum = max(maximum, depth)
    return maximum


def _summary_output(summary: TargetSummary) -> dict[str, Any]:
    return {
        "probability_interval_valid": summary.probability_interval_valid,
        "coverage_semantics": summary.coverage_semantics,
        "target_interval_90": (
            {"p05": summary.p05, "p50": summary.p50, "p95": summary.p95}
            if summary.probability_interval_valid
            else None
        ),
        "interval_width": summary.width,
        "scenario_bounds": _plain(summary.scenario_bounds),
        "calibration": summary.calibration.value,
        "dependence": summary.dependence.value,
        "correlation": summary.correlation,
        "dependency_gaps": list(summary.dependency_gaps),
        "unknown_leaves": list(summary.unknown_leaves),
        "sample_count": summary.sample_count,
        "seed": summary.seed,
        "method": summary.method,
    }


class EvaluationContext:
    """Everything derived from ``state`` that both ``_evaluate`` and
    ``_frontier_test`` need.  Extracting this guarantees the two entry
    points stay in lockstep on the active branch, expression, leaves,
    joint model, summary, tolerance, and sufficiency evidence.
    """

    __slots__ = ("tree", "atoms", "expression", "leaves", "model", "summary", "tolerance", "sufficiency")

    def __init__(self, *, tree, atoms, expression, leaves, model, summary, tolerance, sufficiency) -> None:
        self.tree = tree
        self.atoms = atoms
        self.expression = expression
        self.leaves = leaves
        self.model = model
        self.summary = summary
        self.tolerance = tolerance
        self.sufficiency = sufficiency


def _build_evaluation_context(state: Mapping[str, Any]) -> EvaluationContext:
    """Rebuild the tree, leaf set, joint model, summary, tolerance and
    sufficiency for the current ``state`` exactly once per kernel call.
    Both ``_evaluate`` and ``_frontier_test`` consume the resulting
    context so they cannot diverge on which tree is active, which
    expression is bound, or which summary gates sufficiency.
    """

    question_doc = _mapping(state.get("question_contract"), "question_contract")
    tree, atoms = _rebuild_tree(
        str(state.get("raw_question") or ""),
        question_doc,
        list(state.get("tree_actions") or []),
    )
    expression = _active_expression(tree)
    leaves = _leaf_specs(tree, atoms, expression)
    model = _joint_model(
        state.get("joint_model") if isinstance(state.get("joint_model"), Mapping) else None
    )
    summary = joint_sample(leaves, expression, model)
    tolerance = Tolerance(acceptable_width=float(question_doc["acceptable_width"]))
    sufficiency = evaluate_sufficiency(summary, tolerance)
    return EvaluationContext(
        tree=tree,
        atoms=atoms,
        expression=expression,
        leaves=leaves,
        model=model,
        summary=summary,
        tolerance=tolerance,
        sufficiency=sufficiency,
    )


def _evaluate(state: Mapping[str, Any]) -> dict[str, Any]:
    context = _build_evaluation_context(state)
    contributions = (
        reducible_uncertainty(
            context.leaves, context.expression, context.model, context.summary
        )
        if context.summary.probability_interval_valid
        else ()
    )
    ranking = rank_width_reduction(contributions)
    next_measurement = _plain(ranking[0]) if ranking else None
    return {
        "expression": context.expression.source,
        "leaf_ids": list(context.expression.variables),
        "summary": _summary_output(context.summary),
        "summary_internal": _plain(context.summary),
        "sufficiency": _plain(context.sufficiency),
        "uncertainty_contributions": _plain(contributions),
        "next_measurement": next_measurement,
    }


def _frontier_test(state: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    context = _build_evaluation_context(state)
    evaluation = {
        "expression": context.expression.source,
        "leaf_ids": list(context.expression.variables),
        "summary": _summary_output(context.summary),
        "summary_internal": _plain(context.summary),
        "sufficiency": _plain(context.sufficiency),
    }
    material_degradation = float(payload.get("material_degradation", 0.1))
    leaves = context.leaves
    expression = context.expression
    model = context.model
    summary = context.summary
    necessity = tuple(
        evaluate_necessity(
            leaf,
            tuple(item for item in leaves if item.leaf_id != leaf.leaf_id),
            expression,
            model,
            summary,
            material_degradation=material_degradation,
            delete=lambda leaf_id, source=leaves: tuple(
                item for item in source if item.leaf_id != leaf_id
            ),
        )
        for leaf in leaves
    ) if summary.probability_interval_valid else ()

    minimum_useful_narrowing = float(payload.get("minimum_useful_narrowing", 0.0))
    contributions = (
        reducible_uncertainty(leaves, expression, model, summary)
        if summary.probability_interval_valid
        else ()
    )
    ranked = rank_width_reduction(contributions)
    emitted = False

    def next_refinement(current: Sequence[LeafSpec], baseline: TargetSummary):
        nonlocal emitted
        if emitted or not ranked:
            return None
        emitted = True
        top = ranked[0]
        leaf = next(item for item in current if item.leaf_id == top.leaf_id)
        resolved = LeafSpec(
            leaf_id=leaf.leaf_id,
            marginal=ConstantMarginal(
                value=_midpoint(leaf),
                rationale="counterfactual exact measurement at declared median",
            ),
            unit=leaf.unit,
            measurement_procedure=leaf.measurement_procedure,
        )
        return RefinementProposal(
            leaf_id=leaf.leaf_id,
            proposed_leaves=(resolved,),
            description="counterfactual exact measurement of highest-value leaf",
            expected_target_narrowing=top.expected_narrowing,
        )

    saturation = (
        evaluate_saturation(
            leaves,
            expression,
            model,
            summary,
            material_improvement_threshold=minimum_useful_narrowing,
            next_refinement=next_refinement,
            max_iterations=2,
        )
        if summary.probability_interval_valid
        else SaturationEvidence(
            saturated=False,
            material_threshold=minimum_useful_narrowing,
            explored=(),
            best_observed_narrowing=0.0,
            notes=("probability_interval_invalid",),
        )
    )
    certificate = certify_frontier(summary, context.sufficiency, necessity, saturation)
    evaluation.update(
        {
            "uncertainty_contributions": _plain(contributions),
            "next_measurement": _plain(ranked[0]) if ranked else None,
            "necessity": _plain(necessity),
            "saturation": _plain(saturation),
            "certificate": {
                "status": certificate.status.value,
                "certified": certificate.certified,
                "reasons": list(certificate.reasons),
                "conditional_on": [
                    "declared leaf marginals",
                    "declared joint dependence",
                    "executed deletion interventions",
                    "highest-value exact-measurement refinement",
                ],
            },
        }
    )
    return evaluation
