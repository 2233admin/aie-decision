"""Deterministic integration kernel for AI-directed Fermi decomposition.

The kernel does not decide what a question means and does not call a model.
An AI supplies semantic actions; this module validates and executes them,
rebuilds the decomposition tree, propagates declared uncertainty, and issues
frontier evidence only after executable sufficiency, necessity, and
conditional-saturation checks.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from .decomposition_tree import (
    ChildSpec,
    DecompositionState,
    ExpansionRequest,
    create_decomposition,
    expand_state,
    propose_alternative,
    propose_atom,
)
from .fermi_contracts import (
    AtomicClaim,
    MeasurementKind,
    NodeStatus,
    ObservationKind,
    Question,
    Scope,
)
from .frontier import (
    RefinementProposal,
    SaturationEvidence,
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


KERNEL_VERSION = "fermi-kernel.v1"
_TREE_ACTIONS = {"expand", "propose_alternative", "propose_atom"}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def _scope(document: Mapping[str, Any]) -> Scope:
    extra = document.get("extra") or {}
    return Scope(
        population=_optional_text(document.get("population")),
        geography=_optional_text(document.get("geography")),
        time_window=_optional_text(document.get("time_window")),
        temporal_basis=_optional_text(document.get("temporal_basis")),
        extra={str(key): str(value) for key, value in dict(extra).items()},
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(document: Mapping[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _question(raw_question: str, document: Mapping[str, Any]) -> Question:
    acceptable_width = document.get("acceptable_width")
    return Question(
        question_id="q_root",
        question=raw_question,
        target_subject=_required_text(document, "target_subject"),
        target_measure=_required_text(document, "target_measure"),
        unit=_required_text(document, "unit"),
        time_basis=_required_text(document, "time_basis"),
        scope=_scope(_mapping(document.get("scope"), "scope")),
        decision_use=_optional_text(document.get("decision_use")),
        acceptable_width=(str(acceptable_width) if acceptable_width is not None else None),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _children(items: Any) -> tuple[ChildSpec, ...]:
    if not isinstance(items, list) or not items:
        raise ValueError("children must be a non-empty list")
    result: list[ChildSpec] = []
    for index, raw in enumerate(items):
        child = _mapping(raw, f"children[{index}]")
        scope_doc = child.get("scope")
        result.append(
            ChildSpec(
                label=_required_text(child, "label"),
                unit=_required_text(child, "unit"),
                scope=_scope(_mapping(scope_doc, f"children[{index}].scope")),
                description=str(child.get("description") or ""),
                mechanism=str(child.get("mechanism") or ""),
            )
        )
    return tuple(result)


def _expansion_request(payload: Mapping[str, Any]) -> ExpansionRequest:
    return ExpansionRequest(
        target_node_id=_required_text(payload, "node_id"),
        parent_unit=_required_text(payload, "parent_unit"),
        expression=_required_text(payload, "expression"),
        rationale=_required_text(payload, "rationale"),
        child_specs=_children(payload.get("children")),
    )


def _atomic_claim(payload: Mapping[str, Any]) -> AtomicClaim:
    return AtomicClaim(
        node_id=_required_text(payload, "node_id"),
        target_object=_required_text(payload, "target_object"),
        unit=_required_text(payload, "unit"),
        scope=_scope(_mapping(payload.get("scope"), "scope")),
        measurement_kind=MeasurementKind(_required_text(payload, "measurement_kind")),
        source=_required_text(payload, "source"),
        procedure=_required_text(payload, "procedure"),
        time_basis=str(payload.get("time_basis") or ""),
        observation_kind=ObservationKind(str(payload.get("observation_kind") or "unknown")),
        assumption_notes=str(payload.get("assumption_notes") or ""),
    )


def _clean_tree_export(tree: DecompositionState) -> dict[str, Any]:
    document = tree.export()
    # The runtime trajectory is the event authority.  Tree action timestamps
    # are intentionally omitted so state digests and replay remain deterministic.
    document.pop("actions", None)
    return document


def _rebuild_tree(
    raw_question: str,
    question_document: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> tuple[DecompositionState, dict[str, Mapping[str, Any]]]:
    tree = create_decomposition(_question(raw_question, question_document), now="1970-01-01T00:00:00+00:00")
    atoms: dict[str, Mapping[str, Any]] = {}
    for record in actions:
        name = str(record.get("name") or "")
        payload = _mapping(record.get("payload"), "action payload")
        if name == "expand":
            before = len(tree.expansions)
            tree = expand_state(tree, request=_expansion_request(payload))
            if len(tree.expansions) == before:
                reason = tree.gaps[-1].explanation if tree.gaps else "expansion rejected"
                raise ValueError(reason)
        elif name == "propose_alternative":
            before = len(tree.expansions)
            tree = propose_alternative(tree, request=_expansion_request(payload))
            if len(tree.expansions) == before:
                reason = tree.gaps[-1].explanation if tree.gaps else "alternative rejected"
                raise ValueError(reason)
        elif name == "propose_atom":
            node_id = _required_text(payload, "node_id")
            tree = propose_atom(tree, node_id=node_id, claim=_atomic_claim(payload))
            if tree.node(node_id).status is not NodeStatus.ATOMIC_LEAF:
                reason = tree.gaps[-1].explanation if tree.gaps else "atomic claim rejected"
                raise ValueError(reason)
            atoms[node_id] = payload
        else:
            raise ValueError(f"unsupported tree action: {name}")
    return tree, atoms


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


def _evaluate(state: Mapping[str, Any]) -> dict[str, Any]:
    question_doc = _mapping(state.get("question_contract"), "question_contract")
    tree, atoms = _rebuild_tree(
        str(state.get("raw_question") or ""),
        question_doc,
        list(state.get("tree_actions") or []),
    )
    expression = _active_expression(tree)
    leaves = _leaf_specs(tree, atoms, expression)
    model = _joint_model(state.get("joint_model") if isinstance(state.get("joint_model"), Mapping) else None)
    summary = joint_sample(leaves, expression, model)
    tolerance = Tolerance(acceptable_width=float(question_doc["acceptable_width"]))
    sufficiency = evaluate_sufficiency(summary, tolerance)
    contributions = (
        reducible_uncertainty(leaves, expression, model, summary)
        if summary.probability_interval_valid
        else ()
    )
    ranking = rank_width_reduction(contributions)
    next_measurement = _plain(ranking[0]) if ranking else None
    return {
        "expression": expression.source,
        "leaf_ids": list(expression.variables),
        "summary": _summary_output(summary),
        "summary_internal": _plain(summary),
        "sufficiency": _plain(sufficiency),
        "uncertainty_contributions": _plain(contributions),
        "next_measurement": next_measurement,
    }


def _frontier_test(state: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _evaluate(state)
    question_doc = _mapping(state.get("question_contract"), "question_contract")
    tree, atoms = _rebuild_tree(
        str(state.get("raw_question") or ""),
        question_doc,
        list(state.get("tree_actions") or []),
    )
    expression = _active_expression(tree)
    leaves = _leaf_specs(tree, atoms, expression)
    model = _joint_model(state.get("joint_model") if isinstance(state.get("joint_model"), Mapping) else None)
    summary = joint_sample(leaves, expression, model)
    tolerance = Tolerance(acceptable_width=float(question_doc["acceptable_width"]))
    sufficiency = evaluate_sufficiency(summary, tolerance)
    material_degradation = float(payload.get("material_degradation", 0.1))
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
    certificate = certify_frontier(summary, sufficiency, necessity, saturation)
    return {
        **evaluation,
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


class FermiKernel:
    """Model-free kernel implementing the :class:`KernelProtocol` surface."""

    def initial_state(self, question: str) -> dict[str, Any]:
        return {
            "kernel_version": KERNEL_VERSION,
            "raw_question": question,
            "question_contract": None,
            "tree_actions": [],
            "tree": None,
            "joint_model": None,
            "evaluation": None,
            "frontier_test": None,
            "depth": 0,
        }

    def action_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "define_question",
                "purpose": "Turn the raw question into an explicit target without supplying leaves or a formula in advance.",
                "required_fields": ["target_subject", "target_measure", "unit", "time_basis", "scope", "acceptable_width"],
                "field_contract": {
                    "scope": {"population": "string", "geography": "string|null", "time_window": "string|null"},
                    "acceptable_width": "non-negative absolute width in the target unit",
                },
            },
            {
                "name": "expand",
                "purpose": "Propose and execute a measurable identity for one frontier node.",
                "required_fields": ["node_id", "parent_unit", "expression", "rationale", "children"],
                "field_contract": {
                    "expression": "restricted arithmetic using _child_0, _child_1, ...",
                    "children": [{"label": "string", "unit": "compound unit", "scope": "scope object", "description": "string", "mechanism": "string"}],
                    "common_unit_symbols": ["1", "person", "transaction", "order", "event", "trip", "day", "h", "kg", "litre", "CNY", "USD"],
                },
            },
            {
                "name": "propose_alternative",
                "purpose": "Propose a different identity for a node; redundant identities are rejected structurally.",
                "required_fields": ["node_id", "parent_unit", "expression", "rationale", "children"],
                "field_contract": {"same_payload_as": "expand"},
            },
            {
                "name": "propose_atom",
                "purpose": "Attach a structured measurement procedure and honest marginal to one leaf.",
                "required_fields": ["node_id", "target_object", "unit", "scope", "measurement_kind", "source", "procedure", "marginal"],
                "field_contract": {
                    "measurement_kind": ["direct_observation", "record_lookup", "count", "timed_measurement", "instrument_measurement", "derived_proxy"],
                    "observation_kind": ["observed", "estimated", "unknown"],
                    "marginal": {
                        "constant": {"kind": "constant", "value": "number", "rationale": "string"},
                        "quantile": {"kind": "quantile", "p05": "number", "p50": "number", "p95": "number", "family": "normal|lognormal", "rationale": "evidence and assumptions"},
                        "unknown": {"kind": "unknown", "reason": "string", "domain": ["lower", "upper"]},
                    },
                },
            },
            {
                "name": "set_dependence",
                "purpose": "Declare the joint dependence assumption used for probability propagation.",
                "required_fields": ["dependence", "rationale"],
                "field_contract": {"dependence": ["independent", "positive", "negative", "unknown"], "sample_count": "positive integer", "seed": "integer", "correlation": "single equicorrelation applied to every non-constant leaf; number strictly between -1 and 1", "rationale": "why this global joint assumption is defensible and what it omits"},
            },
            {
                "name": "evaluate",
                "purpose": "Compile the active branch and propagate its declared joint uncertainty.",
                "required_fields": [],
            },
            {
                "name": "test_frontier",
                "purpose": "Execute sufficiency, per-leaf deletion, and best-next-measurement saturation tests.",
                "required_fields": ["material_degradation", "minimum_useful_narrowing"],
                "field_contract": {
                    "material_degradation": "fractional width degradation threshold for necessity",
                    "minimum_useful_narrowing": "absolute target-width reduction below which further measurement is not material",
                },
            },
            {"name": "rollback", "purpose": "Project a prior accepted semantic action out of live state.", "required_fields": ["target_sequence"]},
            {"name": "finalize", "purpose": "Return the already-executed frontier certificate.", "required_fields": []},
        ]

    def validate(self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
        allowed = {item["name"] for item in self.action_specs()}
        if action not in allowed:
            return [{"code": "unknown_action", "path": "$.action", "message": f"unsupported action: {action}"}]
        if action == "define_question" and state.get("question_contract") is not None:
            return [{"code": "illegal_sequence", "path": "$.action", "message": "question is already defined; rollback before redefining"}]
        if action in _TREE_ACTIONS | {"set_dependence", "evaluate", "test_frontier"} and state.get("question_contract") is None:
            return [{"code": "illegal_sequence", "path": "$.action", "message": "define_question must be accepted first"}]
        required = next(item["required_fields"] for item in self.action_specs() if item["name"] == action)
        issues = []
        for name in required:
            if name not in payload:
                issues.append({"code": "missing_field", "path": f"$.{name}", "message": f"{name} is required"})
        return issues

    def execute(self, action: str, payload: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        result = _plain(state)
        result["evaluation"] = None
        result["frontier_test"] = None
        if action == "define_question":
            question = _question(str(state.get("raw_question") or ""), payload)
            tree = create_decomposition(question, now="1970-01-01T00:00:00+00:00")
            result["question_contract"] = _plain(payload)
            result["tree"] = _clean_tree_export(tree)
            result["depth"] = 0
        elif action in _TREE_ACTIONS:
            records = list(result.get("tree_actions") or [])
            records.append({"name": action, "payload": _plain(payload)})
            tree, _ = _rebuild_tree(
                str(state.get("raw_question") or ""),
                _mapping(result.get("question_contract"), "question_contract"),
                records,
            )
            result["tree_actions"] = records
            result["tree"] = _clean_tree_export(tree)
            result["depth"] = _tree_depth(result["tree"])
        elif action == "set_dependence":
            _joint_model(payload)
            result["joint_model"] = _plain(payload)
        elif action == "evaluate":
            result["evaluation"] = _evaluate(result)
        elif action == "test_frontier":
            tested = _frontier_test(result, payload)
            result["evaluation"] = {key: value for key, value in tested.items() if key not in {"necessity", "saturation", "certificate"}}
            result["frontier_test"] = tested
        return result

    def legal_next_actions(self, state: Mapping[str, Any]) -> list[str]:
        if state.get("question_contract") is None:
            return ["define_question"]
        actions = ["expand", "propose_alternative", "propose_atom", "set_dependence", "evaluate", "rollback"]
        if state.get("evaluation") is not None:
            actions.append("test_frontier")
        certificate = ((state.get("frontier_test") or {}).get("certificate") or {})
        if certificate.get("certified"):
            actions.append("finalize")
        return actions

    def active_frontier(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        tree = state.get("tree")
        if not isinstance(tree, Mapping):
            if state.get("question_contract") is not None:
                return [{"node_id": "n_0001", "label": str(state.get("raw_question") or ""), "status": "open"}]
            return [{"node_id": "raw_question", "label": str(state.get("raw_question") or ""), "status": "needs_question_definition"}]
        frontier = [dict(item) for item in tree.get("frontier", []) if isinstance(item, Mapping)]
        ranking = ((state.get("evaluation") or {}).get("uncertainty_contributions") or [])
        priority = {str(item.get("leaf_id")): item for item in ranking if isinstance(item, Mapping)}
        for item in frontier:
            if str(item.get("node_id")) in priority:
                item["uncertainty_priority"] = priority[str(item["node_id"])]
        return frontier

    def evaluate_frontier(self, state: Mapping[str, Any]) -> dict[str, Any]:
        tested = state.get("frontier_test")
        if not isinstance(tested, Mapping):
            return {"status": "insufficient", "reasons": ["frontier tests have not been executed"]}
        certificate = tested.get("certificate") or {}
        return {
            "status": "certified" if certificate.get("certified") else "insufficient",
            "reasons": list(certificate.get("reasons") or []),
            "target_interval_90": (tested.get("summary") or {}).get("target_interval_90"),
            "interval_width": (tested.get("summary") or {}).get("interval_width"),
            "next_measurement": tested.get("next_measurement"),
            "certificate": certificate,
        }


def build() -> FermiKernel:
    return FermiKernel()


__all__ = ["FermiKernel", "KERNEL_VERSION", "build"]
