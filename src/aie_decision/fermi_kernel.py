"""Deterministic integration kernel for AI-directed Fermi decomposition."""

from __future__ import annotations

from typing import Any, Mapping

from .decomposition_tree import create_decomposition
from .fermi_kernel_evaluation import _evaluate, _frontier_test, _joint_model, _tree_depth
from .fermi_kernel_tree import _clean_tree_export, _mapping, _plain, _question, _rebuild_tree

KERNEL_VERSION = "fermi-kernel.v1"
_TREE_ACTIONS = {"expand", "propose_alternative", "propose_atom"}


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
