"""Tree-input parsing and reconstruction for the integrated Fermi kernel."""

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


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        # ``frozenset`` iteration order is unspecified and would silently
        # change state digests across runs; render in deterministic order
        # by sorting on the string form of each element.
        items = sorted((_plain(item) for item in value), key=_digest_sort_key)
        return list(items)
    if isinstance(value, set):
        # ``set`` iteration order is unspecified for the same reason; sort
        # before converting so state exports remain stable.
        items = sorted((_plain(item) for item in value), key=_digest_sort_key)
        return list(items)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest_sort_key(value: Any) -> str:
    """Stable string key used to order heterogeneous digest elements."""

    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive, str() rarely fails
        return repr(value)


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
