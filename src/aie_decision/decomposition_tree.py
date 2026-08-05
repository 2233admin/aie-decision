"""Stable public facade for decomposition-tree construction and mutation.

The immutable state record and its export projection live in
:mod:`aie_decision.tree_state`; validation and mutation policy live in their
dedicated modules. This facade retains the original public API.
"""

from __future__ import annotations

from dataclasses import replace

from .fermi_contracts import (
    ActionKind,
    Branch,
    Gap,
    GapKind,
    Node,
    NodeRole,
    NodeStatus,
    Question,
)
from .tree_state import (
    DecompositionError,
    DecompositionState,
    _append_action,
    _now_iso,
)


def _create_root_node(question: Question, node_id: str, *, mechanism: str) -> Node:
    return Node(
        node_id=node_id,
        label=question.target_measure,
        role=NodeRole.TARGET,
        status=NodeStatus.OPEN,
        parent_id=None,
        unit=question.unit,
        scope=question.scope,
        description=question.target_subject,
        mechanism=mechanism,
    )


def create_decomposition(
    question: Question,
    *,
    now: str | None = None,
) -> DecompositionState:
    """Create a fresh decomposition from a raw :class:`Question`.

    The target node is materialised immediately; no formula, variables, or
    domain label is required. The default branch is created alongside the
    question root. Any unresolved fields declared on the question are exposed
    as an incomplete-root gap.
    """

    if not isinstance(question, Question):
        raise DecompositionError("create_decomposition requires a Question instance")
    now = now or _now_iso()
    root_id = "n_0001"
    root = _create_root_node(question, root_id, mechanism="raw question root")
    branch_id = "br_0001"
    branch = Branch(
        branch_id=branch_id,
        root_question_id=question.question_id,
        expansion_ids=(),
        divergent_at_expansion_id=None,
        note="default branch from raw question",
    )
    state = DecompositionState(
        question=question,
        nodes=(root,),
        relationships=(),
        expansions=(),
        branches=(branch,),
        gaps=(),
        actions=(),
        current_branch_id=branch_id,
        pruned_node_ids=frozenset(),
        node_sequence=1,
        relationship_sequence=0,
        expansion_sequence=0,
        branch_sequence=1,
        gap_sequence=0,
        action_sequence=0,
    )
    state = _append_action(
        state,
        kind=ActionKind.CREATE_QUESTION,
        payload={
            "question_id": question.question_id,
            "root_node_id": root_id,
            "timestamp": now,
        },
        result_summary=f"created decomposition root {root_id} for question {question.question_id}",
        accepted=True,
    )
    if not question.is_minimally_complete() or question.unresolved_fields:
        gap = Gap(
            gap_id="gap_0001",
            kind=GapKind.INCOMPLETE_ROOT,
            target=question.question_id,
            explanation=(
                "question root has unresolved or under-specified fields: "
                + ", ".join(question.unresolved_fields or ("scope anchors",))
            ),
            blocking=True,
        )
        state = replace(state, gaps=state.gaps + (gap,), gap_sequence=1)
    return state


def register_gap(
    state: DecompositionState,
    *,
    kind: GapKind,
    target: str,
    explanation: str,
    blocking: bool = True,
) -> DecompositionState:
    """Append a :class:`Gap` to the state and log an action."""

    if not explanation.strip():
        raise DecompositionError("gap explanation is required")
    gap_id = state._next_gap_id()
    gap = Gap(
        gap_id=gap_id,
        kind=kind,
        target=target,
        explanation=explanation.strip(),
        blocking=blocking,
    )
    new_state = replace(
        state,
        gaps=state.gaps + (gap,),
        gap_sequence=state.gap_sequence + 1,
    )
    return _append_action(
        new_state,
        kind=ActionKind.REGISTER_GAP,
        payload={"gap_id": gap_id, "kind": str(kind), "target": target},
        result_summary=f"recorded gap {gap_id} on {target}",
        accepted=True,
    )


# Public tree operations live behind this stable module facade. Import order is
# deliberate: mutation and validation modules depend on the state symbols above.
from .tree_mutation import (
    _reachable_node_ids,
    current_branch_projection,
    expand_state,
    frontier,
    mark_node_unresolved,
    prune,
    pruning_projection,
    propose_alternative,
    propose_atom,
    register_node,
)
from .tree_validation import (
    ChildSpec,
    ExpansionRequest,
    _dimension_key_for,
    evaluate_expansion,
)


__all__ = [
    "ChildSpec",
    "DecompositionError",
    "DecompositionState",
    "ExpansionRequest",
    "current_branch_projection",
    "create_decomposition",
    "evaluate_expansion",
    "expand_state",
    "frontier",
    "mark_node_unresolved",
    "prune",
    "pruning_projection",
    "propose_alternative",
    "propose_atom",
    "register_gap",
    "register_node",
]
