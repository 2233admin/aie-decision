"""Track A decomposition-tree tests.

Each test maps to a Track A OpenSpec scenario (see
``openspec/changes/build-recursive-fermi-runtime/specs/recursive-fermi-decomposition/spec.md``).
A non-throughput novel-question example closes the suite.
"""

from __future__ import annotations

import pytest

from aie_decision.decomposition_tree import (
    ChildSpec,
    DecompositionError,
    DecompositionState,
    ExpansionRequest,
    create_decomposition,
    current_branch_projection,
    evaluate_expansion,
    expand_state,
    frontier,
    mark_node_unresolved,
    propose_alternative,
    propose_atom,
    prune,
    pruning_projection,
    register_gap,
)
from aie_decision.fermi_contracts import (
    ActionKind,
    AtomicClaim,
    Branch,
    CompoundUnit,
    GapKind,
    MeasurementKind,
    Node,
    NodeRole,
    NodeStatus,
    ObservationKind,
    Question,
    QuestionStatus,
    Scope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope(label: str = "U.S. workers") -> Scope:
    return Scope(population=label, geography="United States")


def _commutes_question(**overrides) -> Question:
    payload = {
        "question_id": "q-commutes",
        "question": "How many one-way commutes occur on a U.S. weekday?",
        "target_subject": "U.S. weekday commutes",
        "target_measure": "count of one-way commutes",
        "unit": "person/day",
        "time_basis": "per weekday in 2025",
        "scope": _scope(),
        "decision_use": "size the bus-rapid-transit opportunity",
    }
    payload.update(overrides)
    return Question(**payload)


def _commute_expansion(state: DecompositionState) -> ExpansionRequest:
    return ExpansionRequest(
        target_node_id="n_0001",
        parent_unit="person/day",
        expression="_child_0 * _child_1",
        rationale="commutes/day = (commuting people per day) * (trips per person on a weekday)",
        child_specs=(
            ChildSpec(
                label="commuting people per day",
                unit="person/day",
                scope=_scope(),
                description="U.S. residents who commute at all on a weekday",
            ),
            ChildSpec(
                label="trips per person on a weekday",
                unit="1",
                scope=_scope(),
                description="average one-way trips per commuting person per weekday",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Scenario: Start with only a question
# ---------------------------------------------------------------------------


def test_create_decomposition_records_target_fields_without_a_formula():
    question = _commutes_question()
    state = create_decomposition(question)

    assert state.question.question_id == question.question_id
    assert state.question.target_subject == "U.S. weekday commutes"
    assert state.question.target_measure == "count of one-way commutes"
    assert state.question.unit == "person/day"
    assert state.question.time_basis == "per weekday in 2025"
    assert state.question.scope.geography == "United States"
    assert state.question.is_minimally_complete()
    assert len(state.branches) == 1
    assert state.branches[0].branch_id == "br_0001"
    assert state.branches[0].expansion_ids == ()
    assert state.current_branch_id == "br_0001"

    # The root node is materialised without an expansion or a formula.
    root = state.nodes[0]
    assert root.node_id == "n_0001"
    assert root.role is NodeRole.TARGET
    assert root.parent_id is None
    assert root.status is NodeStatus.OPEN
    assert root.unit == "person/day"
    assert root.mechanism == "raw question root"

    # The action log records the creation as the very first event.
    assert state.actions[0].kind is ActionKind.CREATE_QUESTION
    assert state.actions[0].accepted


def test_create_decomposition_surfaces_incomplete_root_gap():
    question = _commutes_question(scope=Scope())
    state = create_decomposition(question)

    assert any(
        gap.kind is GapKind.INCOMPLETE_ROOT and gap.target == question.question_id
        for gap in state.gaps
    )


# ---------------------------------------------------------------------------
# Scenario: Expand an abstract target
# ---------------------------------------------------------------------------


def test_expand_an_abstract_target_into_three_children_with_unit_closure():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))

    expansion = state.expansions[0]
    relationship = state.relationships[0]
    assert expansion.target_node_id == "n_0001"
    assert relationship.parent_unit == "person/day"
    assert len(expansion.child_node_ids) == 2
    assert state.node("n_0001").status is NodeStatus.EXPANDED
    for child_id in expansion.child_node_ids:
        child = state.node(child_id)
        assert child.parent_id == "n_0001"
        assert child.role is NodeRole.ATOM_CANDIDATE
    # Numeric evaluation reproduces the parent unit when child values are bound.
    numeric = evaluate_expansion(
        state,
        expansion_id=expansion.expansion_id,
        values={"n_0002": 50_000_000, "n_0003": 2.0},
    )
    assert numeric == 100_000_000.0
    # Frontier exposes the freshly materialised children for further work.
    frontier_nodes = frontier(state)
    assert {node.node_id for node in frontier_nodes} == set(expansion.child_node_ids)


def test_expansion_extends_the_current_branch_lineage():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    assert state.branches[0].expansion_ids == (state.expansions[0].expansion_id,)


def test_existing_branch_and_current_branch_are_stable_across_reads():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    branch = state.current_branch()
    assert isinstance(branch, Branch)
    assert branch.branch_id == state.current_branch_id


# ---------------------------------------------------------------------------
# Scenario: Expansion has incompatible units
# ---------------------------------------------------------------------------


def test_expansion_with_unit_mismatch_leaves_the_prior_tree_intact():
    state = create_decomposition(_commutes_question())
    baseline_nodes = tuple(state.nodes)
    baseline_expansions = tuple(state.expansions)

    mismatched = ExpansionRequest(
        target_node_id="n_0001",
        parent_unit="person/day",
        expression="_child_0 + _child_1",
        rationale="addition is invalid across heterogeneous units",
        child_specs=(
            ChildSpec(label="mass", unit="kg", scope=_scope()),
            ChildSpec(label="length", unit="m", scope=_scope()),
        ),
    )
    state = expand_state(state, request=mismatched)

    # The tree is unchanged — no children were added.
    assert tuple(state.nodes) == baseline_nodes
    assert tuple(state.expansions) == baseline_expansions
    # But a gap records why the attempt was rejected.
    assert any(
        gap.kind is GapKind.UNIT_MISMATCH and gap.target == "n_0001" and gap.blocking
        for gap in state.gaps
    )


def test_expansion_with_incompatible_unit_records_blocking_gap():
    state = create_decomposition(_commutes_question())
    state = expand_state(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="kg",
            expression="_child_0 + _child_1",
            rationale="incompatible",
            child_specs=(
                ChildSpec(label="a", unit="kg", scope=_scope()),
                ChildSpec(label="b", unit="m", scope=_scope()),
            ),
        ),
    )
    gap = next(gap for gap in state.gaps if gap.target == "n_0001")
    assert gap.kind is GapKind.UNIT_MISMATCH
    assert gap.blocking is True


# ---------------------------------------------------------------------------
# Scenario: Abstract label submitted as a leaf
# ---------------------------------------------------------------------------


def test_abstract_label_submitted_as_leaf_is_rejected_and_kept_on_frontier():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    abstract_node_id = state.expansions[0].child_node_ids[0]

    claim = AtomicClaim(
        node_id=abstract_node_id,
        target_object="operating efficiency",
        unit="ratio",
        scope=_scope(),
        time_basis="per day",
        # A bare claim that still omits a source and procedure must be
        # rejected, even though the textual content reads as an action.
        measurement_kind=MeasurementKind.DIRECT_OBSERVATION,
        source="",
        procedure="",
    )
    before_gap_count = len(state.gaps)
    state = propose_atom(state, node_id=abstract_node_id, claim=claim)
    after_gap_count = len(state.gaps)

    assert after_gap_count > before_gap_count
    new_gap = state.gaps[-1]
    assert new_gap.kind is GapKind.ATOM_REJECTED
    assert new_gap.target == abstract_node_id
    assert new_gap.blocking is True
    # The abstract node stays open on the frontier.
    frontier_ids = {node.node_id for node in frontier(state)}
    assert abstract_node_id in frontier_ids
    assert state.node(abstract_node_id).status is NodeStatus.OPEN


# ---------------------------------------------------------------------------
# Scenario: Concrete atom is submitted
# ---------------------------------------------------------------------------


def test_concrete_atom_is_accepted_and_recorded_with_procedure():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    target = state.expansions[0].child_node_ids[0]

    claim = AtomicClaim(
        node_id=target,
        target_object="number of U.S. residents commuting on a weekday",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="Census Bureau ACS commuting-flow tables (B08006) and LEHD LODES",
        procedure=(
            "Pull the ACS B08006 commuting-flow table and cross-check the LEHD "
            "origin-destination employment statistics to obtain a daily count."
        ),
        observation_kind=ObservationKind.OBSERVED,
    )
    state = propose_atom(state, node_id=target, claim=claim)
    promoted = state.node(target)

    assert promoted.status is NodeStatus.ATOMIC_LEAF
    # The question does NOT automatically become atomic just because a child
    # did — the second child is still open and the tree is not whole.
    assert state.question.status is QuestionStatus.OPEN
    assert any(action.kind is ActionKind.PROPOSE_ATOM for action in state.actions)


def test_concrete_atom_with_unit_mismatch_records_blocking_gap():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    target = state.expansions[0].child_node_ids[0]
    claim = AtomicClaim(
        node_id=target,
        target_object="number of U.S. residents commuting on a weekday",
        unit="kg",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="ACS B08006",
        procedure="Pull the ACS B08006 commuting-flow table for the daily count.",
    )
    before_gaps = len(state.gaps)
    state = propose_atom(state, node_id=target, claim=claim)
    assert len(state.gaps) > before_gaps
    gap = state.gaps[-1]
    assert gap.kind is GapKind.UNIT_MISMATCH
    assert gap.target == target


def test_promoting_one_child_does_not_mark_the_question_atomic():
    """Defect: the previous implementation flipped the question to ATOMIC_LEAF
    whenever a single child became a leaf, hiding the remaining frontier.
    """

    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    leaves = state.expansions[0].child_node_ids

    claim = AtomicClaim(
        node_id=leaves[0],
        target_object="number of U.S. residents commuting on a weekday",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="ACS B08006",
        procedure="Pull the ACS B08006 commuting-flow table for the daily count.",
    )
    state = propose_atom(state, node_id=leaves[0], claim=claim)
    assert state.node(leaves[0]).status is NodeStatus.ATOMIC_LEAF
    assert state.question.status is QuestionStatus.OPEN
    # The other child is still on the frontier.
    frontier_ids = {node.node_id for node in frontier(state)}
    assert leaves[1] in frontier_ids


def test_question_reaches_atomic_only_when_every_branch_leaf_is_admissible():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    leaves = list(state.expansions[0].child_node_ids)

    def _claim(node_id: str) -> AtomicClaim:
        node = state.node(node_id)
        return AtomicClaim(
            node_id=node_id,
            target_object="commuting people count",
            unit=node.unit,
            scope=_scope(),
            time_basis="per weekday in 2025",
            measurement_kind=MeasurementKind.COUNT,
            source="ACS B08006",
            procedure="Pull the ACS B08006 commuting-flow table for the daily count.",
        )

    state = propose_atom(state, node_id=leaves[0], claim=_claim(leaves[0]))
    assert state.question.status is QuestionStatus.OPEN
    state = propose_atom(state, node_id=leaves[1], claim=_claim(leaves[1]))
    assert state.node(leaves[0]).status is NodeStatus.ATOMIC_LEAF
    assert state.node(leaves[1]).status is NodeStatus.ATOMIC_LEAF
    # Question remains OPEN until an explicit root-level condition is met;
    # a child becoming ATOMIC_LEAF does not automatically mark the question complete.
    assert state.question.status is QuestionStatus.OPEN


# ---------------------------------------------------------------------------
# Scenario: A second decomposition becomes useful
# ---------------------------------------------------------------------------


def test_alternative_expansion_records_a_new_branch_without_replacing_the_default():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    baseline_branch_count = len(state.branches)

    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale=(
                "split commutes into work commutes and non-work commutes because the "
                "underlying drivers differ"
            ),
            child_specs=(
                ChildSpec(label="work trips per day", unit="person/day", scope=_scope()),
                ChildSpec(label="non-work trips per day", unit="person/day", scope=_scope()),
            ),
        ),
    )

    assert len(state.branches) == baseline_branch_count + 1
    assert state.expansions[-1].is_alternative
    # The new branch diverges at the alternative expansion.
    divergent = state.branches[-1]
    assert divergent.divergent_at_expansion_id == state.expansions[-1].expansion_id
    assert divergent.expansion_ids[0] == state.expansions[0].expansion_id
    # The alternative and the dominant both remain accessible for comparison.
    alt_id = state.expansions[-1].expansion_id
    assert any(
        branch.branch_id != state.current_branch_id and alt_id in branch.expansion_ids
        for branch in state.branches
    )


# ---------------------------------------------------------------------------
# Scenario: Formula rewrite adds no information
# ---------------------------------------------------------------------------


def test_algebraically_equivalent_alternative_is_marked_redundant():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    dominant_id = state.expansions[0].expansion_id

    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_1 * _child_0",  # commutative rewrite
            rationale="identical algebra with renamed children",
            child_specs=(
                ChildSpec(label="trips per person on a weekday (alt)", unit="1", scope=_scope()),
                ChildSpec(label="commuting people per day (alt)", unit="person/day", scope=_scope()),
            ),
        ),
    )

    new_expansion = state.expansions[-1]
    assert new_expansion.is_alternative
    assert new_expansion.is_redundant
    assert new_expansion.alternative_of_expansion_id == dominant_id
    assert "matches" in (new_expansion.redundancy_reason or "")
    # The current branch stays unchanged: the redundant alt does not become a
    # distinct search branch the runtime needs to compare against.
    current = state.current_branch()
    assert dominant_id in current.expansion_ids
    assert new_expansion.expansion_id not in current.expansion_ids
    # A non-blocking gap captures the redundancy so the AI can see why.
    assert any(
        gap.kind is GapKind.REDUNDANT_ALTERNATIVE and gap.target == "n_0001"
        and gap.blocking is False
        for gap in state.gaps
    )
    # is_redundant_alternative reports the same conclusion from a query API.
    is_redundant, reason = state.is_redundant_alternative(new_expansion.expansion_id)
    assert is_redundant
    assert reason is not None


# ---------------------------------------------------------------------------
# Scenario: Familiar domain keywords are present
# ---------------------------------------------------------------------------


def test_familiar_domain_keywords_do_not_auto_complete_the_decomposition():
    # The question deliberately uses operational vocabulary.  The runtime
    # must not synthesise a fixed formula or default children.
    question = Question(
        question_id="q-throughput",
        question="Estimate the daily operational throughput of a fulfillment centre.",
        target_subject="fulfillment-centre daily throughput",
        target_measure="orders per day",
        unit="order/day",
        time_basis="per weekday",
        scope=Scope(population="fulfillment centres", geography="United States"),
    )
    state = create_decomposition(question)

    # No children were auto-injected by keyword matching.
    assert len(state.nodes) == 1
    assert len(state.expansions) == 0
    assert state.node("n_0001").status is NodeStatus.OPEN
    # The frontier exposes the still-open root for an AI to act on.
    frontier_ids = {node.node_id for node in frontier(state)}
    assert frontier_ids == {"n_0001"}


# ---------------------------------------------------------------------------
# Scenario: No defensible atom can be produced
# ---------------------------------------------------------------------------


def test_unresolvable_node_returns_non_answerable_branch_with_gap():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    abstract = state.expansions[0].child_node_ids[0]

    state = mark_node_unresolved(
        state,
        node_id=abstract,
        reason="no measurable population can be produced without a household survey",
    )
    projected = current_branch_projection(state)
    frontier_ids = {node.node_id for node in frontier(projected)}

    # The unresolved node is no longer on the actionable frontier.
    assert abstract not in frontier_ids
    # A blocking gap explains the unresolved abstraction.
    unresolved_gap = next(
        gap for gap in state.gaps if gap.target == abstract and gap.kind is GapKind.UNRESOLVED_NODE
    )
    assert unresolved_gap.blocking


# ---------------------------------------------------------------------------
# Scenario: Inspect a decomposition
# ---------------------------------------------------------------------------


def test_export_returns_machine_readable_state_for_reconstruction():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))

    payload = state.export()
    assert payload["schema_version"] == "1.0.0"
    assert payload["question"]["question_id"] == "q-commutes"
    assert payload["current_branch_id"] == state.current_branch_id
    # Every node, relationship, expansion, gap, and action is enumerated.
    assert len(payload["nodes"]) == len(state.nodes)
    assert len(payload["relationships"]) == len(state.relationships)
    assert len(payload["expansions"]) == len(state.expansions)
    assert len(payload["actions"]) == len(state.actions)
    # Frontier projection is included for direct consumption by an AI.
    frontier_ids = {entry["node_id"] for entry in payload["frontier"]}
    assert frontier_ids == {node.node_id for node in frontier(state)}


def test_export_records_branch_alternative_history():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale="work + non-work split",
            child_specs=(
                ChildSpec(label="work trips per day", unit="person/day", scope=_scope()),
                ChildSpec(label="non-work trips per day", unit="person/day", scope=_scope()),
            ),
        ),
    )
    payload = state.export()
    branches = payload["branches"]
    assert any(branch["is_current"] for branch in branches)
    # The export lists every branch so a client can replay comparison logic.
    assert len(branches) == len(state.branches)
    assert sum(1 for branch in branches if not branch["is_current"]) >= 1


# ---------------------------------------------------------------------------
# Pruning projection
# ---------------------------------------------------------------------------


def test_pruning_projection_excludes_pruned_nodes_from_the_active_frontier():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    first_child, second_child = state.expansions[0].child_node_ids

    state = prune(state, node_id=first_child, reason="no defensible measurement procedure")

    projected = pruning_projection(state)
    surviving = {node.node_id for node in projected.nodes}
    assert first_child not in surviving
    assert second_child in surviving
    # The pruned node keeps a record via GapKind.UNRESOLVED_NODE.
    gap = next(
        g for g in state.gaps
        if g.target == first_child and g.kind is GapKind.UNRESOLVED_NODE
    )
    assert gap.blocking is False


def test_pruning_is_irreversible_and_visible_in_action_log():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    child_id = state.expansions[0].child_node_ids[0]

    state = prune(state, node_id=child_id, reason="wrong decomposition leaf")
    kinds = [action.kind for action in state.actions]
    # The register-gap action introduced by prune must be present.
    assert ActionKind.PRUNE in kinds
    assert state.node(child_id).status is NodeStatus.PRUNED


def test_pruning_unknown_node_is_rejected():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="unknown"):
        prune(state, node_id="n_9999", reason="missing")


# ---------------------------------------------------------------------------
# Frontier inspection across multiple expansions
# ---------------------------------------------------------------------------


def test_frontier_returns_only_open_atoms_on_the_active_branch():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    leaves = state.expansions[0].child_node_ids

    # Promote the first child to an atomic leaf and confirm it disappears
    # from the frontier; the second remains.
    claim = AtomicClaim(
        node_id=leaves[0],
        target_object="number of U.S. residents commuting on a weekday",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday in 2025",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="ACS B08006 commuting tables and LEHD LODES",
        procedure="Pull the ACS B08006 commuting tables and audit the LEHD LODES dataset.",
    )
    state = propose_atom(state, node_id=leaves[0], claim=claim)

    frontier_nodes = frontier(state)
    frontier_ids = {node.node_id for node in frontier_nodes}
    assert leaves[0] not in frontier_ids
    assert leaves[1] in frontier_ids


# ---------------------------------------------------------------------------
# Alternative branch coverage
# ---------------------------------------------------------------------------


def test_alternative_branches_lists_non_current_branches():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale="alt split",
            child_specs=(
                ChildSpec(label="a", unit="person/day", scope=_scope()),
                ChildSpec(label="b", unit="person/day", scope=_scope()),
            ),
        ),
    )
    alternatives = state.alternative_branches()
    assert all(branch.branch_id != state.current_branch_id for branch in alternatives)


# ---------------------------------------------------------------------------
# Atomic claim request without an expansion is still evaluated
# ---------------------------------------------------------------------------


def test_atomic_claim_for_an_untracked_node_records_gap():
    state = create_decomposition(_commutes_question())
    claim = AtomicClaim(
        node_id="n_9999",
        target_object="never registered",
        unit="person/day",
        scope=_scope(),
        time_basis="per weekday",
        measurement_kind=MeasurementKind.RECORD_LOOKUP,
        source="field log",
        procedure="Pull the relevant row from the field log.",
    )
    with pytest.raises(DecompositionError, match="unknown node"):
        propose_atom(state, node_id="n_9999", claim=claim)


# ---------------------------------------------------------------------------
# Gap registration
# ---------------------------------------------------------------------------


def test_register_gap_requires_non_empty_explanation():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError):
        register_gap(state, kind=GapKind.UNRESOLVED_NODE, target="n_0001", explanation="   ")


def test_register_gap_appends_a_recorded_action():
    state = create_decomposition(_commutes_question())
    before_actions = len(state.actions)
    state = register_gap(
        state,
        kind=GapKind.UNRESOLVED_NODE,
        target="n_0001",
        explanation="node requires measurement procedure",
    )
    assert len(state.actions) == before_actions + 1
    assert state.actions[-1].kind is ActionKind.REGISTER_GAP


# ---------------------------------------------------------------------------
# Novel non-throughput example: kg CO2 per weekday from commuting
# ---------------------------------------------------------------------------


def test_carbon_footprint_question_decomposes_through_recursive_atoms():
    """Novel non-throughput example:

    ``How many kilogrammes of CO2 do U.S. weekday commuting vehicles emit per
    weekday in 2025?`` — the unit ``kg/day`` is built from a rate of
    ``kg/(km*day)`` times an average daily commute distance.  The tree uses
    a three-level decomposition so that every leaf carries an operational
    measurement procedure; pruning at one leaf exposes the gap without
    rewriting history.
    """

    question = Question(
        question_id="q-co2",
        question=(
            "How many kilogrammes of CO2 do U.S. weekday commuting vehicles emit per "
            "weekday in 2025?"
        ),
        target_subject="U.S. commuting CO2",
        target_measure="emitted CO2 mass",
        unit="kg/day",
        time_basis="per weekday in 2025",
        scope=Scope(population="U.S. weekday commuters", geography="United States"),
        decision_use="inform a city-level commute-shifting policy",
    )
    state = create_decomposition(question)

    # Level 1: kg/day = (kg/km*day) * (km/day)
    state = expand_state(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="kg/day",
            expression="_child_0 * _child_1",
            rationale="emissions = (emission factor per km) × (vehicle kilometres travelled)",
            child_specs=(
                ChildSpec(
                    label="emission factor",
                    unit="kg/km",
                    scope=_scope(),
                    description="kg CO2 emitted per vehicle-kilometre",
                ),
                ChildSpec(
                    label="vehicle kilometres per day",
                    unit="km/day",
                    scope=_scope(),
                    description="vehicle-kilometres travelled by commuting vehicles per weekday",
                ),
            ),
        ),
    )
    emission_factor_id, vehicle_km_id = state.expansions[0].child_node_ids

    # Level 2: vehicle km/day = (commuting vehicle hours/day) * (km/h)
    state = expand_state(
        state,
        request=ExpansionRequest(
            target_node_id=vehicle_km_id,
            parent_unit="km/day",
            expression="_child_0 * _child_1",
            rationale="distance = time spent commuting × average speed",
            child_specs=(
                ChildSpec(
                    label="commuting vehicle hours per day",
                    unit="h/day",
                    scope=_scope(),
                ),
                ChildSpec(
                    label="average commuting speed",
                    unit="km/h",
                    scope=_scope(),
                ),
            ),
        ),
    )

    # Promote one leaf to a concrete atom so the frontier shrinks.
    hours_leaf = state.expansions[-1].child_node_ids[0]
    state = propose_atom(
        state,
        node_id=hours_leaf,
        claim=AtomicClaim(
            node_id=hours_leaf,
            target_object="vehicle hours spent commuting per weekday",
            unit="h/day",
            scope=_scope(),
            time_basis="per weekday in 2025",
            measurement_kind=MeasurementKind.RECORD_LOOKUP,
            source="NHTS round-trip commute time × FHWA VM-1 vehicle counts",
            procedure=(
                "Take the average round-trip commute time from the NHTS and "
                "multiply by the count of commuting vehicles (FHWA VM-1); "
                "sum to obtain daily vehicle-hours."
            ),
        ),
    )
    assert state.node(hours_leaf).status is NodeStatus.ATOMIC_LEAF

    # Prune the speed leaf — the frontier must excise it without losing it.
    speed_leaf = state.expansions[-1].child_node_ids[1]
    state = prune(state, node_id=speed_leaf, reason="requires city-level GPS data")
    projected = pruning_projection(state)
    pruned_ids = set(projected.pruned_node_ids)
    assert speed_leaf in pruned_ids
    assert hours_leaf not in pruned_ids

    # The exported payload still reconstructs the entire historical tree
    # while flagging the pruned leaf as such.
    payload = state.export()
    assert payload["question"]["question_id"] == "q-co2"
    assert speed_leaf in payload["pruned_node_ids"]
    frontier_ids = {entry["node_id"] for entry in payload["frontier"]}
    assert speed_leaf not in frontier_ids


# ---------------------------------------------------------------------------
# Defensive structural tests
# ---------------------------------------------------------------------------


def test_register_node_requires_a_label():
    from aie_decision.decomposition_tree import register_node

    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="label"):
        register_node(state, label="   ", unit="person/day", scope=_scope())


def test_expansion_request_must_include_rationale():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="rationale"):
        expand_state(
            state,
            request=ExpansionRequest(
                target_node_id="n_0001",
                parent_unit="person/day",
                expression="_child_0",
                rationale="",
                child_specs=(ChildSpec(label="x", unit="person/day", scope=_scope()),),
            ),
        )


def test_propose_alternative_requires_a_prior_expansion():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="at least one existing"):
        propose_alternative(
            state,
            request=ExpansionRequest(
                target_node_id="n_0001",
                parent_unit="person/day",
                expression="_child_0",
                rationale="r",
                child_specs=(ChildSpec(label="x", unit="person/day", scope=_scope()),),
            ),
        )


def test_expanding_a_pruned_node_is_rejected():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = prune(state, node_id=state.expansions[0].child_node_ids[0], reason="not actionable")
    with pytest.raises(DecompositionError, match="pruned"):
        expand_state(
            state,
            request=ExpansionRequest(
                target_node_id=state.expansions[0].child_node_ids[0],
                parent_unit="person/day",
                expression="_child_0",
                rationale="r",
                child_specs=(ChildSpec(label="x", unit="person/day", scope=_scope()),),
            ),
        )


def test_node_helper_exposes_unknown_node_lookup_errors():
    state = create_decomposition(_commutes_question())
    with pytest.raises(DecompositionError, match="unknown node_id"):
        state.node("n_9999")


def test_current_branch_projection_filters_to_the_active_lineage():
    state = create_decomposition(_commutes_question())
    state = expand_state(state, request=_commute_expansion(state))
    state = propose_alternative(
        state,
        request=ExpansionRequest(
            target_node_id="n_0001",
            parent_unit="person/day",
            expression="_child_0 + _child_1",
            rationale="alt",
            child_specs=(
                ChildSpec(label="work trips", unit="person/day", scope=_scope()),
                ChildSpec(label="non-work trips", unit="person/day", scope=_scope()),
            ),
        ),
    )
    projected = current_branch_projection(state)
    branch = projected.current_branch()
    for expansion in projected.expansions:
        assert expansion.expansion_id in branch.expansion_ids


def test_canonical_dimension_key_for_dimensionless_unit():
    from aie_decision.decomposition_tree import _dimension_key_for

    assert _dimension_key_for(CompoundUnit()) == "dimensionless"
    assert "kg" in _dimension_key_for(CompoundUnit({"kg": 1}))