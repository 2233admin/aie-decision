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

def _scope(label: str = "U.S. workers") -> Scope:
    return Scope(population=label, geography="United States")

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
