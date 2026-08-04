import copy
import json

import pytest

from aie_decision.search import search_fermi
from aie_decision.search_replay import (
    SearchReplayError,
    create_search_checkpoint,
    replay_search_ledger,
    verify_search_checkpoint,
)


def _payload(max_rounds=5):
    return {
        "run_id": "replay-golden",
        "question": "How many daily buyers?",
        "target": "daily_buyers",
        "unit": "buyers/day",
        "coverage": 0.9,
        "reference_value": 100,
        "acceptable_width": 40,
        "variables": [
            {"name": "rough_total", "lower": 0, "upper": 1000, "method": "assumed"},
            {"name": "population", "lower": 950, "upper": 1050, "method": "observed"},
            {
                "name": "participation_rate",
                "lower": 0.09,
                "upper": 0.11,
                "method": "observed",
            },
        ],
        "candidates": [
            {
                "candidate_id": "rough",
                "formula": "rough_total",
                "mutation_kind": "seed",
            },
            {
                "candidate_id": "expanded",
                "parent_candidate_id": "rough",
                "mutation_kind": "expand",
                "formula": "population * participation_rate",
            },
        ],
        "budget": {
            "max_candidates": 10,
            "max_rounds": max_rounds,
            "max_evaluations": 10,
            "max_seconds": 5,
        },
    }


def test_replay_rebuilds_terminal_and_evaluated_candidates():
    result = search_fermi(_payload())

    replay = replay_search_ledger(result["ledger"])

    assert replay["terminal"] == {
        "state": "RESULT",
        "candidate_id": "expanded",
        "round_index": 2,
        "reason": "result-found",
        "data": {},
    }
    assert [item["candidate_id"] for item in replay["evaluated_candidates"]] == [
        "expanded",
        "rough",
    ]
    assert replay["pending_candidates"] == []


def test_replay_preserves_pending_candidate_at_budget_stop():
    result = search_fermi(_payload(max_rounds=1))

    replay = replay_search_ledger(result["ledger"])

    assert replay["terminal"]["state"] == "STOP"
    assert replay["terminal"]["reason"] == "budget-exhausted"
    assert replay["pending_candidates"] == ["expanded"]


def test_checkpoint_is_json_serializable_and_verifiable():
    ledger = search_fermi(_payload())["ledger"]

    checkpoint = create_search_checkpoint(ledger)
    restored = verify_search_checkpoint(json.loads(json.dumps(checkpoint)))

    assert restored == replay_search_ledger(ledger)


@pytest.mark.parametrize("tamper", ["payload", "sequence", "checkpoint"])
def test_replay_and_checkpoint_detect_tampering(tamper):
    ledger = search_fermi(_payload())["ledger"]
    if tamper == "checkpoint":
        checkpoint = create_search_checkpoint(ledger)
        checkpoint["replay"]["pending_candidates"].append("invented")
        with pytest.raises(SearchReplayError, match="checkpoint hash mismatch"):
            verify_search_checkpoint(checkpoint)
        return

    changed = copy.deepcopy(ledger)
    if tamper == "payload":
        changed["entries"][1]["payload"]["reason"] = "rewritten"
        message = "payload hash mismatch"
    else:
        changed["entries"][1]["sequence"] = 99
        message = "sequence must be contiguous"
    with pytest.raises(SearchReplayError, match=message):
        replay_search_ledger(changed)


def test_replay_rejects_impossible_event_transition_even_when_rehashed():
    ledger = copy.deepcopy(search_fermi(_payload())["ledger"])
    entry = ledger["entries"][1]
    entry["payload"]["state"] = "RANK"
    canonical = json.dumps(
        entry["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    entry["payload_hash"] = __import__("hashlib").sha256(canonical.encode()).hexdigest()

    with pytest.raises(SearchReplayError, match="invalid RANK transition"):
        replay_search_ledger(ledger)
