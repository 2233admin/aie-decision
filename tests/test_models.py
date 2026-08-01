from dataclasses import FrozenInstanceError
import unittest

from aie_decision.ledger import AnalysisLedger, LedgerError
from aie_decision.models import (
    AnswerContract, AnswerTarget, AnswerType, CoverageSemantics,
    ProvenanceRef, Revision, to_dict,
)


def contract(revision: Revision | None = None) -> AnswerContract:
    return AnswerContract(
        question_id="q-price",
        question="What range will the price occupy tomorrow?",
        answer_type=AnswerType.FUTURE_PREDICTION,
        target=AnswerTarget("commodity-a", "market-price", "CNY/unit"),
        observation_cutoff="2026-08-01T15:00:00+08:00",
        prediction_horizon="P1D",
        requested_coverage=0.9,
        uncertainty_semantics=CoverageSemantics.SUBJECTIVE_CREDIBLE_INTERVAL,
        revision=revision or Revision("rev-q-price-1", 1, "2026-08-01T00:00:00Z"),
        provenance=(ProvenanceRef("src-user", "prompt"),),
    )


class ModelTests(unittest.TestCase):
    def test_models_are_frozen_and_json_compatible(self):
        value = contract()
        with self.assertRaises(FrozenInstanceError):
            value.question = "changed"
        document = to_dict(value)
        self.assertEqual(document["schema_version"], "1.0.0")
        self.assertEqual(document["answer_type"], "future_prediction")
        self.assertEqual(document["uncertainty_semantics"], "subjective_credible_interval")

    def test_ledger_is_append_only_and_enforces_lineage(self):
        ledger = AnalysisLedger("run-price")
        first = ledger.append("answer_contract", contract())
        second_contract = contract(Revision("rev-q-price-2", 2, "2026-08-01T01:00:00Z", "rev-q-price-1"))
        second = ledger.append("answer_contract", second_contract)
        self.assertNotEqual(first.payload_hash, second.payload_hash)
        self.assertEqual(ledger.latest("answer_contract", "q-price"), second)
        self.assertEqual(len(ledger.records()), 2)
        with self.assertRaisesRegex(LedgerError, "revision already exists"):
            ledger.append("answer_contract", second_contract)
