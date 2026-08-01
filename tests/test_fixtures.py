import json
from pathlib import Path
import unittest

from aie_decision.validation import validate_document

ROOT = Path(__file__).parents[1]


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class FixtureTests(unittest.TestCase):
    def test_all_versioned_schemas_are_well_formed_and_identified(self):
        schemas = list((ROOT / "schemas").glob("*.schema.json"))
        expected = {
            "answer-contract.schema.json", "required-condition-graph.schema.json",
            "evidence-atom.schema.json", "reconstructed-scene.schema.json",
            "condition-estimate.schema.json", "bayesian-update-record.schema.json",
            "derived-factor-candidate.schema.json", "forecast-interval-evaluation.schema.json",
            "complete-analysis-package.schema.json", "partial-analysis-package.schema.json",
        }
        self.assertTrue(expected <= {path.name for path in schemas})
        for path in schemas:
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertIn("/v1/", schema["$id"])
            self.assertTrue(schema["title"].endswith("v1"))

    def test_golden_corpus_covers_required_source_and_question_classes(self):
        corpus = read("fixtures/golden/v1/corpus.json")
        self.assertEqual(corpus["fixture_version"], "1.0.0")
        categories = {case["category"] for case in corpus["cases"]}
        self.assertEqual(categories, {
        "policy_text", "event_report", "mixed_fact_opinion_article",
        "contradictory_sources", "future_price_question", "sparse_evidence",
        })
        self.assertTrue(all(case.get("expected") for case in corpus["cases"]))

    def test_each_adversarial_fixture_triggers_its_named_guard(self):
        corpus = read("fixtures/adversarial/v1/corpus.json")
        self.assertEqual(corpus["fixture_version"], "1.0.0")
        self.assertEqual(len(corpus["cases"]), 6)
        for case in corpus["cases"]:
            issues = validate_document(case["contract_kind"], case["document"], raise_on_error=False)
            self.assertIn(case["expected_issue"], {issue.code for issue in issues}, case["case_id"])
