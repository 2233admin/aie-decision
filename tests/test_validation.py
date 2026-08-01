import unittest

from aie_decision.models import (
    ConditionEdge, ConditionGraph, ConditionNode, ConditionStatus,
    Necessity, ProvenanceRef, Revision,
)
from aie_decision.validation import validate_document, validate_graph


def rev(name: str) -> Revision:
    return Revision(f"rev-{name}-1", 1, "2026-08-01T00:00:00Z")


def provenance() -> tuple[ProvenanceRef, ...]:
    return (ProvenanceRef("src-user", "prompt"),)


def graph(edges: tuple[ConditionEdge, ...]) -> ConditionGraph:
    nodes = (
        ConditionNode("c-supply", "Supply", "interval", Necessity.REQUIRED, ConditionStatus.MISSING, "constrains price", provenance=provenance(), revision=rev("c-supply")),
        ConditionNode("c-demand", "Demand", "interval", Necessity.REQUIRED, ConditionStatus.MISSING, "constrains price", provenance=provenance(), revision=rev("c-demand")),
    )
    return ConditionGraph("rcg-price", "q-price", "rev-q-price-1", nodes, edges, (("c-supply", "c-demand"),), rev("rcg-price"), provenance())


class ValidationTests(unittest.TestCase):
    def test_unknown_truth_confidence_is_explicit_and_valid(self):
        document = {
            "schema_version": "1.0.0", "evidence_atom_id": "ea-unknown", "source_id": "src-a",
            "source_locator": "p:1", "claim": "A statement was extracted", "epistemic_type": "primary_record",
            "independence_group": "origin-a", "target_relevance": ["condition:a"],
            "extraction_confidence": 0.95, "truth_confidence": None,
            "provenance": [{"source_id": "src-a", "locator": "p:1"}],
            "revision": {"revision_id": "rev-ea-unknown-1", "sequence": 1, "created_at": "2026-08-01T00:00:00Z"},
        }
        self.assertEqual(validate_document("evidence_atom", document), ())

    def test_valid_graph_passes_integrity_checks(self):
        edge = ConditionEdge("edge-supply-price", "c-supply", "target:market-price", "constrains", "hypothesis", "moves price", provenance=provenance(), revision=rev("edge-supply-price"))
        self.assertEqual(validate_graph(graph((edge,))), ())

    def test_graph_rejects_cycles_and_dangling_edges(self):
        edges = (
            ConditionEdge("edge-supply-demand", "c-supply", "c-demand", "depends", "hypothesis", "feeds demand", provenance=provenance(), revision=rev("edge-supply-demand")),
            ConditionEdge("edge-demand-supply", "c-demand", "c-supply", "depends", "hypothesis", "feeds supply", provenance=provenance(), revision=rev("edge-demand-supply")),
            ConditionEdge("edge-ghost-price", "c-ghost", "target:market-price", "constrains", "hypothesis", "moves price", provenance=provenance(), revision=rev("edge-ghost-price")),
        )
        codes = {issue.code for issue in validate_graph(graph(edges), raise_on_error=False)}
        self.assertTrue({"graph_cycle", "dangling_edge"} <= codes)

    def test_future_contract_requires_explicit_uncertainty_semantics(self):
        document = {
        "schema_version": "1.0.0", "question_id": "q-price", "question": "Tomorrow?",
        "answer_type": "future_prediction", "target": {"entity": "a", "measure": "price", "unit": "CNY"},
        "observation_cutoff": "2026-08-01T00:00:00Z", "prediction_horizon": "P1D",
        "revision": {"revision_id": "rev-q-price-1", "sequence": 1, "created_at": "2026-08-01T00:00:00Z"},
        }
        codes = {issue.code for issue in validate_document("answer_contract", document, raise_on_error=False)}
        self.assertIn("incomplete_answer_contract", codes)

    def test_package_requires_terminal_state_and_empty_section_reasons(self):
        document = {"schema_version": "1.0.0", "package_id": "pkg-x", "run_id": "run-x", "package_state": "partial", "answer_contract": {}, "condition_graph": {}, "sources": [], "evidence_propositions": [], "event_scene": None, "missing_conditions": [], "derived_factors": [], "calculations": [], "interval_audit": None, "conclusion": None, "answerability": {"status": "running", "reasons": []}, "empty_section_reasons": {}, "revision": {"revision_id": "rev-pkg-x-1", "sequence": 1, "created_at": "2026-08-01T00:00:00Z"}}
        codes = {issue.code for issue in validate_document("analysis_package", document, raise_on_error=False)}
        self.assertIn("terminal_answerability_missing", codes)
        self.assertIn("empty_section_reason_missing", codes)
        required_paths = {issue.path for issue in validate_document("analysis_package", document, raise_on_error=False) if issue.code == "required_field"}
        self.assertNotIn("$.event_scene", required_paths)
        self.assertNotIn("$.interval_audit", required_paths)
