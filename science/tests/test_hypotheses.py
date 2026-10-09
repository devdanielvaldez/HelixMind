from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

from medai_science.hypotheses import (
    critique_candidate,
    link_contradiction_search_candidates,
    propose_candidates,
    validate_hypothesis_proposal_artifact,
)


class HypothesisCriticLinkTests(unittest.TestCase):
    def candidate(self):
        path = [
            {"source": "GENE:1", "relation": "ENCODES", "target": "PROTEIN:1"},
            {"source": "PROTEIN:1", "relation": "ASSOCIATED_WITH", "target": "DISEASE:1"},
        ]
        statement = "Gene A may be connected to Disease C through Protein B."
        return {
            "id": "HYP-1", "classification": "CANDIDATE RESEARCH HYPOTHESIS", "statement": statement,
            "entity_labels": ["Gene A", "Protein B", "Disease C"], "path": path,
            "path_evidence": [
                {**path[0], "evidence_ids": ["E1"], "polarity_counts": {"supports": 1, "contradicts": 0, "context": 0}, "interpretation": "Source-assigned annotations only."},
                {**path[1], "evidence_ids": ["E2"], "polarity_counts": {"supports": 1, "contradicts": 0, "context": 0}, "interpretation": "Source-assigned annotations only."},
            ],
            "dimensions": {
                "supporting_records": 2, "contradictory_records": 0, "context_records": 0,
                "mean_record_quality_annotation": 0.8, "quality_annotated_records": 2,
                "mean_edge_confidence_annotation": 0.7, "edge_confidence_annotated_count": 2,
                "contradictory_share_of_linked_records": 0.0, "novelty": None,
            },
            "evidence_ids": ["E1", "E2"], "limitations": ["Candidate only."],
        }

    def proposal_artifact(self, payload=None):
        payload = dict(payload or {"candidates": [self.candidate()]})
        payload.setdefault("candidate_count", len(payload.get("candidates", [])))
        payload.setdefault("returned_count", len(payload.get("candidates", [])))
        payload.setdefault("truncated", False)
        payload.setdefault("traversal_truncated", False)
        payload.setdefault("candidate_limit_omitted_count", 0)
        payload.setdefault("candidate_byte_omitted_count", 0)
        payload.setdefault("candidate_payload_bytes", sum(
            len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            for row in payload.get("candidates", [])
        ))
        encoded = json.dumps(payload, separators=(",", ":"))
        return {
            "id": "ART-PROPOSAL",
            "session_id": "RES-1",
            "kind": "hypothesis_candidates",
            "payload": encoded,
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }

    def test_proposal_artifact_requires_exact_session_kind_and_content_digest(self):
        artifact = self.proposal_artifact()
        self.assertEqual(validate_hypothesis_proposal_artifact(artifact, "RES-1")["candidates"][0]["id"], "HYP-1")
        with self.assertRaisesRegex(ValueError, "outside the requested"):
            validate_hypothesis_proposal_artifact(artifact, "RES-2")
        artifact["payload"] += " "
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")

    def test_proposal_artifact_reads_legacy_dimensions_without_annotation_denominators(self):
        candidate = self.candidate()
        candidate["dimensions"].pop("quality_annotated_records")
        candidate["dimensions"].pop("edge_confidence_annotated_count")
        artifact = self.proposal_artifact({"candidates": [candidate]})
        self.assertEqual(validate_hypothesis_proposal_artifact(artifact, "RES-1")["returned_count"], 1)

    def test_proposal_artifact_rejects_duplicate_json_keys_and_candidate_ids(self):
        artifact = self.proposal_artifact()
        artifact["payload"] = '{"candidates":[],"candidates":[]}'
        artifact["sha256"] = hashlib.sha256(artifact["payload"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "duplicate JSON keys"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        artifact = self.proposal_artifact({"candidates": [{"id": "HYP-1"}, {"id": "HYP-1"}]})
        with self.assertRaisesRegex(ValueError, "duplicate candidate IDs"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")

    def test_proposal_artifact_rejects_disconnected_or_unreceipted_path_edges(self):
        candidate = self.candidate()
        candidate["path"][1]["source"] = "DISEASE:other"
        artifact = self.proposal_artifact({"candidates": [candidate]})
        with self.assertRaisesRegex(ValueError, "not connected"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        candidate = self.candidate()
        candidate["evidence_ids"] = ["E1"]
        artifact = self.proposal_artifact({"candidates": [candidate]})
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        candidate = self.candidate()
        candidate["dimensions"]["novelty"] = 0.7
        artifact = self.proposal_artifact({"candidates": [candidate]})
        with self.assertRaisesRegex(ValueError, "unscored novelty"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        candidate = self.candidate()
        candidate["dimensions"]["contradictory_share_of_linked_records"] = 1.5
        artifact = self.proposal_artifact({"candidates": [candidate]})
        with self.assertRaisesRegex(ValueError, "finite annotations from zero to one"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        artifact = self.proposal_artifact({"candidates": [self.candidate()], "candidate_count": 3, "returned_count": 1})
        with self.assertRaisesRegex(ValueError, "counts and truncation flags are inconsistent"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")

    def test_proposal_artifact_checks_byte_receipts_and_truncation_accounting(self):
        payload = {
            "candidates": [self.candidate()], "candidate_count": 1, "returned_count": 1,
            "truncated": False, "traversal_truncated": False,
            "candidate_limit_omitted_count": 0, "candidate_byte_omitted_count": 0,
            "candidate_payload_bytes": 1,
        }
        artifact = self.proposal_artifact(payload)
        with self.assertRaisesRegex(ValueError, "byte-bound receipt"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        payload["candidate_payload_bytes"] = sum(
            len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            for row in payload["candidates"]
        )
        payload["truncated"] = True
        artifact = self.proposal_artifact(payload)
        with self.assertRaisesRegex(ValueError, "byte-bound receipt"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")

    def test_candidate_byte_limit_omits_candidate_with_explicit_accounting(self):
        nodes = [{"id": value, "label": value} for value in ("A", "B", "C")]
        edges = [
            {"source": "A", "target": "B", "relation": "R1", "evidence_ids": '["E1"]'},
            {"source": "B", "target": "C", "relation": "R2", "evidence_ids": '["E2"]'},
        ]
        evidence = [{"id": "E1", "polarity": "supports"}, {"id": "E2", "polarity": "supports"}]
        with patch("medai_science.hypotheses.MAX_PROPOSAL_CANDIDATE_BYTES", 1):
            result = propose_candidates(nodes, edges, evidence)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["returned_count"], 0)
        self.assertEqual(result["candidate_byte_omitted_count"], 1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["candidate_payload_bytes"], 0)
        artifact = self.proposal_artifact(result)
        self.assertEqual(validate_hypothesis_proposal_artifact(artifact, "RES-1")["candidate_byte_omitted_count"], 1)

    def make_artifact(self, *, target="E1", source_id="12345"):
        payload = {
            "target_evidence_id": target,
            "paper": {
                "title": "A retained literature record",
                "publication_date": "2025",
                "journal": "Example Journal",
                "doi": "10.1234/example",
                "provenance": {"source": "PubMed", "source_id": source_id},
            },
            "interpretation": "Retrieved as a contradiction-search candidate only; no contradictory relation or evidence polarity was inferred.",
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {"id": "ART-1", "kind": "contradiction_search_candidate", "source": "PubMed", "source_id": source_id, "payload": encoded, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}

    def test_links_search_candidates_only_to_the_exact_supporting_evidence(self):
        candidate = {"id": "HYP-1", "evidence_ids": ["E1"]}
        evidence = [{"id": "E1", "polarity": "supports"}, {"id": "E2", "polarity": "contradicts"}]
        result = link_contradiction_search_candidates(candidate, [self.make_artifact(target="E1"), self.make_artifact(target="E2")], evidence)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["target_evidence_id"], "E1")
        self.assertEqual(result[0]["source_id"], "12345")
        self.assertEqual(result[0]["relationship"], "retrieved_for_linked_supporting_claim_unassessed")
        self.assertEqual(result[0]["review_status"], "requires_human_scientific_review")

    def test_rejects_tampered_or_mislabeled_candidate_artifacts(self):
        artifact = self.make_artifact()
        artifact["payload"] += " "
        with self.assertRaisesRegex(ValueError, "hash"):
            link_contradiction_search_candidates({"evidence_ids": ["E1"]}, [artifact], [{"id": "E1", "polarity": "supports"}])
        artifact = self.make_artifact()
        artifact["source_id"] = "other"
        with self.assertRaisesRegex(ValueError, "identity"):
            link_contradiction_search_candidates({"evidence_ids": ["E1"]}, [artifact], [{"id": "E1", "polarity": "supports"}])

    def test_alternative_graph_routes_keep_source_polarity_receipts_and_omit_unlinked_routes(self):
        nodes = [{"id": value, "label": value} for value in ("A", "B", "C", "D")]
        candidate = {"id": "HYP-1", "evidence_ids": ["E1"], "path": [{"source": "A", "target": "B"}, {"source": "B", "target": "D"}]}
        edges = [
            {"source": "A", "target": "C", "relation": "ASSOCIATED", "evidence_ids": '["E2"]'},
            {"source": "C", "target": "D", "relation": "INVOLVED_IN", "evidence_ids": '["E3"]'},
            {"source": "A", "target": "C", "relation": "ASSOCIATED", "evidence_ids": '[]'},
        ]
        evidence = [{"id": "E1", "polarity": "supports"}, {"id": "E2", "polarity": "contradicts"}, {"id": "E3", "polarity": "context"}]
        result = critique_candidate(candidate, nodes, edges, evidence)
        self.assertEqual(len(result["alternative_graph_routes"]), 1)
        self.assertFalse(result["alternative_graph_routes_truncated"])
        route = result["alternative_graph_routes"][0]
        self.assertEqual(route["evidence_ids"], ["E2", "E3"])
        self.assertEqual(route["evidence_polarity_counts"], {"supports": 0, "contradicts": 1, "context": 1})
        self.assertIn("do not independently refute", route["interpretation"])

    def test_critic_lead_limit_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "1..100"):
            critique_candidate({}, [], [], [], lead_limit=101)

    def test_candidate_preserves_support_and_contradiction_counts_per_path_edge(self):
        nodes = [{"id": value, "label": value} for value in ("GENE:A", "PROTEIN:B", "DISEASE:C")]
        edges = [
            {"source": "GENE:A", "relation": "ENCODES", "target": "PROTEIN:B", "evidence_ids": '["E1","E2"]', "confidence": 0.7},
            {"source": "PROTEIN:B", "relation": "ASSOCIATED_WITH", "target": "DISEASE:C", "evidence_ids": '["E3","E4"]', "confidence": 0.8},
        ]
        evidence = [
            {"id": "E1", "polarity": "supports", "quality": 0.8},
            {"id": "E2", "polarity": "contradicts", "quality": 0.5},
            {"id": "E3", "polarity": "context", "quality": 0.4},
            {"id": "E4", "polarity": "supports", "quality": 0.6},
        ]
        result = propose_candidates(nodes, edges, evidence)
        self.assertEqual(result["candidate_count"], 1)
        path_evidence = result["candidates"][0]["path_evidence"]
        self.assertEqual(path_evidence[0]["evidence_ids"], ["E1", "E2"])
        self.assertEqual(path_evidence[0]["polarity_counts"], {"supports": 1, "contradicts": 1, "context": 0})
        self.assertEqual(path_evidence[1]["polarity_counts"], {"supports": 1, "contradicts": 0, "context": 1})

    def test_candidate_requires_supporting_evidence_for_each_path_edge(self):
        nodes = [{"id": value, "label": value} for value in ("GENE:A", "PROTEIN:B", "DISEASE:C")]
        edges = [
            {"source": "GENE:A", "relation": "ENCODES", "target": "PROTEIN:B", "evidence_ids": '["E1"]', "confidence": 0.7},
            {"source": "PROTEIN:B", "relation": "ASSOCIATED_WITH", "target": "DISEASE:C", "evidence_ids": '["E2"]', "confidence": 0.8},
        ]
        evidence = [{"id": "E1", "polarity": "supports"}, {"id": "E2", "polarity": "contradicts"}]
        self.assertEqual(propose_candidates(nodes, edges, evidence)["candidate_count"], 0)

    def test_candidate_keeps_evidence_with_missing_quality_annotations(self):
        nodes = [{"id": value, "label": value} for value in ("GENE:A", "PROTEIN:B", "DISEASE:C")]
        edges = [
            {"source": "GENE:A", "relation": "ENCODES", "target": "PROTEIN:B", "evidence_ids": '["E1"]', "confidence": 0.7},
            {"source": "PROTEIN:B", "relation": "ASSOCIATED_WITH", "target": "DISEASE:C", "evidence_ids": '["E2"]', "confidence": 0.8},
        ]
        evidence = [{"id": "E1", "polarity": "supports", "quality": None}, {"id": "E2", "polarity": "supports"}]
        candidate = propose_candidates(nodes, edges, evidence)["candidates"][0]
        self.assertEqual(candidate["evidence_ids"], ["E1", "E2"])
        self.assertIsNone(candidate["dimensions"]["mean_record_quality_annotation"])
        self.assertEqual(candidate["dimensions"]["quality_annotated_records"], 0)

    def test_candidate_rejects_invalid_quality_annotations(self):
        nodes = [{"id": value, "label": value} for value in ("GENE:A", "PROTEIN:B", "DISEASE:C")]
        edges = [
            {"source": "GENE:A", "relation": "ENCODES", "target": "PROTEIN:B", "evidence_ids": '["E1"]', "confidence": 0.7},
            {"source": "PROTEIN:B", "relation": "ASSOCIATED_WITH", "target": "DISEASE:C", "evidence_ids": '["E2"]', "confidence": 0.8},
        ]
        evidence = [{"id": "E1", "polarity": "supports", "quality": True}, {"id": "E2", "polarity": "supports"}]
        with self.assertRaisesRegex(ValueError, "quality annotations"):
            propose_candidates(nodes, edges, evidence)

    def test_candidate_keeps_path_when_edge_confidence_is_missing(self):
        nodes = [{"id": value, "label": value} for value in ("GENE:A", "PROTEIN:B", "DISEASE:C")]
        edges = [
            {"source": "GENE:A", "relation": "ENCODES", "target": "PROTEIN:B", "evidence_ids": '["E1"]'},
            {"source": "PROTEIN:B", "relation": "ASSOCIATED_WITH", "target": "DISEASE:C", "evidence_ids": '["E2"]', "confidence": None},
        ]
        evidence = [{"id": "E1", "polarity": "supports"}, {"id": "E2", "polarity": "supports"}]
        candidate = propose_candidates(nodes, edges, evidence)["candidates"][0]
        self.assertIsNone(candidate["dimensions"]["mean_edge_confidence_annotation"])
        self.assertEqual(candidate["dimensions"]["edge_confidence_annotated_count"], 0)

    def test_candidate_rejects_invalid_edge_confidence_annotations(self):
        nodes = [{"id": value, "label": value} for value in ("GENE:A", "PROTEIN:B", "DISEASE:C")]
        edges = [
            {"source": "GENE:A", "relation": "ENCODES", "target": "PROTEIN:B", "evidence_ids": '["E1"]', "confidence": True},
            {"source": "PROTEIN:B", "relation": "ASSOCIATED_WITH", "target": "DISEASE:C", "evidence_ids": '["E2"]', "confidence": 0.8},
        ]
        evidence = [{"id": "E1", "polarity": "supports"}, {"id": "E2", "polarity": "supports"}]
        with self.assertRaisesRegex(ValueError, "edge confidence annotations"):
            propose_candidates(nodes, edges, evidence)

    def test_candidate_enumerates_supported_three_edge_path_and_preserves_each_edge_receipt(self):
        nodes = [{"id": value, "label": label} for value, label in (
            ("GENE:APOE", "APOE"), ("PROTEIN:TREM2", "TREM2"),
            ("PATHWAY:LIPID", "lipid metabolism"), ("DISEASE:AD", "Alzheimer disease"),
        )]
        edges = [
            {"source": "GENE:APOE", "relation": "ENCODES", "target": "PROTEIN:TREM2", "evidence_ids": '["E1"]', "confidence": 0.7},
            {"source": "PROTEIN:TREM2", "relation": "INVOLVED_IN", "target": "PATHWAY:LIPID", "evidence_ids": '["E2"]', "confidence": 0.8},
            {"source": "PATHWAY:LIPID", "relation": "ASSOCIATED_WITH", "target": "DISEASE:AD", "evidence_ids": '["E3"]', "confidence": 0.9},
        ]
        evidence = [{"id": f"E{i}", "polarity": "supports", "quality": 0.5 + i / 10} for i in (1, 2, 3)]
        result = propose_candidates(nodes, edges, evidence)
        candidate = next(row for row in result["candidates"] if len(row["path"]) == 3)
        self.assertEqual(candidate["entity_labels"], ["APOE", "TREM2", "lipid metabolism", "Alzheimer disease"])
        self.assertEqual([row["evidence_ids"] for row in candidate["path_evidence"]], [["E1"], ["E2"], ["E3"]])
        self.assertIn("TREM2 → lipid metabolism", candidate["statement"])
        self.assertEqual(result["traversal_truncated"], False)

    def test_candidate_path_length_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "2..4 edges"):
            propose_candidates([], [], [], max_edges=5)

    def test_critic_finds_evidence_linked_alternative_route_for_multi_hop_candidate(self):
        nodes = [{"id": value, "label": value} for value in ("A", "B", "C", "D", "X", "Y")]
        candidate = {
            "id": "HYP-long",
            "evidence_ids": ["E1", "E2", "E3"],
            "path": [
                {"source": "A", "relation": "ASSOCIATED_WITH", "target": "B"},
                {"source": "B", "relation": "ASSOCIATED_WITH", "target": "C"},
                {"source": "C", "relation": "ASSOCIATED_WITH", "target": "D"},
            ],
        }
        edges = [
            {"source": "A", "relation": "ASSOCIATED_WITH", "target": "X", "evidence_ids": '["E4"]'},
            {"source": "X", "relation": "ASSOCIATED_WITH", "target": "Y", "evidence_ids": '["E5"]'},
            {"source": "Y", "relation": "ASSOCIATED_WITH", "target": "D", "evidence_ids": '["E6"]'},
        ]
        evidence = [{"id": f"E{i}", "polarity": polarity} for i, polarity in (
            (1, "supports"), (2, "supports"), (3, "supports"),
            (4, "context"), (5, "contradicts"), (6, "supports"),
        )]
        result = critique_candidate(candidate, nodes, edges, evidence)
        self.assertEqual(len(result["alternative_graph_routes"]), 1)
        self.assertFalse(result["alternative_graph_routes_truncated"])
        alternative = result["alternative_graph_routes"][0]
        self.assertEqual(len(alternative["path"]), 3)
        self.assertEqual(alternative["evidence_ids"], ["E4", "E5", "E6"])
        self.assertEqual(alternative["evidence_polarity_counts"], {"supports": 1, "contradicts": 1, "context": 1})


if __name__ == "__main__":
    unittest.main()
