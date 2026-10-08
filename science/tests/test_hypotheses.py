from __future__ import annotations

import hashlib
import json
import unittest

from medai_science.hypotheses import (
    critique_candidate,
    link_contradiction_search_candidates,
    propose_candidates,
    validate_hypothesis_proposal_artifact,
)


class HypothesisCriticLinkTests(unittest.TestCase):
    def proposal_artifact(self, payload=None):
        encoded = json.dumps(payload or {"candidates": [{"id": "HYP-1"}]}, separators=(",", ":"))
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

    def test_proposal_artifact_rejects_duplicate_json_keys_and_candidate_ids(self):
        artifact = self.proposal_artifact()
        artifact["payload"] = '{"candidates":[],"candidates":[]}'
        artifact["sha256"] = hashlib.sha256(artifact["payload"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "duplicate JSON keys"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")
        artifact = self.proposal_artifact({"candidates": [{"id": "HYP-1"}, {"id": "HYP-1"}]})
        with self.assertRaisesRegex(ValueError, "duplicate candidate IDs"):
            validate_hypothesis_proposal_artifact(artifact, "RES-1")

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


if __name__ == "__main__":
    unittest.main()
