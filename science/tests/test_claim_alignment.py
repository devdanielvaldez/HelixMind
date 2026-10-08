from __future__ import annotations

import unittest
import hashlib
import json

from medai_science.claim_alignment import build_claim_alignment_candidates, decode_verified_annotation_artifact, validate_semantic_claim_pair


def annotation_artifact(document_id: str, candidate_id: str, direction: str, entities: list[str], *, artifact_id: str | None = None, study_system: str = "human") -> dict:
    payload = {
            "document_id": document_id,
            "source_sha256": "a" * 64,
            "candidate_text_hashes": {candidate_id: "b" * 64},
            "source_anchors": {candidate_id: {"passage_index": 3, "sentence_index": 0, "source_offset": 120, "source_local_offset": 0, "source_length": 50}},
            "annotations": [{
                "candidate_id": candidate_id,
                "result_kind": "association",
                "direction": direction,
                "study_system": study_system,
                "entity_ids": entities,
            }],
            "paper_context": {"study_design": "observational", "population": "patients", "method_family": "cohort_or_case_control"},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "id": artifact_id or "artifact-" + document_id,
        "session_id": "session-1",
        "kind": "paper_result_annotations",
        "source": "PMC BioC local-model result annotation",
        "source_id": document_id,
        "payload": encoded,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


class ClaimAlignmentTests(unittest.TestCase):
    def test_semantic_pair_requires_same_result_category_and_exact_shared_entity(self):
        first = {"result_kind": "association", "direction": "increase", "entity_ids": ["NCBI:1", "NCBI:2"]}
        second = {"result_kind": "association", "direction": "decrease", "entity_ids": ["NCBI:2"]}
        self.assertEqual(validate_semantic_claim_pair(first, second), ["NCBI:2"])
        with self.assertRaisesRegex(ValueError, "same source-annotated result category"):
            validate_semantic_claim_pair(first, {**second, "result_kind": "clinical"})
        with self.assertRaisesRegex(ValueError, "exact shared source entity"):
            validate_semantic_claim_pair(first, {**second, "entity_ids": ["NCBI:3"]})

    def test_direct_annotation_receipt_requires_session_kind_and_exact_payload_hash(self):
        payload = json.dumps({"document_id": "PMC1", "source_sha256": "a" * 64, "annotations": []}, separators=(",", ":"))
        artifact = {
            "id": "artifact-1", "session_id": "session-1", "kind": "paper_result_annotations",
            "source": "PMC BioC local-model result annotation", "payload": payload,
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
        decoded = decode_verified_annotation_artifact(artifact, "session-1")
        self.assertEqual(decoded["decoded_payload"]["document_id"], "PMC1")
        with self.assertRaisesRegex(ValueError, "requested research session"):
            decode_verified_annotation_artifact(artifact, "session-2")
        tampered = {**artifact, "payload": payload + " "}
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            decode_verified_annotation_artifact(tampered, "session-1")

    def test_direct_annotation_receipt_rejects_wrong_artifact_kind(self):
        payload = json.dumps({"document_id": "PMC1", "source_sha256": "a" * 64, "annotations": []})
        artifact = {
            "session_id": "session-1", "kind": "paper_extraction_metadata",
            "source": "PMC BioC local-model result annotation", "payload": payload,
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
        with self.assertRaisesRegex(ValueError, "not a retained PMC result-annotation"):
            decode_verified_annotation_artifact(artifact, "session-1")

    def test_direct_annotation_receipt_rejects_duplicate_keys_and_nonstandard_numbers(self):
        for payload in (
            '{"document_id":"PMC1","document_id":"PMC2","source_sha256":"' + "a" * 64 + '","annotations":[]}',
            '{"document_id":"PMC1","source_sha256":"' + "a" * 64 + '","annotations":[],"unexpected":NaN}',
        ):
            artifact = {
                "session_id": "session-1", "kind": "paper_result_annotations",
                "source": "PMC BioC local-model result annotation", "payload": payload,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "strict, unambiguous JSON"):
                decode_verified_annotation_artifact(artifact, "session-1")

    def test_links_opposite_directions_across_documents_with_provenance_only(self):
        result = build_claim_alignment_candidates([
            annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1", "NCBI:2"]),
            annotation_artifact("PMC2", "claim-2", "decrease", ["NCBI:1", "NCBI:2"]),
        ], session_id="session-1")
        self.assertEqual(len(result["links"]), 1)
        link = result["links"][0]
        self.assertEqual(link["relation"], "potential_directional_disagreement")
        self.assertEqual(link["shared_entity_ids"], ["NCBI:1", "NCBI:2"])
        self.assertEqual([item["document_id"] for item in link["sources"]], ["PMC1", "PMC2"])
        self.assertTrue(all("candidate_text_sha256" in item for item in link["sources"]))
        self.assertTrue(all(item["sentence_index"] == 0 and item["source_length"] == 50 for item in link["sources"]))
        self.assertNotIn("text", link)
        self.assertIn("not semantic", link["interpretation"])
        comparison = link["study_context_comparison"]
        self.assertEqual(comparison["study_system"]["relation"], "same_category_label")
        self.assertEqual(comparison["paper_context_dimensions"]["study_design"]["relation"], "same_category_label")
        self.assertEqual(comparison["paper_context_dimensions"]["population"]["relation"], "same_category_label")

    def test_study_and_paper_context_differences_are_review_context_not_contradiction(self):
        result = build_claim_alignment_candidates([
            annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"], study_system="human"),
            annotation_artifact("PMC2", "claim-2", "decrease", ["NCBI:1"], study_system="animal"),
        ], session_id="session-1")
        comparison = result["links"][0]["study_context_comparison"]
        self.assertEqual(comparison["study_system"]["relation"], "different_category_labels")
        self.assertEqual(comparison["study_system"]["source_labels"], ["human", "animal"])
        self.assertEqual(comparison["paper_context_dimensions"]["population"]["relation"], "same_category_label")
        self.assertTrue(comparison["review_required"])
        self.assertEqual(result["links"][0]["review_status"], "unreviewed_candidate")

    def test_invalid_model_context_labels_become_unassessed(self):
        result = build_claim_alignment_candidates([
            annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"], study_system="human"),
            annotation_artifact("PMC2", "claim-2", "decrease", ["NCBI:1"], study_system="nonsense"),
        ], session_id="session-1")
        comparison = result["links"][0]["study_context_comparison"]
        self.assertEqual(comparison["study_system"]["relation"], "unassessed")

    def test_population_difference_is_exposed_as_screening_label(self):
        first = annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"])
        second = annotation_artifact("PMC2", "claim-2", "decrease", ["NCBI:1"])
        second_payload = json.loads(second["payload"])
        second_payload["paper_context"]["population"] = "healthy_volunteers"
        second["payload"] = json.dumps(second_payload, sort_keys=True, separators=(",", ":"))
        second["sha256"] = hashlib.sha256(second["payload"].encode()).hexdigest()
        result = build_claim_alignment_candidates([first, second], session_id="session-1")
        self.assertEqual(result["links"][0]["study_context_comparison"]["paper_context_dimensions"]["population"]["relation"], "different_category_labels")

    def test_does_not_link_same_direction_or_same_document(self):
        result = build_claim_alignment_candidates([
            annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"], artifact_id="a"),
            annotation_artifact("PMC2", "claim-2", "increase", ["NCBI:1"], artifact_id="b"),
        ], session_id="session-1")
        self.assertEqual(result["links"], [])

        same_paper = build_claim_alignment_candidates([
            annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"], artifact_id="a"),
            annotation_artifact("PMC1", "claim-2", "decrease", ["NCBI:1"], artifact_id="b"),
        ], session_id="session-1")
        self.assertEqual(same_paper["links"], [])

    def test_ignores_invalid_or_unanchored_annotations(self):
        artifact = annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"])
        artifact_payload = json.loads(artifact["payload"])
        artifact_payload["source_anchors"] = {}
        artifact["payload"] = json.dumps(artifact_payload, sort_keys=True, separators=(",", ":"))
        artifact["sha256"] = hashlib.sha256(artifact["payload"].encode()).hexdigest()
        result = build_claim_alignment_candidates([artifact], session_id="session-1")
        self.assertEqual(result["links"], [])

    def test_candidate_builder_excludes_tampered_or_cross_session_annotation_artifacts(self):
        valid = annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"])
        other = annotation_artifact("PMC2", "claim-2", "decrease", ["NCBI:1"])
        tampered = {**valid, "sha256": "f" * 64}
        cross_session = {**other, "session_id": "session-2"}
        result = build_claim_alignment_candidates([tampered, cross_session], session_id="session-1")
        self.assertEqual(result["links"], [])
        self.assertEqual(result["invalid_annotation_artifact_count"], 2)
        self.assertTrue(result["needs_review"])

    def test_candidate_builder_excludes_ambiguous_json_even_with_matching_hash(self):
        payload = '{"document_id":"PMC1","document_id":"PMC2","source_sha256":"' + "a" * 64 + '","annotations":[]}'
        artifact = annotation_artifact("PMC1", "claim-1", "increase", ["NCBI:1"])
        artifact["payload"] = payload
        artifact["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        result = build_claim_alignment_candidates([artifact], session_id="session-1")
        self.assertEqual(result["invalid_annotation_artifact_count"], 1)
        self.assertTrue(result["needs_review"])

    def test_candidate_ids_reused_across_documents_do_not_collapse_distinct_pairs(self):
        result = build_claim_alignment_candidates([
            annotation_artifact("PMC1", "result_0", "increase", ["NCBI:1"], artifact_id="a"),
            annotation_artifact("PMC2", "result_0", "decrease", ["NCBI:1"], artifact_id="b"),
            annotation_artifact("PMC3", "result_0", "increase", ["NCBI:1"], artifact_id="c"),
            annotation_artifact("PMC4", "result_0", "decrease", ["NCBI:1"], artifact_id="d"),
        ], session_id="session-1")
        self.assertEqual(len(result["links"]), 4)
        source_pairs = {
            tuple(source["document_id"] for source in link["sources"])
            for link in result["links"]
        }
        self.assertEqual(
            source_pairs,
            {("PMC1", "PMC2"), ("PMC1", "PMC4"), ("PMC2", "PMC3"), ("PMC3", "PMC4")},
        )


if __name__ == "__main__":
    unittest.main()
