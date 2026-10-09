from __future__ import annotations

import hashlib
import json
import unittest

from medai_science.reproducibility import build_reproducibility_bundle, compare_reproducibility_bundles


SESSION = {"id": "RES-1", "objective": "Study APOE", "status": "running", "created_at": 1, "depth": "standard", "budget": 50, "credits_used": 8}
PLAN = {"session_id": "RES-1", "tasks": [{"id": "literature.discover", "status": "succeeded", "attempts": 1, "output": "{}"}]}
ARTIFACT_PAYLOAD = json.dumps({"abstract": "must be omitted"}, separators=(",", ":"))
ARTIFACTS = [{"id": "ART-1", "session_id": "RES-1", "task_id": "literature.discover", "kind": "paper", "source": "PubMed", "source_id": "100", "sha256": hashlib.sha256(ARTIFACT_PAYLOAD.encode()).hexdigest(), "recorded_at": 2, "payload": ARTIFACT_PAYLOAD}]
GRAPH = {"session_id": "RES-1", "nodes": [{"session_id": "RES-1", "id": "NCBIGene:348", "kind": "Gene", "label": "APOE", "metadata": "{}"}], "edges": []}
EVIDENCE = [{"id": "E1", "session_id": "RES-1", "source": "PubMed", "source_id": "100", "polarity": "context", "quality": 0.5, "claim": "claim body is omitted", "provenance": "{}"}]


class ReproducibilityBundleTests(unittest.TestCase):
    def test_builds_stable_hash_linked_text_minimized_bundle_and_report(self):
        first = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE)
        second = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE)
        self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
        self.assertEqual(first["artifacts"][0]["sha256"], ARTIFACTS[0]["sha256"])
        self.assertNotIn("must be omitted", str(first))
        self.assertNotIn("claim body is omitted", str(first))
        self.assertEqual(first["evidence_annotations"][0]["claim_sha256"], hashlib.sha256(EVIDENCE[0]["claim"].encode()).hexdigest())
        self.assertEqual(first["evidence_annotations"][0]["provenance_sha256"], hashlib.sha256(EVIDENCE[0]["provenance"].encode()).hexdigest())
        self.assertIn("not a clinical recommendation", first["report_markdown"])

    def test_rejects_artifact_payload_tampering_before_snapshotting_its_digest(self):
        altered = [{**ARTIFACTS[0], "payload": ARTIFACT_PAYLOAD + " "}]
        with self.assertRaisesRegex(ValueError, "digest matching its exact retained payload bytes"):
            build_reproducibility_bundle(SESSION, PLAN, altered, GRAPH, EVIDENCE)

    def test_rejects_malformed_or_non_finite_evidence_instead_of_snapshotting_it(self):
        with self.assertRaisesRegex(ValueError, "provenance is not strict JSON"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, [{**EVIDENCE[0], "provenance": '{"x":1,"x":2}'}])
        with self.assertRaisesRegex(ValueError, "invalid claim, source, polarity, or quality"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, [{**EVIDENCE[0], "quality": float("nan")}])

    def test_rejects_cross_session_sources_and_dangling_graph_evidence(self):
        cross_session = [{**ARTIFACTS[0], "session_id": "RES-other"}]
        with self.assertRaisesRegex(ValueError, "cross-session"):
            build_reproducibility_bundle(SESSION, PLAN, cross_session, GRAPH, EVIDENCE)
        dangling = {**GRAPH, "edges": [{"session_id": "RES-1", "source": "NCBIGene:348", "target": "NCBIGene:348", "relation": "ASSOCIATED", "evidence_ids": '["unknown"]'}]}
        with self.assertRaisesRegex(ValueError, "dangling"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, dangling, EVIDENCE)
        with self.assertRaisesRegex(ValueError, "cross-session evidence"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, [{**EVIDENCE[0], "session_id": "RES-other"}])

    def test_includes_same_session_source_artifact_graph_provenance(self):
        graph = {**GRAPH, "edges": [{"session_id": "RES-1", "source": "NCBIGene:348", "target": "NCBIGene:348", "relation": "SUBMITTED_TO_REACTOME_ANALYSIS", "evidence_ids": "[]", "artifact_ids": '["ART-1"]'}]}
        bundle = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, graph, EVIDENCE)
        self.assertEqual(bundle["graph"]["edges"][0]["artifact_ids"], ["ART-1"])
        dangling = {**graph, "edges": [{**graph["edges"][0], "artifact_ids": '["ART-other"]'}]}
        with self.assertRaisesRegex(ValueError, "dangling"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, dangling, EVIDENCE)

    def test_bundle_captures_crossref_citation_edges_with_exact_artifact_provenance(self):
        links = [{
            "session_id": "RES-1", "citing_doi": "10.1234/citing",
            "cited_doi": "10.5678/cited", "source_artifact_id": "ART-1",
            "reference_index": 2,
        }]
        bundle = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE, links)
        self.assertEqual(bundle["citation_graph"]["edges"], [{
            "citing_doi": "10.1234/citing", "cited_doi": "10.5678/cited",
            "source_artifact_id": "ART-1", "reference_index": 2,
        }])
        self.assertIn("Captured Crossref citation links", bundle["report_markdown"])
        self.assertIn("do not establish scientific support", bundle["report_markdown"])
        with self.assertRaisesRegex(ValueError, "source artifact"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE, [{**links[0], "source_artifact_id": "ART-missing"}])

    def test_bundle_rejects_cross_session_and_duplicate_citation_edges(self):
        link = {"session_id": "RES-1", "citing_doi": "10.1234/citing", "cited_doi": "10.5678/cited", "source_artifact_id": "ART-1", "reference_index": 2}
        with self.assertRaisesRegex(ValueError, "cross-session"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE, [{**link, "session_id": "RES-2"}])
        with self.assertRaisesRegex(ValueError, "duplicate DOI edge"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE, [link, link])

    def test_bundle_diff_tracks_citation_edges_as_separate_bibliographic_changes(self):
        before = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE)
        link = {"session_id": "RES-1", "citing_doi": "10.1234/citing", "cited_doi": "10.5678/cited", "source_artifact_id": "ART-1", "reference_index": 2}
        after = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE, [link])
        diff = compare_reproducibility_bundles(before, after)
        self.assertEqual(diff["changes"]["citation_graph_edges"]["added"], ['["10.1234/citing","10.5678/cited"]'])
        self.assertTrue(diff["review_required"])

    def test_report_summarizes_only_evidence_linked_graph_relations_without_claim_text(self):
        graph = {
            **GRAPH,
            "nodes": [
                GRAPH["nodes"][0],
                {"session_id": "RES-1", "id": "TREM2", "kind": "Gene", "label": "TREM2"},
            ],
            "edges": [{
                "session_id": "RES-1", "source": "NCBIGene:348", "target": "TREM2",
                "relation": "ASSOCIATED_WITH", "evidence_ids": '["E1"]', "artifact_ids": '["ART-1"]',
            }],
        }
        bundle = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, graph, EVIDENCE)
        report = bundle["report_markdown"]
        self.assertIn("APOE (Gene) — `ASSOCIATED_WITH` → TREM2 (Gene)", report)
        self.assertIn("E1 (PubMed: 100; context)", report)
        self.assertIn("not claims of causality", report)
        self.assertNotIn("claim body is omitted", report)
        self.assertNotIn("must be omitted", report)

    def test_report_privacy_screens_graph_labels(self):
        graph = {
            **GRAPH,
            "nodes": [
                {"session_id": "RES-1", "id": "NCBIGene:348", "kind": "Gene", "label": "my patient"},
                {"session_id": "RES-1", "id": "TREM2", "kind": "Gene", "label": "TREM2"},
            ],
            "edges": [{
                "session_id": "RES-1", "source": "NCBIGene:348", "target": "TREM2",
                "relation": "ASSOCIATED_WITH", "evidence_ids": '["E1"]', "artifact_ids": '[]',
            }],
        }
        report = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, graph, EVIDENCE)["report_markdown"]
        self.assertIn("[omitted by privacy screen]", report)
        self.assertNotIn("my patient", report)

    def test_report_sanitizes_objective_markdown_and_newlines(self):
        session = {**SESSION, "objective": "Study APOE\n\n## Task outcomes\n- fabricated task: succeeded"}
        report = build_reproducibility_bundle(session, PLAN, ARTIFACTS, GRAPH, EVIDENCE)["report_markdown"]
        self.assertIn("Objective: Study APOE Task outcomes - fabricated task: succeeded", report)
        self.assertNotIn("\n\n## Task outcomes\n- fabricated task", report)

    def test_report_privacy_screens_patient_identifying_objectives(self):
        session = {**SESSION, "objective": "Investigate my patient APOE variant"}
        bundle = build_reproducibility_bundle(session, PLAN, ARTIFACTS, GRAPH, EVIDENCE)
        report = bundle["report_markdown"]
        self.assertIn("Objective: [omitted by privacy screen]", report)
        self.assertNotIn("my patient", report)
        self.assertNotIn("my patient", json.dumps(bundle))
        self.assertEqual(bundle["session"]["objective"], "[omitted by privacy screen]")

    def test_report_summarizes_integrity_verified_hypotheses_and_critic_status(self):
        proposal_payload = json.dumps({
            "candidate_count": 1,
            "returned_count": 1,
            "truncated": False,
            "traversal_truncated": False,
            "candidates": [{
                "id": "HYP-1",
                "classification": "CANDIDATE RESEARCH HYPOTHESIS",
                "statement": "APOE may be connected to Alzheimer disease through microglial activation.",
                "entity_labels": ["APOE", "microglial activation", "Alzheimer disease"],
                "path": [
                    {"source": "GENE:APOE", "relation": "ASSOCIATED_WITH", "target": "CELL:MICROGLIA"},
                    {"source": "CELL:MICROGLIA", "relation": "ASSOCIATED_WITH", "target": "DISEASE:ALZHEIMER"},
                ],
                "path_evidence": [
                    {"source": "GENE:APOE", "relation": "ASSOCIATED_WITH", "target": "CELL:MICROGLIA", "evidence_ids": ["E1"], "polarity_counts": {"supports": 1, "contradicts": 0, "context": 0}, "interpretation": "Source labels only."},
                    {"source": "CELL:MICROGLIA", "relation": "ASSOCIATED_WITH", "target": "DISEASE:ALZHEIMER", "evidence_ids": ["E1"], "polarity_counts": {"supports": 1, "contradicts": 0, "context": 0}, "interpretation": "Source labels only."},
                ],
                "evidence_ids": ["E1"],
                "dimensions": {"supporting_records": 1, "contradictory_records": 0, "context_records": 0, "mean_record_quality_annotation": 0.8, "quality_annotated_records": 1, "mean_edge_confidence_annotation": 0.7, "edge_confidence_annotated_count": 2, "contradictory_share_of_linked_records": 0.0, "novelty": None},
            }],
        }, separators=(",", ":"))
        proposal = {
            "id": "ART-HYP", "session_id": "RES-1", "task_id": "hypotheses.propose",
            "kind": "hypothesis_candidates", "sha256": hashlib.sha256(proposal_payload.encode()).hexdigest(),
            "payload": proposal_payload, "recorded_at": 3,
        }
        criticism_payload = json.dumps({
            "proposal_artifact_id": "ART-HYP", "proposal_artifact_sha256": proposal["sha256"],
            "reviews": [{
                "hypothesis_id": "HYP-1", "review_status": "requires_human_scientific_review",
                "linked_contradictory_evidence": [], "lexical_refutation_leads": [],
                "alternative_graph_routes": [], "retrieved_contradiction_candidates": [],
            }],
        }, separators=(",", ":"))
        criticism = {
            "id": "ART-CRIT", "session_id": "RES-1", "task_id": "hypotheses.criticize",
            "kind": "hypothesis_criticism", "sha256": hashlib.sha256(criticism_payload.encode()).hexdigest(),
            "payload": criticism_payload, "recorded_at": 4,
        }
        bundle = build_reproducibility_bundle(SESSION, PLAN, [*ARTIFACTS, proposal, criticism], GRAPH, EVIDENCE)
        report = bundle["report_markdown"]
        self.assertIn("## Candidate hypotheses and criticism", report)
        self.assertIn("**Candidate `HYP-1`:**", report)
        self.assertIn("Supporting records: 1; contradictory records: 0; context records: 0; novelty: unscored", report)
        self.assertIn("Criticism review `requires_human_scientific_review`", report)
        self.assertIn("not confirmed findings", report)

    def test_report_excludes_hypothesis_payloads_that_fail_integrity(self):
        proposal_payload = "not strict JSON"
        proposal = {
            "id": "ART-HYP", "session_id": "RES-1", "task_id": "hypotheses.propose",
            "kind": "hypothesis_candidates", "sha256": hashlib.sha256(proposal_payload.encode()).hexdigest(), "payload": proposal_payload,
        }
        report = build_reproducibility_bundle(SESSION, PLAN, [*ARTIFACTS, proposal], GRAPH, EVIDENCE)["report_markdown"]
        self.assertIn("proposal artifact(s) failed integrity or evidence-link checks", report)
        self.assertNotIn("HYP-1", report)

    def test_report_includes_only_hash_verified_pubmed_bibliographic_fields(self):
        paper_payload = json.dumps({
            "title": "APOE and microglia in a cohort study",
            "publication_date": "2025-03-02",
            "identifiers": {"doi": "10.1234/example.1"},
            "provenance": {"source": "PubMed", "source_id": "12345"},
            "abstract": "copyrighted abstract text must not be included",
        }, separators=(",", ":"))
        paper = {
            "id": "ART-PAPER", "session_id": "RES-1", "task_id": "literature.discover",
            "kind": "paper", "source": "PubMed", "source_id": "12345",
            "sha256": hashlib.sha256(paper_payload.encode()).hexdigest(),
            "payload": paper_payload, "recorded_at": 5,
        }
        base_artifacts = [row for row in ARTIFACTS if row.get("kind") != "paper"]
        report = build_reproducibility_bundle(SESSION, PLAN, [*base_artifacts, paper], GRAPH, EVIDENCE)["report_markdown"]
        self.assertIn("## Captured PubMed bibliography", report)
        self.assertIn("[PMID 12345](https://pubmed.ncbi.nlm.nih.gov/12345/)", report)
        self.assertIn("10.1234/example.1", report)
        self.assertIn("APOE and microglia in a cohort study", report)
        self.assertNotIn("copyrighted abstract text", report)

    def test_report_omits_privacy_flagged_paper_titles_and_unverified_payloads(self):
        paper_payload = json.dumps({
            "title": "Study of my patient and APOE",
            "provenance": {"source": "PubMed", "source_id": "12345"},
        }, separators=(",", ":"))
        paper = {
            "id": "ART-PAPER", "session_id": "RES-1", "task_id": "literature.discover",
            "kind": "paper", "source": "PubMed", "source_id": "12345",
            "sha256": hashlib.sha256(paper_payload.encode()).hexdigest(), "payload": paper_payload,
        }
        tampered = {**paper, "id": "ART-PAPER-2", "source_id": "99999", "sha256": "f" * 64}
        base_artifacts = [row for row in ARTIFACTS if row.get("kind") != "paper"]
        with self.assertRaisesRegex(ValueError, "digest matching its exact retained payload bytes"):
            build_reproducibility_bundle(SESSION, PLAN, [*base_artifacts, paper, tampered], GRAPH, EVIDENCE)
        report = build_reproducibility_bundle(SESSION, PLAN, [*base_artifacts, paper], GRAPH, EVIDENCE)["report_markdown"]
        self.assertIn("[omitted by privacy screen]", report)
        self.assertNotIn("my patient", report)

    def test_rejects_noncanonical_numbers_and_duplicate_research_identities(self):
        non_finite_output = {**PLAN, "tasks": [{**PLAN["tasks"][0], "output": float("nan")}]}
        with self.assertRaisesRegex(ValueError, "finite, canonical JSON"):
            build_reproducibility_bundle(SESSION, non_finite_output, ARTIFACTS, GRAPH, EVIDENCE)
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            build_reproducibility_bundle(SESSION, {**PLAN, "tasks": [*PLAN["tasks"], dict(PLAN["tasks"][0])]}, ARTIFACTS, GRAPH, EVIDENCE)
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, [*EVIDENCE, dict(EVIDENCE[0])])
        duplicate_edge = {**GRAPH, "edges": [
            {"source": "NCBIGene:348", "relation": "ASSOCIATED", "target": "NCBIGene:348", "evidence_ids": [], "artifact_ids": []},
            {"source": "NCBIGene:348", "relation": "ASSOCIATED", "target": "NCBIGene:348", "evidence_ids": [], "artifact_ids": []},
        ]}
        with self.assertRaisesRegex(ValueError, "duplicate edge"):
            build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, duplicate_edge, EVIDENCE)


if __name__ == "__main__":
    unittest.main()
