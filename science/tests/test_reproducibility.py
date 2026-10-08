from __future__ import annotations

import hashlib
import json
import unittest

from medai_science.reproducibility import build_reproducibility_bundle


SESSION = {"id": "RES-1", "objective": "Study APOE", "status": "running", "created_at": 1, "depth": "standard", "budget": 50, "credits_used": 8}
PLAN = {"session_id": "RES-1", "tasks": [{"id": "literature.discover", "status": "succeeded", "attempts": 1, "output": "{}"}]}
ARTIFACTS = [{"id": "ART-1", "session_id": "RES-1", "task_id": "literature.discover", "kind": "paper", "source": "PubMed", "source_id": "100", "sha256": "a" * 64, "recorded_at": 2, "payload": {"abstract": "must be omitted"}}]
GRAPH = {"session_id": "RES-1", "nodes": [{"session_id": "RES-1", "id": "NCBIGene:348", "kind": "Gene", "label": "APOE", "metadata": "{}"}], "edges": []}
EVIDENCE = [{"id": "E1", "session_id": "RES-1", "source": "PubMed", "source_id": "100", "polarity": "context", "quality": 0.5, "claim": "claim body is omitted"}]


class ReproducibilityBundleTests(unittest.TestCase):
    def test_builds_stable_hash_linked_text_minimized_bundle_and_report(self):
        first = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE)
        second = build_reproducibility_bundle(SESSION, PLAN, ARTIFACTS, GRAPH, EVIDENCE)
        self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
        self.assertEqual(first["artifacts"][0]["sha256"], "a" * 64)
        self.assertNotIn("must be omitted", str(first))
        self.assertNotIn("claim body is omitted", str(first))
        self.assertIn("not a clinical recommendation", first["report_markdown"])

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
            "truncated": False,
            "candidates": [{
                "id": "HYP-1",
                "statement": "APOE may be connected to TREM2 through microglial activation.",
                "evidence_ids": ["E1"],
                "dimensions": {"supporting_records": 1, "contradictory_records": 0, "context_records": 0, "novelty": None},
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
        proposal_payload = json.dumps({"candidates": [{"id": "HYP-1", "evidence_ids": ["E1"]}]}, separators=(",", ":"))
        proposal = {
            "id": "ART-HYP", "session_id": "RES-1", "task_id": "hypotheses.propose",
            "kind": "hypothesis_candidates", "sha256": "a" * 64, "payload": proposal_payload,
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
        report = build_reproducibility_bundle(SESSION, PLAN, [*base_artifacts, paper, tampered], GRAPH, EVIDENCE)["report_markdown"]
        self.assertIn("[omitted by privacy screen]", report)
        self.assertNotIn("my patient", report)
        self.assertIn("failed source identity or payload integrity checks", report)

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
