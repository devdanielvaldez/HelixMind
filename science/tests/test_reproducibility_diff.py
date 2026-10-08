from __future__ import annotations

import unittest
import hashlib
import json

from medai_science.reproducibility import build_reproducibility_bundle, compare_reproducibility_bundles, parse_research_bundle_artifact


SESSION = {"id": "RES-1", "objective": "Study APOE", "status": "running", "created_at": 1, "depth": "standard", "budget": 50, "credits_used": 8}
PLAN = {"session_id": "RES-1", "tasks": [{"id": "literature.discover", "status": "succeeded", "attempts": 1, "output": "one"}]}
ARTIFACTS = [{"id": "ART-1", "session_id": "RES-1", "task_id": "literature.discover", "kind": "paper", "source": "PubMed", "source_id": "100", "sha256": "a" * 64, "recorded_at": 2}]
GRAPH = {"session_id": "RES-1", "nodes": [{"session_id": "RES-1", "id": "APOE", "kind": "Gene", "label": "APOE"}], "edges": []}
EVIDENCE = [{"id": "E1", "session_id": "RES-1", "source": "PubMed", "source_id": "100", "polarity": "context", "quality": 0.5}]


def snapshot(session=SESSION, plan=PLAN, artifacts=ARTIFACTS, graph=GRAPH, evidence=EVIDENCE):
    return build_reproducibility_bundle(session, plan, artifacts, graph, evidence)


class ReproducibilityDiffTests(unittest.TestCase):
    def test_identical_captures_have_no_review_delta(self):
        result = compare_reproducibility_bundles(snapshot(), snapshot())
        self.assertFalse(result["review_required"])
        self.assertEqual(result["changes"]["tasks"]["modified"], [])

    def test_reports_session_status_and_budget_metadata_changes(self):
        changed = snapshot(session={**SESSION, "status": "complete", "credits_used": 12})
        result = compare_reproducibility_bundles(snapshot(), changed)
        self.assertEqual(result["changes"]["session_metadata"]["modified"], ["credits_used", "status"])
        self.assertTrue(result["review_required"])

    def test_reports_changes_by_research_record_category(self):
        next_plan = {"session_id": "RES-1", "tasks": [{"id": "literature.discover", "status": "needs_review", "attempts": 2, "output": "two"}]}
        next_artifacts = [*ARTIFACTS, {"id": "ART-2", "session_id": "RES-1", "task_id": "literature.discover", "kind": "paper", "source": "PubMed", "source_id": "200", "sha256": "b" * 64, "recorded_at": 3}]
        next_graph = {**GRAPH, "nodes": [*GRAPH["nodes"], {"session_id": "RES-1", "id": "TREM2", "kind": "Gene", "label": "TREM2"}], "edges": [{"source": "APOE", "relation": "ASSOCIATED_WITH", "target": "TREM2", "evidence_ids": ["E1"], "artifact_ids": ["ART-2"]}]}
        next_evidence = [*EVIDENCE, {"id": "E2", "session_id": "RES-1", "source": "PubMed", "source_id": "200", "polarity": "supports", "quality": 0.7}]
        result = compare_reproducibility_bundles(snapshot(), snapshot(plan=next_plan, artifacts=next_artifacts, graph=next_graph, evidence=next_evidence))
        changes = result["changes"]
        self.assertTrue(result["review_required"])
        self.assertEqual(changes["tasks"]["modified"], ["literature.discover"])
        self.assertEqual(changes["artifacts"]["added"], ["ART-2"])
        self.assertEqual(changes["evidence_annotations"]["added"], ["E2"])
        self.assertEqual(changes["graph_nodes"]["added"], ["TREM2"])
        self.assertEqual(len(changes["graph_edges"]["added"]), 1)

    def test_rejects_tampering_cross_session_comparison_and_duplicate_ids(self):
        original = snapshot()
        tampered = {**original, "session": {**original["session"], "status": "complete"}}
        with self.assertRaisesRegex(ValueError, "hash verification"):
            compare_reproducibility_bundles(original, tampered)
        with self.assertRaisesRegex(ValueError, "same research session"):
            compare_reproducibility_bundles(original, snapshot(session={**SESSION, "id": "RES-2"}, plan={**PLAN, "session_id": "RES-2"}, artifacts=[{**ARTIFACTS[0], "session_id": "RES-2"}], graph={**GRAPH, "session_id": "RES-2"}, evidence=[{**EVIDENCE[0], "session_id": "RES-2"}]))
        duplicate = [*ARTIFACTS, dict(ARTIFACTS[0])]
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            compare_reproducibility_bundles(original, snapshot(artifacts=duplicate))

    def test_parses_only_hash_verified_tenant_scoped_bundle_artifacts(self):
        bundle = snapshot()
        payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        artifact = {
            "kind": "research_reproducibility_bundle",
            "session_id": "RES-1",
            "payload": payload,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        self.assertEqual(parse_research_bundle_artifact(artifact, "RES-1"), bundle)
        with self.assertRaisesRegex(ValueError, "payload hash"):
            parse_research_bundle_artifact({**artifact, "payload": payload + " "}, "RES-1")
        with self.assertRaisesRegex(ValueError, "different session"):
            parse_research_bundle_artifact(artifact, "RES-2")
        with self.assertRaisesRegex(ValueError, "not a research reproducibility bundle"):
            parse_research_bundle_artifact({**artifact, "kind": "paper"}, "RES-1")


if __name__ == "__main__":
    unittest.main()
