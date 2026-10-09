from __future__ import annotations

import unittest
import hashlib
import json

from medai_science.reproducibility import build_reproducibility_bundle, compare_reproducibility_bundles, parse_research_bundle_artifact


SESSION = {"id": "RES-1", "objective": "Study APOE", "status": "running", "created_at": 1, "depth": "standard", "budget": 50, "credits_used": 8}
PLAN = {"session_id": "RES-1", "tasks": [{"id": "literature.discover", "status": "succeeded", "attempts": 1, "output": "one"}]}
ARTIFACT_PAYLOAD = json.dumps({"title": "APOE paper"}, separators=(",", ":"))
ARTIFACTS = [{"id": "ART-1", "session_id": "RES-1", "task_id": "literature.discover", "kind": "paper", "source": "PubMed", "source_id": "100", "sha256": hashlib.sha256(ARTIFACT_PAYLOAD.encode()).hexdigest(), "recorded_at": 2, "payload": ARTIFACT_PAYLOAD}]
GRAPH = {"session_id": "RES-1", "nodes": [{"session_id": "RES-1", "id": "APOE", "kind": "Gene", "label": "APOE"}], "edges": []}
EVIDENCE = [{"id": "E1", "session_id": "RES-1", "source": "PubMed", "source_id": "100", "polarity": "context", "quality": 0.5, "claim": "Context statement", "provenance": "{}"}]


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
        artifact_two_payload = json.dumps({"title": "TREM2 paper"}, separators=(",", ":"))
        next_artifacts = [*ARTIFACTS, {"id": "ART-2", "session_id": "RES-1", "task_id": "literature.discover", "kind": "paper", "source": "PubMed", "source_id": "200", "sha256": hashlib.sha256(artifact_two_payload.encode()).hexdigest(), "recorded_at": 3, "payload": artifact_two_payload}]
        next_graph = {**GRAPH, "nodes": [*GRAPH["nodes"], {"session_id": "RES-1", "id": "TREM2", "kind": "Gene", "label": "TREM2"}], "edges": [{"source": "APOE", "relation": "ASSOCIATED_WITH", "target": "TREM2", "evidence_ids": ["E2"], "artifact_ids": ["ART-2"]}]}
        next_evidence = [*EVIDENCE, {"id": "E2", "session_id": "RES-1", "source": "PubMed", "source_id": "200", "polarity": "supports", "quality": 0.7, "claim": "A second statement", "provenance": "{}"}]
        result = compare_reproducibility_bundles(snapshot(), snapshot(plan=next_plan, artifacts=next_artifacts, graph=next_graph, evidence=next_evidence))
        changes = result["changes"]
        self.assertTrue(result["review_required"])
        self.assertEqual(changes["tasks"]["modified"], ["literature.discover"])
        self.assertEqual(changes["artifacts"]["added"], ["ART-2"])
        self.assertEqual(changes["evidence_annotations"]["added"], ["E2"])
        self.assertEqual(changes["graph_nodes"]["added"], ["TREM2"])
        self.assertEqual(len(changes["graph_edges"]["added"]), 1)
        impact = result["change_impact"]
        self.assertEqual(impact["changed_artifact_ids"], ["ART-2"])
        self.assertEqual(impact["changed_evidence_annotation_ids"], ["E2"])
        self.assertEqual(impact["affected_graph_node_ids"], ["APOE", "TREM2"])
        self.assertEqual(impact["affected_graph_edges"][0]["changed_artifact_ids"], ["ART-2"])
        self.assertEqual(impact["affected_graph_edges"][0]["changed_evidence_ids"], ["E2"])
        self.assertTrue(impact["review_required"])

    def test_unchanged_bundle_has_no_traceability_impact(self):
        result = compare_reproducibility_bundles(snapshot(), snapshot())
        self.assertFalse(result["change_impact"]["review_required"])
        self.assertEqual(result["change_impact"]["affected_graph_edges"], [])

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

    def test_rejects_rehashed_bundle_with_dangling_graph_source_references(self):
        baseline = snapshot()
        malformed = snapshot()
        malformed["graph"] = {
            "nodes": [{"id": "APOE", "kind": "Gene", "label": "APOE"}],
            "edges": [{"source": "APOE", "relation": "RELATED_TO", "target": "APOE", "evidence_ids": ["MISSING"], "artifact_ids": []}],
        }
        digest_payload = {key: value for key, value in malformed.items() if key not in {"bundle_sha256", "report_markdown"}}
        malformed["bundle_sha256"] = hashlib.sha256(json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "dangling evidence or artifact reference"):
            compare_reproducibility_bundles(baseline, malformed)

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

    def test_bundle_parser_checks_internal_digest_even_when_api_payload_digest_is_recomputed(self):
        bundle = snapshot()
        tampered = {**bundle, "session": {**bundle["session"], "status": "complete"}}
        payload = json.dumps(tampered, ensure_ascii=False, separators=(",", ":"))
        artifact = {
            "kind": "research_reproducibility_bundle",
            "session_id": "RES-1",
            "payload": payload,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        with self.assertRaisesRegex(ValueError, "bundle hash verification failed"):
            parse_research_bundle_artifact(artifact, "RES-1")

    def test_bundle_parser_keeps_legacy_v1_snapshots_readable(self):
        bundle = snapshot()
        legacy = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
        legacy["schema_version"] = "1"
        legacy["evidence_annotations"] = [
            {key: value for key, value in row.items() if key not in {"claim_sha256", "provenance_sha256"}}
            for row in legacy["evidence_annotations"]
        ]
        legacy["bundle_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in legacy.items() if key != "report_markdown"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = json.dumps(legacy, ensure_ascii=False, separators=(",", ":"))
        artifact = {
            "kind": "research_reproducibility_bundle",
            "session_id": "RES-1",
            "payload": payload,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        self.assertEqual(parse_research_bundle_artifact(artifact, "RES-1"), legacy)


if __name__ == "__main__":
    unittest.main()
