from __future__ import annotations

import unittest
import hashlib

from medai_science.amcp_generation import build_generated_capability_amcp_package


class GeneratedAmcpPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal = {
            "manifest": {
                "id": "genomics.variant_summary",
                "version": "0.1.0",
                "description": "Summarize a bounded set of public variant annotations.",
            },
            "contract": {
                "input": {
                    "type": "object",
                    "required": ["variants"],
                    "additionalProperties": False,
                    "properties": {"variants": {"type": "array", "maxItems": 20, "items": {"type": "object"}}},
                },
                "output": {
                    "type": "object",
                    "required": ["count"],
                    "additionalProperties": False,
                    "properties": {"count": {"type": "integer", "minimum": 0}},
                },
            },
        }
        self.arguments = {
            "source_tenant_id": "lab-a",
            "source_session_id": "RES-2026-000128",
            "proposal_artifact_id": "artifact-001",
            "proposal_artifact_sha256": "a" * 64,
        }

    def test_builds_deterministic_pending_review_package_bound_to_proposal(self) -> None:
        package = build_generated_capability_amcp_package(self.proposal, **self.arguments)
        self.assertEqual(package, build_generated_capability_amcp_package(self.proposal, **self.arguments))
        self.assertEqual(package["schemaVersion"], "amcp/0.1")
        self.assertEqual(package["capability"]["id"], "biomedical.generated.genomics-variant-summary-" + hashlib.sha256(b"genomics.variant_summary").hexdigest()[:16])
        inputs = package["contract"]["inputs"]
        self.assertNotIn("const", inputs["properties"]["session_id"])
        self.assertEqual(inputs["properties"]["source_tenant_id"]["const"], self.arguments["source_tenant_id"])
        self.assertEqual(inputs["properties"]["source_session_id"]["const"], self.arguments["source_session_id"])
        self.assertEqual(inputs["properties"]["proposal_artifact_id"]["const"], self.arguments["proposal_artifact_id"])
        self.assertEqual(inputs["properties"]["proposal_artifact_sha256"]["const"], self.arguments["proposal_artifact_sha256"])
        self.assertEqual(inputs["properties"]["proposal_capability_id"]["const"], "genomics.variant_summary")
        self.assertEqual(inputs["properties"]["input_data"], self.proposal["contract"]["input"])
        self.assertEqual(package["contract"]["outputs"]["properties"]["result"], self.proposal["contract"]["output"])
        self.assertEqual(package["implementations"][0]["reference"], "mcp://medai/medai_capabilities_execute_generated")
        self.assertFalse(package["contract"]["outputs"]["properties"]["validation_performed"]["const"])
        self.assertEqual(package["contract"]["outputs"]["properties"]["status"]["enum"], ["completed", "failed"])
        self.assertFalse(package["contract"]["outputs"]["properties"]["promoted"]["const"])

    def test_rejects_invalid_proposal_identity_contract_and_source_digest(self) -> None:
        invalid = {**self.proposal, "manifest": {**self.proposal["manifest"], "id": "../escape"}}
        with self.assertRaisesRegex(ValueError, "invalid stable ID"):
            build_generated_capability_amcp_package(invalid, **self.arguments)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            build_generated_capability_amcp_package(self.proposal, **{**self.arguments, "proposal_artifact_sha256": "A" * 64})
        with self.assertRaisesRegex(ValueError, "schema"):
            build_generated_capability_amcp_package({**self.proposal, "contract": {"input": {}, "output": {}}}, **self.arguments)

    def test_r_proposal_package_binds_language_in_intent_and_implementation(self) -> None:
        package = build_generated_capability_amcp_package({**self.proposal, "language": "r"}, **self.arguments)
        self.assertIn("R capability", package["capability"]["intent"])
        self.assertIn("-r-", package["implementations"][0]["id"])


if __name__ == "__main__":
    unittest.main()
