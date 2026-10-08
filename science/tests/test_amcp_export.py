from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import httpx
import yaml

from medai_science.amcp_export import export_generated_amcp_artifact
from medai_science.amcp_generation import build_generated_capability_amcp_package
from medai_science.provenance_signing import sign_capability_proposal


class GeneratedAmcpExportTests(unittest.TestCase):
    def _fixtures(self, *, tamper_package_digest: bool = False, language: str = "python"):
        session_id = "RES-2026-000128"
        amcp_artifact_id = "generated-amcp-artifact"
        source_artifact_id = "signed-source-artifact"
        signing_key = b"test-amcp-export-signing-key-0123456789"
        source = "run <- function(input) list(count = length(input$variants))\n" if language == "r" else "def run(input):\n    return {'count': len(input['variants'])}\n"
        proposal = {
            "schema_version": "1",
            **({"language": language} if language == "r" else {}),
            "manifest": {"id": "genomics.variant_summary", "version": "0.1.0", "description": "Summarize public variant annotations."},
            "contract": {
                "input": {"type": "object", "required": ["variants"], "properties": {"variants": {"type": "array", "items": {"type": "object"}}}},
                "output": {"type": "object", "required": ["count"], "properties": {"count": {"type": "integer", "minimum": 0}}},
            },
            "source": source,
            "static_review": {"source_sha256": hashlib.sha256(source.encode()).hexdigest()},
            "execution": {"enabled": False},
        }
        proposal = sign_capability_proposal(proposal, signing_key)
        source_payload = json.dumps(proposal, sort_keys=True, separators=(",", ":"))
        source_digest = hashlib.sha256(source_payload.encode()).hexdigest()
        package = build_generated_capability_amcp_package(
            proposal,
            source_tenant_id="lab-a",
            source_session_id=session_id,
            proposal_artifact_id=source_artifact_id,
            proposal_artifact_sha256=source_digest,
        )
        record = {
            "schema_version": "1",
            **({"language": language} if language == "r" else {}),
            "package": package,
            "source_tenant_id": "lab-a",
            "source_session_id": session_id,
            "source_proposal_artifact_id": source_artifact_id,
            "source_proposal_artifact_sha256": source_digest,
            "source_sha256": proposal["static_review"]["source_sha256"],
            "execution": {"enabled": False, "lifecycle": "pending_review"},
            "registered": False,
            "promoted": False,
        }
        amcp_payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        amcp_digest = hashlib.sha256(amcp_payload.encode()).hexdigest()
        source_artifact = {"id": source_artifact_id, "session_id": session_id, "kind": "generated_r_capability_proposal" if language == "r" else "generated_capability_proposal", "payload": source_payload, "sha256": source_digest}
        amcp_artifact = {"id": amcp_artifact_id, "session_id": session_id, "kind": "generated_capability_amcp_proposal", "payload": amcp_payload, "sha256": "0" * 64 if tamper_package_digest else amcp_digest}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/capabilities":
                return httpx.Response(200, json={"items": []}, headers={"content-type": "application/json"})
            if request.url.path.endswith("/" + amcp_artifact_id):
                return httpx.Response(200, json=amcp_artifact, headers={"content-type": "application/json"})
            if request.url.path.endswith("/" + source_artifact_id):
                return httpx.Response(200, json=source_artifact, headers={"content-type": "application/json"})
            return httpx.Response(404)

        return session_id, amcp_artifact_id, signing_key, package, httpx.MockTransport(handler)

    def test_exports_only_the_verified_source_bound_review_package(self) -> None:
        session, artifact_id, signing_key, package, transport = self._fixtures()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated.yaml"
            receipt = export_generated_amcp_artifact(
                "http://127.0.0.1:8090", "platform-admin-token-value-0123456789", session, artifact_id, output,
                transport=transport, signing_key=signing_key,
            )
            self.assertEqual(yaml.safe_load(output.read_text()), package)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertTrue(receipt["review_required"])
            self.assertFalse(receipt["execution_enabled"])
            self.assertFalse(receipt["registered"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                export_generated_amcp_artifact(
                    "http://127.0.0.1:8090", "platform-admin-token-value-0123456789", session, artifact_id, output,
                    transport=transport, signing_key=signing_key,
                )

    def test_refuses_tampered_package_artifact_before_writing(self) -> None:
        session, artifact_id, signing_key, _, transport = self._fixtures(tamper_package_digest=True)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated.yaml"
            with self.assertRaisesRegex(ValueError, "payload hash"):
                export_generated_amcp_artifact(
                    "http://127.0.0.1:8090", "platform-admin-token-value-0123456789", session, artifact_id, output,
                    transport=transport, signing_key=signing_key,
                )
            self.assertFalse(output.exists())

    def test_exports_signed_r_amcp_for_review(self) -> None:
        session, artifact_id, signing_key, package, transport = self._fixtures(language="r")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated-r.yaml"
            receipt = export_generated_amcp_artifact(
                "http://127.0.0.1:8090", "platform-admin-token-value-0123456789", session, artifact_id, output,
                transport=transport, signing_key=signing_key,
            )
            self.assertEqual(yaml.safe_load(output.read_text()), package)
            self.assertEqual(receipt["capability_id"], package["capability"]["id"])
            self.assertFalse(receipt["execution_enabled"])


if __name__ == "__main__":
    unittest.main()
