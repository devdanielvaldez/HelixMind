from __future__ import annotations

import hashlib
import json
import unittest

from medai_science.amcp_generation import build_generated_capability_amcp_package
from medai_science.capabilities import canonical_amcp_registry_package
from medai_science.generated_execution import require_exact_active_generated_route, resolve_production_generated_package


class GeneratedExecutionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal = {"manifest": {"id": "genomics.variant_summary", "version": "0.1.0", "description": "Summarize bounded variant data."}, "contract": {
            "input": {"type": "object", "properties": {"variants": {"type": "array", "items": {"type": "object"}}}, "required": ["variants"], "additionalProperties": False},
            "output": {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"], "additionalProperties": False},
        }}
        self.binding = {"source_tenant_id": "lab-a", "source_session_id": "RES-2026-1", "proposal_artifact_id": "artifact-1", "proposal_artifact_sha256": "a" * 64}
        self.package = build_generated_capability_amcp_package(self.proposal, **self.binding)
        self.capability_id = self.package["capability"]["id"]
        self.version = self.package["capability"]["version"]
        package, package_digest = canonical_amcp_registry_package(self.package)
        lifecycle = {"source": "agentmesh_amcp", "package": package, "package_sha256": package_digest, "execution": {"enabled": False, "network": False}}
        raw_manifest = json.dumps(lifecycle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.row = {"id": self.capability_id, "version": self.version, "kind": "mcp", "status": "production", "manifest": raw_manifest, "manifest_sha256": hashlib.sha256(raw_manifest.encode()).hexdigest()}

    def _resolve(self, rows=None, **overrides):
        return resolve_production_generated_package(rows or [self.row], capability_id=self.capability_id, version=self.version, **{**self.binding, **overrides}, proposal_capability_id="genomics.variant_summary")

    def test_resolves_exact_production_package_and_source_binding(self) -> None:
        package, receipt = self._resolve()
        self.assertEqual(package, self.package)
        self.assertEqual(receipt["status"], "production")

    def test_rejects_unapproved_tampered_or_cross_proposal_package(self) -> None:
        with self.assertRaisesRegex(ValueError, "production lifecycle"):
            self._resolve([{**self.row, "status": "verified"}])
        with self.assertRaisesRegex(ValueError, "digest"):
            self._resolve([{**self.row, "manifest_sha256": "0" * 64}])
        with self.assertRaisesRegex(ValueError, "does not bind"):
            self._resolve(proposal_artifact_id="artifact-other")

    def test_requires_exact_active_route_and_snapshot_provenance(self) -> None:
        route = "medai-" + self.capability_id + "-" + self.version
        snapshot = {"tenant": "medai", "namespace": "biomedical", "revision": 3, "digest": "b" * 64, "resources": [{"kind": "route", "name": route, "state": "active", "spec": {"service": "medai-mcp", "capability": self.capability_id}}]}
        result = require_exact_active_generated_route(self.package, snapshot, "medai-mcp")
        self.assertTrue(result["active"])
        with self.assertRaisesRegex(ValueError, "exact active AgentMesh route"):
            require_exact_active_generated_route(self.package, {**snapshot, "resources": []}, "medai-mcp")


if __name__ == "__main__":
    unittest.main()
