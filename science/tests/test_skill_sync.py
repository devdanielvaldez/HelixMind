from __future__ import annotations

import json
import os
import re
import unittest
from unittest.mock import patch

from medai_science.capabilities import canonical_amcp_registry_package
from medai_science.agentmesh_sync import _validate_route_plan
from medai_science.provenance_signing import sign_capability_proposal
from medai_science.skill_sync import build_signed_skill_route_plan, reconcile_skill_agentmesh_status


KEY = b"test-only-signing-key-with-at-least-32-bytes"


def signed_skill():
    payload = {
        "kind": "skill",
        "id": "biomedical.test_procedure",
        "version": "1.0.0+0123456789abcdef",
        "status": "pending_review",
        "review_required": True,
        "execution": {"enabled": False, "network": False},
        "procedure": [{"task_id": "literature.search", "description": "Search retained public literature records."}],
        "skill_markdown": "# Test procedure\n\nReview candidate sources before drawing conclusions.\n",
    }
    return sign_capability_proposal(payload, KEY)


class SkillAgentMeshReconciliationTests(unittest.TestCase):
    def fixture(self, status="verified", active=True, mutate_catalog=False):
        skill = signed_skill()
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            from medai_science.skill_export import build_agentmesh_skill_review_package
            package = build_agentmesh_skill_review_package(skill)
        package, package_digest = canonical_amcp_registry_package(package)
        local = json.loads(json.dumps(package))
        if mutate_catalog:
            local["capability"]["intent"] += " tampered"
        capability = package["capability"]
        key = f"{capability['id']}@{capability['version']}"
        registry = [{
            "id": capability["id"],
            "version": capability["version"],
            "status": status,
            "manifest": json.dumps({"source": "agentmesh_amcp", "package": package, "package_sha256": package_digest}),
        }]
        route_name = "medai-" + re.sub(r"[^A-Za-z0-9._-]", "-", key)
        snapshot = {
            "control_plane_status": "operational",
            "tenant": "medai",
            "namespace": "biomedical",
            "revision": 9,
            "digest": "sha256:" + "a" * 64,
            "empty": not active,
            "resources": [{"kind": "route", "name": route_name, "state": "active" if active else "disabled", "spec": {"service": "medai-mcp", "capability": capability["id"]}}],
        }
        return skill, local, registry, snapshot

    def test_signed_skill_route_plan_uses_exact_authenticated_package_lifecycle(self):
        skill, local, registry, _ = self.fixture(status="production", active=False)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            plan = build_signed_skill_route_plan(
                [skill], [local], registry, "a" * 64, "medai-mcp",
                {"url": "http://127.0.0.1:8091/mcp", "request_timeout_ms": 30_000}, "b" * 64,
            )
        self.assertEqual(plan["lifecycle_source"], "authenticated-medai-api-exact-package-match")
        self.assertEqual(plan["signed_skill_sources"][0]["lifecycle_status"], "production")
        self.assertEqual(plan["active_count"], 1)
        self.assertEqual(plan["resources"][0]["state"], "active")
        self.assertEqual(len(plan["plan_review_sha256"]), 64)
        self.assertEqual(_validate_route_plan(plan), plan["resources"])
        self.assertEqual(plan["execution_policy"], "routes expose review-only Skill retrieval; Skill procedures remain non-executable")
        self.assertEqual(set(plan["resources"][0]["spec"]), {"service", "capability", "strategy"})

    def test_signed_skill_route_plan_fails_closed_on_registry_package_drift(self):
        skill, local, registry, _ = self.fixture(status="production", active=False)
        manifest = json.loads(registry[0]["manifest"])
        manifest["package"]["capability"]["intent"] += " unreviewed change"
        registry[0]["manifest"] = json.dumps(manifest)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            plan = build_signed_skill_route_plan(
                [skill], [local], registry, "c" * 64, "medai-mcp",
                {"url": "http://127.0.0.1:8091/mcp", "request_timeout_ms": 30_000}, "d" * 64,
            )
        self.assertEqual(plan["signed_skill_sources"][0]["lifecycle_status"], "experimental")
        self.assertEqual(plan["active_count"], 0)
        self.assertEqual(plan["resources"][0]["state"], "disabled")

    def test_pending_review_registry_row_plans_a_disabled_route(self):
        skill, local, registry, _ = self.fixture(status="pending_review", active=False)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            plan = build_signed_skill_route_plan(
                [skill], [local], registry, "e" * 64, "medai-mcp",
                {"url": "http://127.0.0.1:8091/mcp", "request_timeout_ms": 30_000}, "f" * 64,
            )
        self.assertEqual(plan["signed_skill_sources"][0]["lifecycle_status"], "experimental")
        self.assertEqual(plan["resources"][0]["state"], "disabled")

    def test_empty_signed_skill_set_can_plan_retirement_of_previously_journaled_routes(self):
        plan = build_signed_skill_route_plan(
            [], [], [], "1" * 64, "medai-mcp",
            {"url": "http://127.0.0.1:8091/mcp", "request_timeout_ms": 30_000}, "2" * 64,
        )
        self.assertEqual(plan["resource_count"], 0)
        self.assertEqual(_validate_route_plan(plan), [])

    def test_reports_verified_active_skill_route_as_review_only(self):
        skill, local, registry, snapshot = self.fixture()
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        row = report["skills"][0]
        self.assertEqual(row["reconciliation"], "approved_and_routed_review_only")
        self.assertTrue(row["registry_package_matches_signed_skill"])
        self.assertTrue(row["runtime_route"]["active"])
        self.assertFalse(row["execution_enabled"])
        self.assertEqual(report["runtime_snapshot"]["control_plane_status"], "operational")
        self.assertEqual(report["runtime_snapshot"]["digest"], snapshot["digest"])

    def test_pending_review_stays_unapproved_even_if_a_route_is_active(self):
        skill, local, registry, snapshot = self.fixture(status="pending_review", active=True)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        self.assertEqual(report["skills"][0]["reconciliation"], "route_enabled_without_approval")
        self.assertTrue(report["skills"][0]["runtime_route"]["active"])
        self.assertFalse(report["skills"][0]["execution_enabled"])

    def test_flags_active_route_when_runtime_target_does_not_match_reviewed_scope(self):
        skill, local, registry, snapshot = self.fixture(status="verified", active=True)
        snapshot["resources"][0]["spec"]["service"] = "different-service"
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        row = report["skills"][0]
        self.assertEqual(row["reconciliation"], "runtime_route_scope_mismatch")
        self.assertFalse(row["runtime_route"]["active"])
        self.assertTrue(row["runtime_route"]["route_present"])
        self.assertFalse(row["runtime_route"]["scope_matches"])

    def test_detects_local_package_drift_and_route_left_active_after_retirement(self):
        skill, local, registry, snapshot = self.fixture(mutate_catalog=True)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        self.assertEqual(report["skills"][0]["reconciliation"], "local_package_mismatch")

        skill, local, registry, snapshot = self.fixture(status="deprecated", active=True)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        self.assertEqual(report["skills"][0]["reconciliation"], "route_enabled_after_retirement")

    def test_detects_registry_package_drift_and_approved_but_missing_route(self):
        skill, local, registry, snapshot = self.fixture()
        stored = json.loads(registry[0]["manifest"])
        stored["package"]["capability"]["intent"] += " changed"
        registry[0]["manifest"] = json.dumps(stored)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        self.assertEqual(report["skills"][0]["reconciliation"], "registry_package_mismatch")

        skill, local, registry, snapshot = self.fixture(status="verified", active=False)
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": KEY.decode()}, clear=True):
            report = reconcile_skill_agentmesh_status([skill], [local], registry, snapshot, "medai-mcp")
        self.assertEqual(report["skills"][0]["reconciliation"], "approved_route_not_active")


if __name__ == "__main__":
    unittest.main()
