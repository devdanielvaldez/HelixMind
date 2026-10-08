from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from medai_science.agentmesh_control import AgentMeshControlPlaneClient, build_capability_route_plan
from medai_science.agentmesh_sync import AgentMeshSyncError, inspect_capability_sync, reconcile_runtime_lifecycle, seal_route_plan_for_review, sync_capability_routes


SIGNING_KEY = b"test-only-agentmesh-sync-key-with-32-bytes"


class FakeControlPlane:
    def __init__(self):
        self.desired = {}
        self.resource_revisions = {}
        self.snapshot_revision = 0
        self.fail_resource_apply_number = None
        self.commit_then_fail_resource_apply_number = None
        self.fail_reconcile_count = 0
        self.apply_count = 0

    @staticmethod
    def _sort(value):
        if isinstance(value, dict):
            return {key: FakeControlPlane._sort(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [FakeControlPlane._sort(item) for item in value]
        return value

    def snapshot(self):
        resources = [row for row in self.desired.values() if row["state"] != "disabled"]
        resources = [
            {"kind": row["kind"], "name": row["name"], "state": row["state"], "spec": self._sort(row["spec"])}
            for row in sorted(resources, key=lambda value: (value["kind"], value["name"]))
        ]
        encoded = json.dumps(resources, ensure_ascii=False, separators=(",", ":")).encode()
        digest = "sha256:" + hashlib.sha256(self.snapshot_revision.to_bytes(8, "big") + encoded).hexdigest()
        return {"resources": resources, "revision": self.snapshot_revision, "digest": digest}

    async def handle(self, request):
        if request.url.path.endswith("/snapshot"):
            return httpx.Response(200, json=self.snapshot())
        if request.url.path.endswith("/resources") and request.method == "POST":
            self.apply_count += 1
            fail_before_write = self.fail_resource_apply_number == self.apply_count
            fail_after_write = self.commit_then_fail_resource_apply_number == self.apply_count
            if fail_before_write:
                return httpx.Response(503, json={"error": "injected apply failure"})
            body = json.loads(request.content)
            resource = body["resource"]
            key = (resource["kind"], resource["name"])
            current = self.resource_revisions.get(key)
            expected = body.get("expected_revision")
            if (current is None and expected is not None) or (current is not None and expected != current):
                return httpx.Response(409, json={"error": "revision conflict"})
            revision = (current or 0) + 1
            self.resource_revisions[key] = revision
            self.desired[key] = resource
            if fail_after_write:
                return httpx.Response(503, json={"error": "response lost after desired-state commit"})
            return httpx.Response(200, json={"revision": revision})
        if request.url.path.endswith("/reconcile") and request.method == "POST":
            if self.fail_reconcile_count:
                self.fail_reconcile_count -= 1
                return httpx.Response(503, json={"error": "injected reconcile failure"})
            self.snapshot_revision += 1
            return httpx.Response(200, json={"revision": self.snapshot_revision})
        return httpx.Response(404)

    def client(self):
        return AgentMeshControlPlaneClient(
            "http://127.0.0.1:8081", "unit-test-admin-token", "medai", "biomedical",
            httpx.MockTransport(self.handle),
        )


def route_plan(strategy="least_active", second=False):
    packages = [{"capability": {"id": "biomedical.research", "version": "1.0.0", "status": "production"}}]
    if second:
        packages.append({"capability": {"id": "biomedical.experimental", "version": "0.1.0", "status": "experimental"}})
    plan = build_capability_route_plan(packages, "medai-mcp", strategy)
    plan["gateway_upstream"] = {"url": "http://127.0.0.1:8091/mcp", "request_timeout_ms": 30_000}
    plan["gateway_config_sha256"] = hashlib.sha256(b"gateway config fixture").hexdigest()
    plan["lifecycle_overrides_sha256"] = None
    return seal_route_plan_for_review(plan)


def approval_receipt(plan):
    scope = plan["control_plane_scope"]
    return {
        "receipt_id": "approval-test-001",
        "actor": "release-operator",
        "medai_tenant_id": "platform",
        "agentmesh_tenant": scope["tenant"],
        "agentmesh_namespace": scope["namespace"],
        "plan_review_sha256": plan["plan_review_sha256"],
        "desired_sha256": plan["desired_sha256"],
        "issued_at": int(time.time()),
        "expires_at": int(time.time()) + 600,
    }


class AgentMeshSyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_medai_lifecycle_is_reconciled_against_active_agentmesh_routes(self):
        plan = route_plan()
        report = {
            "runtime_active_routes": plan["resources"],
            "managed_route_names": [plan["resources"][0]["name"]],
        }
        production = [{"id": "biomedical.research", "version": "1.0.0", "status": "production", "kind": "mcp"}]
        matched = reconcile_runtime_lifecycle(report, production, "d" * 64)
        self.assertEqual(matched["status"], "in_sync")
        self.assertFalse(matched["mutation_performed"])

        revoked = [{"id": "biomedical.research", "version": "1.0.0", "status": "deprecated", "kind": "mcp"}]
        drift = reconcile_runtime_lifecycle(report, revoked, "e" * 64)
        self.assertEqual(drift["status"], "drift")
        self.assertEqual(drift["stale_active_routes"][0]["current_status"], "deprecated")
        self.assertTrue(drift["registry_changed_since_rollout"] is False)

        missing = reconcile_runtime_lifecycle(
            {"runtime_active_routes": [], "managed_route_names": [plan["resources"][0]["name"]]},
            production, "d" * 64,
        )
        self.assertEqual(missing["missing_approved_routes"], [plan["resources"][0]["name"]])
        non_mcp = reconcile_runtime_lifecycle(
            {"runtime_active_routes": [], "managed_route_names": []},
            [{"id": "biomedical.report", "version": "1.0.0", "status": "production", "kind": "workflow"}],
            "d" * 64,
        )
        self.assertEqual(non_mcp["missing_approved_routes"], [])

        outside_plan = reconcile_runtime_lifecycle(
            {"runtime_active_routes": [], "managed_route_names": []}, production, "d" * 64,
        )
        self.assertEqual(outside_plan["approved_routes_not_in_plan"], [plan["resources"][0]["name"]])
        self.assertEqual(outside_plan["status"], "in_sync")

        orphan = reconcile_runtime_lifecycle(
            {"runtime_active_routes": [{"kind": "route", "name": "medai-orphan-1", "state": "active", "spec": {}}], "managed_route_names": []},
            [], "d" * 64,
        )
        self.assertEqual(orphan["unregistered_active_routes"], ["medai-orphan-1"])

    async def test_applies_reviewed_routes_reconciles_verifies_and_is_idempotent(self):
        control = FakeControlPlane()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            plan = route_plan(second=True)
            approval = approval_receipt(plan)
            first = await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval, signing_key=SIGNING_KEY)
            self.assertEqual(first["changed_count"], 2)
            self.assertEqual(len(control.snapshot()["resources"]), 1)
            self.assertEqual(state.stat().st_mode & 0o777, 0o600)
            second = await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            self.assertEqual(second["changed_count"], 0)
            self.assertEqual(second["after_revision"], first["after_revision"])
            journal = json.loads(state.read_text())
            self.assertEqual(journal["approved_plan_sha256"], plan["plan_review_sha256"])
            self.assertEqual(journal["approval_receipt"]["actor"], "release-operator")
            self.assertEqual(first["approval_receipt_id"], "approval-test-001")
            self.assertEqual([event["outcome"] for event in journal["history"]], ["applied", "no_change"])

    async def test_sync_rejects_expired_or_mismatched_approval_receipts(self):
        control = FakeControlPlane()
        plan = route_plan()
        bad_receipt = approval_receipt(plan)
        bad_receipt["desired_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "approval receipt is expired or does not match"):
                await sync_capability_routes(
                    control.client(), plan, Path(directory) / "sync.json",
                    approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=bad_receipt,
                    signing_key=SIGNING_KEY,
                )
        self.assertEqual(control.apply_count, 0)

    async def test_sync_requires_authenticated_api_approval_receipt(self):
        control = FakeControlPlane()
        plan = route_plan()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "authenticated MedAI approval receipt is required"):
                await sync_capability_routes(
                    control.client(), plan, Path(directory) / "sync.json",
                    approved_plan_sha256=plan["plan_review_sha256"], signing_key=SIGNING_KEY,
                )
        self.assertEqual(control.apply_count, 0)

    async def test_reviewed_approval_cannot_be_replayed_in_another_tenant_or_namespace(self):
        control = FakeControlPlane()
        plan = route_plan()
        cross_scope = AgentMeshControlPlaneClient(
            "http://127.0.0.1:8081", "unit-test-admin-token", "tenant-b", "biomedical",
            httpx.MockTransport(control.handle),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "different AgentMesh tenant/namespace"):
                await sync_capability_routes(
                    cross_scope, plan, Path(directory) / "sync.json",
                    approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY,
                )
        self.assertEqual(control.apply_count, 0)

    async def test_sync_status_reports_in_sync_then_detects_external_route_drift_without_mutation(self):
        control = FakeControlPlane()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            plan = route_plan()
            await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            matched = await inspect_capability_sync(control.client(), state, signing_key=SIGNING_KEY)
            self.assertEqual(matched["status"], "in_sync")
            self.assertFalse(matched["mutation_performed"])

            key = ("route", plan["resources"][0]["name"])
            control.desired[key] = {**control.desired[key], "spec": {"service": "changed-outside-medai", "capability": "biomedical.research", "strategy": "least_active"}}
            control.snapshot_revision += 1
            drift = await inspect_capability_sync(control.client(), state, signing_key=SIGNING_KEY)
            self.assertEqual(drift["status"], "drift")
            self.assertEqual(drift["changed_runtime_routes"], ["route/medai-biomedical.research-1.0.0"])
            self.assertEqual(drift["snapshot_revision"], 2)
            self.assertFalse(drift["mutation_performed"])

    async def test_sync_status_can_inspect_but_does_not_clear_recovery_required_state(self):
        control = FakeControlPlane()
        control.commit_then_fail_resource_apply_number = 1
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            plan = route_plan()
            with self.assertRaises(AgentMeshSyncError):
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            report = await inspect_capability_sync(control.client(), state, signing_key=SIGNING_KEY)
            self.assertEqual(report["status"], "recovery_required")
            self.assertTrue(report["untracked_runtime_routes"])
            self.assertFalse(report["mutation_performed"])
            with self.assertRaisesRegex(ValueError, "recovery_required"):
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)

    async def test_sync_status_without_a_journal_does_not_provision_a_signing_key(self):
        control = FakeControlPlane()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            key = Path(directory) / "provenance.key"
            with patch.dict(os.environ, {"MEDAI_PROVENANCE_KEY_FILE": str(key)}, clear=False):
                with patch.dict(os.environ):
                    os.environ.pop("MEDAI_PROVENANCE_HMAC_KEY", None)
                    report = await inspect_capability_sync(control.client(), state)
            self.assertEqual(report["status"], "uninitialized")
            self.assertFalse(key.exists())

    async def test_failed_reconcile_restores_prior_snapshot_and_journals_new_revisions(self):
        control = FakeControlPlane()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            initial = route_plan()
            await sync_capability_routes(control.client(), initial, state, approved_plan_sha256=initial["plan_review_sha256"], approval_receipt=approval_receipt(initial), signing_key=SIGNING_KEY)
            before = control.snapshot()["resources"]
            control.fail_reconcile_count = 1
            with self.assertRaisesRegex(AgentMeshSyncError, "previous active snapshot restored"):
                changed_plan = route_plan("round_robin")
                await sync_capability_routes(control.client(), changed_plan, state, approved_plan_sha256=changed_plan["plan_review_sha256"], approval_receipt=approval_receipt(changed_plan), signing_key=SIGNING_KEY)
            self.assertEqual(control.snapshot()["resources"], before)
            journal = json.loads(state.read_text())
            self.assertEqual(journal["history"][-1]["outcome"], "rolled_back")
            retry = await sync_capability_routes(control.client(), initial, state, approved_plan_sha256=initial["plan_review_sha256"], approval_receipt=approval_receipt(initial), signing_key=SIGNING_KEY)
            self.assertEqual(retry["changed_count"], 0)

    async def test_uncertain_partial_apply_rolls_back_known_writes_and_blocks_unsafe_retry(self):
        control = FakeControlPlane()
        control.fail_resource_apply_number = 2
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            with self.assertRaisesRegex(AgentMeshSyncError, "recovery_required"):
                plan = route_plan(second=True)
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            self.assertEqual(control.snapshot()["resources"], [])
            journal = json.loads(state.read_text())
            self.assertTrue(journal["recovery_required"])
            with self.assertRaisesRegex(ValueError, "recovery_required"):
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)

    async def test_server_failure_after_commit_marks_remote_ambiguity_and_blocks_retry(self):
        control = FakeControlPlane()
        control.commit_then_fail_resource_apply_number = 1
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync.json"
            plan = route_plan()
            with self.assertRaisesRegex(AgentMeshSyncError, "outcome is uncertain"):
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            self.assertEqual(len(control.snapshot()["resources"]), 1)
            journal = json.loads(state.read_text())
            self.assertTrue(journal["recovery_required"])
            with self.assertRaisesRegex(ValueError, "recovery_required"):
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)

    async def test_refuses_unmanaged_medai_routes_without_a_signed_revision_journal(self):
        control = FakeControlPlane()
        resource = {"kind": "route", "name": "medai-existing", "state": "active", "spec": {"service": "other"}}
        control.desired[("route", "medai-existing")] = resource
        control.resource_revisions[("route", "medai-existing")] = 1
        control.snapshot_revision = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no signed sync journal"):
                plan = route_plan()
                await sync_capability_routes(control.client(), plan, Path(directory) / "sync.json", approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)

    async def test_unjournaled_disabled_desired_route_conflict_does_not_get_misreported_as_uncertain(self):
        control = FakeControlPlane()
        resource = route_plan()["resources"][0]
        resource = {**resource, "state": "disabled"}
        control.desired[(resource["kind"], resource["name"])] = resource
        control.resource_revisions[(resource["kind"], resource["name"])] = 1
        with tempfile.TemporaryDirectory() as directory:
            plan = route_plan()
            state = Path(directory) / "sync.json"
            with self.assertRaisesRegex(AgentMeshSyncError, "unmanaged AgentMesh desired resource"):
                await sync_capability_routes(control.client(), plan, state, approved_plan_sha256=plan["plan_review_sha256"], approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            journal = json.loads(state.read_text())
            self.assertFalse(journal["recovery_required"])
            self.assertIsNone(journal["pending_resource"])

    async def test_requires_operator_approval_of_the_exact_unmodified_plan(self):
        control = FakeControlPlane()
        plan = route_plan()
        plan["gateway_config_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "review digest"):
                await sync_capability_routes(control.client(), plan, Path(directory) / "sync.json", approved_plan_sha256="0" * 64, approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
            sealed = route_plan()
            with self.assertRaisesRegex(ValueError, "approved plan digest"):
                await sync_capability_routes(control.client(), sealed, Path(directory) / "sync.json", approved_plan_sha256="0" * 64, approval_receipt=approval_receipt(plan), signing_key=SIGNING_KEY)
        self.assertEqual(control.apply_count, 0)


if __name__ == "__main__":
    unittest.main()
