import unittest
import hashlib
import json

import httpx

from medai_science.agentmesh_control import AgentMeshControlPlaneClient, build_capability_route_plan, load_gateway_upstream


class AgentMeshControlPlaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_runtime_snapshot_checks_service_and_treats_empty_routes_as_unapproved(self):
        requests = []

        async def handler(request):
            requests.append(request.url.path)
            if request.url.path == "/v1/status":
                return httpx.Response(200, json={"status": "operational", "api_version": "v1"})
            if request.url.path.endswith("/snapshot"):
                return httpx.Response(404, json={"error": "no active snapshot"})
            return httpx.Response(404)

        client = AgentMeshControlPlaneClient(
            "http://127.0.0.1:8081", "token", "tenant-a", "namespace-b",
            httpx.MockTransport(handler),
        )
        snapshot = await client.authenticated_runtime_snapshot()
        self.assertEqual(snapshot["resources"], [])
        self.assertTrue(snapshot["empty"])
        self.assertEqual(snapshot["tenant"], "tenant-a")
        self.assertEqual(snapshot["namespace"], "namespace-b")
        self.assertEqual(requests, ["/v1/status", "/v1/scopes/tenant-a/namespace-b/snapshot"])

    async def test_authenticated_runtime_snapshot_returns_digest_checked_routes(self):
        resource = {"kind": "route", "name": "medai-test", "state": "active", "spec": {"service": "medai-mcp"}}
        encoded = json.dumps([resource], separators=(",", ":"), ensure_ascii=False).encode()
        digest = "sha256:" + hashlib.sha256((3).to_bytes(8, "big") + encoded).hexdigest()

        async def handler(request):
            if request.url.path == "/v1/status":
                return httpx.Response(200, json={"status": "operational", "api_version": "v1"})
            return httpx.Response(200, json={"resources": [resource], "revision": 3, "digest": digest})

        client = AgentMeshControlPlaneClient("http://localhost:8081", "token", "medai", "bio", httpx.MockTransport(handler))
        snapshot = await client.authenticated_runtime_snapshot()
        self.assertEqual(snapshot["revision"], 3)
        self.assertEqual(snapshot["digest"], digest)
        self.assertEqual(snapshot["control_plane_status"], "operational")

    async def test_authenticated_runtime_snapshot_rejects_unsupported_control_plane_status(self):
        async def handler(_request):
            return httpx.Response(200, json={"status": "operational", "api_version": "v2"})

        client = AgentMeshControlPlaneClient(
            "http://127.0.0.1:8081", "token", "tenant-a", "namespace-b",
            httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(ValueError, "supported API v1"):
            await client.authenticated_runtime_snapshot()

    def test_capability_lifecycle_builds_stable_auditable_routes(self):
        packages = [
            {"capability": {"id": "biomedical.research", "version": "1.0.0", "status": "production"}},
            {"capability": {"id": "biomedical.experimental", "version": "0.1.0", "status": "experimental"}},
        ]
        plan = build_capability_route_plan(packages, "medai-mcp")
        self.assertEqual(plan["resource_count"], 2)
        self.assertEqual(plan["active_count"], 1)
        self.assertEqual(plan["disabled_count"], 1)
        self.assertEqual(plan["resources"][0]["name"], "medai-biomedical.experimental-0.1.0")
        self.assertEqual(plan["resources"][0]["state"], "disabled")
        self.assertEqual(plan["resources"][1]["state"], "active")
        self.assertEqual(len(plan["desired_sha256"]), 64)
        self.assertEqual(plan["desired_sha256"], build_capability_route_plan(packages, "medai-mcp")["desired_sha256"])

    def test_capability_routes_fail_closed_on_duplicate_names_and_unreviewed_service(self):
        duplicate = [
            {"capability": {"id": "bio.cap", "version": "1.0", "status": "production"}},
            {"capability": {"id": "bio.cap", "version": "1.0", "status": "production"}},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_capability_route_plan(duplicate, "medai")
        with self.assertRaisesRegex(ValueError, "safe"):
            build_capability_route_plan([], "../service")

    def test_loads_gateway_upstream_without_credentials(self):
        path = __import__("pathlib").Path(__file__).parents[2] / "integrations/agentmesh/config/agentmesh-medai.yaml"
        upstream = load_gateway_upstream(path)
        self.assertEqual(upstream["url"], "http://127.0.0.1:8091/mcp")

    async def test_authenticated_status_discovery_apply_and_reconcile(self):
        requests = []

        async def handler(request):
            requests.append(request)
            if request.url.path == "/v1/status":
                return httpx.Response(200, json={"status": "operational", "api_version": "v1"})
            if request.url.path.endswith("/snapshot"):
                digest = "sha256:" + hashlib.sha256((0).to_bytes(8, "big") + b"[]").hexdigest()
                return httpx.Response(200, json={"resources": [], "revision": 0, "digest": digest})
            if request.url.path.endswith("/resources") and request.method == "POST":
                return httpx.Response(200, json={"revision": 1, "state": "active"})
            if request.url.path.endswith("/reconcile"):
                return httpx.Response(200, json={"revision": 1, "applied": 1})
            return httpx.Response(404)

        client = AgentMeshControlPlaneClient(
            "http://127.0.0.1:8081", "unit-test-admin-token", "medai", "biomedical",
            httpx.MockTransport(handler),
        )
        self.assertEqual((await client.status())["status"], "operational")
        await client.snapshot()
        resource = {"kind": "route", "name": "medai-research", "state": "active", "spec": {"service": "medai-mcp"}}
        await client.apply(resource, expected_revision=0)
        await client.reconcile()

        self.assertEqual(len(requests), 4)
        self.assertTrue(all(request.headers["authorization"] == "Bearer unit-test-admin-token" for request in requests))
        self.assertEqual(requests[1].url.path, "/v1/scopes/medai/biomedical/snapshot")
        self.assertEqual(requests[2].method, "POST")
        self.assertEqual(__import__("json").loads(requests[2].content), {"resource": resource, "expected_revision": 0})
        self.assertEqual(requests[3].content, b"")

    async def test_snapshot_digest_is_recomputed_and_tampering_rejected(self):
        resource = {"kind": "route", "name": "main", "state": "active", "spec": {"service": "medai", "capability": "papers.search"}}
        normalized = {"kind": "route", "name": "main", "state": "active", "spec": {"capability": "papers.search", "service": "medai"}}
        encoded = json.dumps([normalized], separators=(",", ":"), ensure_ascii=False).encode()
        digest = "sha256:" + hashlib.sha256((1).to_bytes(8, "big") + encoded).hexdigest()

        async def handler(request):
            return httpx.Response(200, json={"resources": [resource], "revision": 1, "digest": digest})

        client = AgentMeshControlPlaneClient("http://localhost:8081", "token", "medai", "bio", httpx.MockTransport(handler))
        self.assertEqual((await client.snapshot())["digest"], digest)

        async def tampered_handler(request):
            return httpx.Response(200, json={"resources": [{**resource, "spec": {"service": "attacker"}}], "revision": 1, "digest": digest})

        tampered = AgentMeshControlPlaneClient("http://localhost:8081", "token", "medai", "bio", httpx.MockTransport(tampered_handler))
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            await tampered.snapshot()

    async def test_validates_agentmesh_desired_resource_contract(self):
        client = AgentMeshControlPlaneClient("http://localhost:8081", "token", "medai", "bio")
        for resource in (
            {"kind": "route", "name": "main", "state": "active", "spec": {}, "extra": True},
            {"kind": "route", "name": "main", "state": "enabled", "spec": {}},
            {"kind": "route", "name": "main", "state": "active", "spec": float("nan")},
        ):
            with self.subTest(resource=resource), self.assertRaises(ValueError):
                await client.apply(resource)

    def test_rejects_cleartext_remote_admin_urls_and_bad_scope(self):
        with self.assertRaisesRegex(ValueError, "requires HTTPS"):
            AgentMeshControlPlaneClient("http://example.org", "token", "medai", "biomedical")
        with self.assertRaisesRegex(ValueError, "safe identifier"):
            AgentMeshControlPlaneClient("http://localhost:8081", "token", "../other", "biomedical")

    async def test_redirects_are_not_followed_with_admin_credentials(self):
        async def handler(request):
            return httpx.Response(307, headers={"Location": "https://attacker.invalid/collect"})

        client = AgentMeshControlPlaneClient("http://localhost:8081", "token", "medai", "bio", httpx.MockTransport(handler))
        with self.assertRaises(httpx.HTTPStatusError):
            await client.status()


if __name__ == "__main__":
    unittest.main()
