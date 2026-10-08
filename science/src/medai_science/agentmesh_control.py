"""Authenticated client for the AgentMesh administrative control plane."""
from __future__ import annotations

import ipaddress
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from medai_science.sources.bounded_http import read_bounded_response

_SCOPE_PART = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_RESPONSE_BYTES = 4_000_000


def build_capability_route_plan(
    packages: list[dict[str, Any]], service: str, strategy: str = "least_active"
) -> dict[str, Any]:
    """Translate AMCP lifecycle metadata into deterministic, reviewable AgentMesh routes."""
    if not isinstance(service, str) or not _RESOURCE_NAME.fullmatch(service):
        raise ValueError("AgentMesh service must be a safe 1..128 character identifier")
    if strategy not in {"least_active", "round_robin", "weighted"}:
        raise ValueError("unsupported AgentMesh route strategy")
    resources = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("capability"), dict):
            raise ValueError("AMCP package is missing capability metadata")
        capability = package["capability"]
        capability_id, version = capability.get("id"), capability.get("version")
        lifecycle = capability.get("status", "experimental")
        if not isinstance(capability_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", capability_id):
            raise ValueError("AMCP capability has an invalid stable id")
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._+-]{1,64}", version):
            raise ValueError(f"AMCP capability {capability_id} has an invalid version")
        if lifecycle not in {"experimental", "testing", "validating", "verified", "production", "production_certified", "deprecated", "rolled_back"}:
            raise ValueError(f"AMCP capability {capability_id}@{version} has an invalid lifecycle state")
        state = "active" if lifecycle in {"verified", "production", "production_certified"} else "disabled"
        route_id = f"{capability_id}@{version}"
        route_name = re.sub(r"[^A-Za-z0-9._-]", "-", route_id)
        resources.append({
            "kind": "route",
            "name": f"medai-{route_name}",
            "state": state,
            "spec": {"service": service, "capability": capability_id, "strategy": strategy},
        })
    resources.sort(key=lambda item: item["name"])
    if len({item["name"] for item in resources}) != len(resources):
        raise ValueError("AMCP catalog contains duplicate capability versions that map to one route")
    canonical = json.dumps(resources, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": "1",
        "service": service,
        "strategy": strategy,
        "resources": resources,
        "resource_count": len(resources),
        "active_count": sum(item["state"] == "active" for item in resources),
        "disabled_count": sum(item["state"] == "disabled" for item in resources),
        "desired_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "policy": "only verified and production lifecycle capabilities are routed; lifecycle labels are administrative metadata, not proof of scientific validity",
    }


def load_gateway_upstream(config_path: str | Path) -> dict[str, Any]:
    """Read and validate the static MedAI gateway config used to target a route plan."""
    import yaml

    path = Path(config_path)
    if path.stat().st_size > 256_000:
        raise ValueError("AgentMesh gateway configuration exceeds 256 KiB")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    gateway = config.get("gateway") if isinstance(config, dict) else None
    upstream = gateway.get("upstream") if isinstance(gateway, dict) else None
    url = upstream.get("url") if isinstance(upstream, dict) else None
    parsed = urlsplit(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("gateway configuration must contain a credential-free HTTP(S) upstream URL")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if not loopback and upstream.get("allow_insecure_http") is not True:
            raise ValueError("cleartext remote MCP upstream is not permitted")
    return {"url": url, "request_timeout_ms": upstream.get("request_timeout_ms", 30000)}


class AgentMeshControlPlaneClient:
    """Call AgentMesh's versioned admin API without forwarding credentials on redirects."""

    def __init__(self, base_url: str, token: str, tenant: str, namespace: str, transport: httpx.AsyncBaseTransport | None = None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("AgentMesh control URL must be an absolute HTTP(S) URL without credentials, query, or fragment")
        if parsed.scheme == "http":
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = parsed.hostname.casefold() == "localhost"
            if not loopback:
                raise ValueError("AgentMesh administrative traffic requires HTTPS except for loopback control planes")
        if not token or len(token) > 4096:
            raise ValueError("AgentMesh admin token must contain 1..4096 characters")
        if not _SCOPE_PART.fullmatch(tenant) or not _SCOPE_PART.fullmatch(namespace):
            raise ValueError("tenant and namespace must each be 1..128 safe identifier characters")
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.namespace = namespace
        self._client_factory = lambda: httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=20,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    @property
    def scope_path(self) -> str:
        return f"/v1/scopes/{quote(self.tenant, safe='')}/{quote(self.namespace, safe='')}"

    async def _request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._client_factory() as client:
            async with client.stream(method, path, json=json_body) as response:
                body = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "AgentMesh control plane")
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("AgentMesh control plane returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("AgentMesh control-plane response must be a JSON object")
        return parsed

    async def status(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/status")

    async def snapshot(self) -> dict[str, Any]:
        snapshot = await self._request("GET", self.scope_path + "/snapshot")
        revision, resources, digest = snapshot.get("revision"), snapshot.get("resources"), snapshot.get("digest")
        if isinstance(revision, bool) or not isinstance(revision, int) or not 0 <= revision < 2**64:
            raise ValueError("AgentMesh snapshot revision is invalid")
        if not isinstance(resources, list) or len(resources) > 50_000:
            raise ValueError("AgentMesh snapshot resources are invalid or exceed 50000 rows")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("AgentMesh snapshot digest is malformed")
        normalized = []
        seen = set()
        for row in resources:
            if not isinstance(row, dict) or set(row) != {"kind", "name", "state", "spec"}:
                raise ValueError("AgentMesh snapshot contains a malformed desired resource")
            if not isinstance(row["kind"], str) or not _RESOURCE_NAME.fullmatch(row["kind"]) or not isinstance(row["name"], str) or not _RESOURCE_NAME.fullmatch(row["name"]):
                raise ValueError("AgentMesh snapshot contains an invalid resource identifier")
            if row["state"] not in {"active", "draining"}:
                raise ValueError("AgentMesh active snapshot contains an invalid resource state")
            key = (row["kind"], row["name"])
            if key in seen:
                raise ValueError("AgentMesh snapshot contains duplicate resource identifiers")
            seen.add(key)
            json.dumps(row["spec"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            normalized.append({"kind": row["kind"], "name": row["name"], "state": row["state"], "spec": self._sort_json(row["spec"])})
        normalized.sort(key=lambda row: (row["kind"], row["name"]))
        encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        calculated = "sha256:" + hashlib.sha256(revision.to_bytes(8, "big") + encoded).hexdigest()
        if calculated != digest:
            raise ValueError("AgentMesh snapshot digest does not match its resources")
        return snapshot

    async def authenticated_runtime_snapshot(self) -> dict[str, Any]:
        """Read a verified runtime snapshot; treat an empty active registry as no approved routes."""
        status = await self.status()
        if status.get("status") != "operational" or status.get("api_version") != "v1":
            raise ValueError("AgentMesh control plane is not operational on supported API v1")
        try:
            snapshot = await self.snapshot()
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            snapshot = {"resources": [], "revision": None, "digest": None, "empty": True}
        return {**snapshot, "control_plane_status": status["status"], "tenant": self.tenant, "namespace": self.namespace}

    async def apply(self, resource: dict[str, Any], expected_revision: int | None = None) -> dict[str, Any]:
        if not isinstance(resource, dict) or not resource:
            raise ValueError("desired resource must be a non-empty JSON object")
        if set(resource) != {"kind", "name", "state", "spec"}:
            raise ValueError("desired resource must contain exactly kind, name, state, and spec")
        if not isinstance(resource["kind"], str) or not _RESOURCE_NAME.fullmatch(resource["kind"]):
            raise ValueError("desired resource kind must be a 1..128 character identifier")
        if not isinstance(resource["name"], str) or not _RESOURCE_NAME.fullmatch(resource["name"]):
            raise ValueError("desired resource name must be a 1..128 character identifier")
        if not isinstance(resource["state"], str) or resource["state"] not in {"active", "draining", "disabled"}:
            raise ValueError("desired resource state must be active, draining, or disabled")
        try:
            spec_json = json.dumps(resource["spec"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("desired resource spec must be finite JSON data") from error
        if len(spec_json.encode("utf-8")) > 1_048_576 or self._json_depth(resource["spec"]) > 32:
            raise ValueError("desired resource spec exceeds AgentMesh size or nesting limits")
        if expected_revision is not None and (isinstance(expected_revision, bool) or expected_revision < 0):
            raise ValueError("expected_revision must be a non-negative integer")
        payload = {"resource": resource, "expected_revision": expected_revision}
        return await self._request("POST", self.scope_path + "/resources", payload)

    async def reconcile(self) -> dict[str, Any]:
        return await self._request("POST", self.scope_path + "/reconcile")

    @classmethod
    def _json_depth(cls, value: Any) -> int:
        if isinstance(value, dict):
            return 1 + max((cls._json_depth(item) for item in value.values()), default=0)
        if isinstance(value, list):
            return 1 + max((cls._json_depth(item) for item in value), default=0)
        return 1

    @classmethod
    def _sort_json(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._sort_json(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [cls._sort_json(item) for item in value]
        return value
