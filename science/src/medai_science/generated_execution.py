"""Fail-closed checks for production execution of generated Python AMCP proposals."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from medai_science.capabilities import canonical_amcp_registry_package, runtime_capability_statuses


def resolve_production_generated_package(
    rows: list[dict[str, Any]],
    *,
    capability_id: str,
    version: str,
    source_tenant_id: str,
    source_session_id: str,
    proposal_artifact_id: str,
    proposal_artifact_sha256: str,
    proposal_capability_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one exact production AMCP package and verify its immutable source binding."""
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == capability_id and row.get("version") == version]
    if len(matches) != 1:
        raise ValueError("exact generated capability version is absent or duplicated in the authenticated lifecycle registry")
    row = matches[0]
    if row.get("status") not in {"production", "production_certified"} or row.get("kind") != "mcp":
        raise ValueError("generated capability requires exact production lifecycle approval")
    raw_manifest = row.get("manifest")
    if not isinstance(raw_manifest, str):
        raise ValueError("generated capability lifecycle manifest must be a serialized JSON object")
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise ValueError("generated capability lifecycle manifest is malformed") from error
    if not isinstance(manifest, dict) or manifest.get("source") != "agentmesh_amcp" or manifest.get("execution") != {"enabled": False, "network": False}:
        raise ValueError("registry entry is not a disabled AgentMesh AMCP package")
    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ValueError("registered AMCP package is missing")
    canonical, digest = canonical_amcp_registry_package(package)
    if manifest.get("package_sha256") != digest or row.get("manifest_sha256") != hashlib.sha256(raw_manifest.encode()).hexdigest():
        raise ValueError("registered package or lifecycle manifest digest failed verification")
    capability = canonical.get("capability", {})
    if capability.get("id") != capability_id or capability.get("version") != version:
        raise ValueError("registered AMCP identity differs from requested capability")
    authority_block = canonical.get("authority")
    authority = authority_block.get("permissions", []) if isinstance(authority_block, dict) else []
    if not isinstance(authority, list) or "biomedical.capabilities.execute" not in authority:
        raise ValueError("registered AMCP package does not declare generated-capability execution authority")
    implementations = canonical.get("implementations", [])
    if not isinstance(implementations, list) or len(implementations) != 1 or not isinstance(implementations[0], dict) or implementations[0].get("reference") != "mcp://medai/medai_capabilities_execute_generated":
        raise ValueError("registered AMCP package does not bind the supported generated execution handler")
    contract = canonical.get("contract")
    contract_inputs = contract.get("inputs") if isinstance(contract, dict) else None
    inputs = contract_inputs.get("properties", {}) if isinstance(contract_inputs, dict) else {}
    expected = {
        "source_tenant_id": source_tenant_id,
        "source_session_id": source_session_id,
        "proposal_artifact_id": proposal_artifact_id,
        "proposal_artifact_sha256": proposal_artifact_sha256,
        "proposal_capability_id": proposal_capability_id,
        "capability_id": capability_id,
        "version": version,
    }
    if any(not isinstance(inputs.get(key), dict) or inputs[key].get("const") != value for key, value in expected.items()):
        raise ValueError("registered AMCP contract does not bind the exact tenant, source proposal, and capability identity")
    return canonical, {"package_sha256": digest, "status": row["status"], "manifest_sha256": row["manifest_sha256"]}


def require_exact_active_generated_route(package: dict[str, Any], snapshot: dict[str, Any], service: str) -> dict[str, Any]:
    """Require an exact active AgentMesh route for the package and configured service."""
    route = runtime_capability_statuses([package], snapshot, service).get(
        f"{package['capability']['id']}@{package['capability']['version']}"
    )
    if not isinstance(route, dict) or route.get("active") is not True or route.get("scope_matches") is not True:
        raise ValueError("generated capability has no exact active AgentMesh route in the authenticated runtime snapshot")
    if not isinstance(route.get("snapshot_revision"), int) or not isinstance(route.get("snapshot_digest"), str):
        raise ValueError("AgentMesh runtime snapshot lacks revision or digest provenance")
    return route
