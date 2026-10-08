"""Read-only reconciliation of signed MedAI Skills with AgentMesh lifecycle state."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from medai_science.agentmesh_control import build_capability_route_plan
from medai_science.agentmesh_sync import seal_route_plan_for_review
from medai_science.capabilities import canonical_amcp_registry_package, runtime_capability_statuses
from medai_science.skill_export import build_agentmesh_skill_review_package


def build_signed_skill_route_plan(
    skills: list[dict[str, Any]],
    catalog_packages: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    registry_sha256: str,
    service: str,
    gateway_upstream: dict[str, Any],
    gateway_config_sha256: str,
    tenant: str = "medai",
    namespace: str = "biomedical",
) -> dict[str, Any]:
    """Plan routes only for signed Skills whose exact package is installed and reviewed."""
    if not isinstance(registry_sha256, str) or len(registry_sha256) != 64 or any(c not in "0123456789abcdef" for c in registry_sha256):
        raise ValueError("authenticated lifecycle registry digest is invalid")
    if not isinstance(gateway_config_sha256, str) or len(gateway_config_sha256) != 64 or any(c not in "0123456789abcdef" for c in gateway_config_sha256):
        raise ValueError("gateway configuration digest is invalid")

    catalog_by_key: dict[str, dict[str, Any]] = {}
    for package in catalog_packages:
        capability = package.get("capability") if isinstance(package, dict) else None
        if not isinstance(capability, dict):
            raise ValueError("AgentMesh catalog contains a malformed package")
        key = f"{capability.get('id')}@{capability.get('version')}"
        if key in catalog_by_key:
            raise ValueError(f"AgentMesh catalog contains duplicate package {key}")
        catalog_by_key[key] = package

    registry_by_key: dict[str, dict[str, Any]] = {}
    for row in registry_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("version"), str):
            raise ValueError("MedAI lifecycle registry contains a malformed identity")
        key = f"{row['id']}@{row['version']}"
        if key in registry_by_key:
            raise ValueError(f"MedAI lifecycle registry contains duplicate package {key}")
        registry_by_key[key] = row

    route_packages: list[dict[str, Any]] = []
    skill_sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for skill in skills:
        package = build_agentmesh_skill_review_package(skill)
        package, package_sha256 = canonical_amcp_registry_package(package)
        capability = package["capability"]
        key = f"{capability['id']}@{capability['version']}"
        if key in seen:
            raise ValueError(f"multiple signed Skills map to one AgentMesh route {key}")
        seen.add(key)
        catalog = catalog_by_key.get(key)
        if catalog is None:
            raise ValueError(f"signed Skill package {key} is not installed in the selected AgentMesh catalog")
        catalog_package, catalog_digest = canonical_amcp_registry_package(catalog)
        if catalog_digest != package_sha256 or catalog_package != package:
            raise ValueError(f"installed AgentMesh package {key} differs from its signed Skill")

        row = registry_by_key.get(key)
        lifecycle = "experimental"
        if row is not None:
            try:
                manifest = json.loads(row.get("manifest", "{}"))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"MedAI registry package {key} has invalid manifest JSON") from error
            if (
                isinstance(manifest, dict)
                and manifest.get("source") == "agentmesh_amcp"
                and manifest.get("package_sha256") == package_sha256
                and manifest.get("package") == package
            ):
                lifecycle = {
                    "pending_review": "experimental",
                    "experimental": "experimental",
                    "testing": "testing",
                    "validating": "validating",
                    "verified": "verified",
                    "production": "production",
                    "production_certified": "production_certified",
                    "deprecated": "deprecated",
                    "rolled_back": "rolled_back",
                }.get(row.get("status"), "experimental")
        routable = json.loads(json.dumps(package))
        routable["capability"]["status"] = lifecycle
        route_packages.append(routable)
        skill_sources.append({
            "skill_id": skill["id"],
            "skill_version": skill["version"],
            "skill_sha256": hashlib.sha256(json.dumps(skill, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "capability_id": capability["id"],
            "capability_version": capability["version"],
            "package_sha256": package_sha256,
            "lifecycle_status": lifecycle,
        })

    plan = build_capability_route_plan(route_packages, service)
    plan.update({
        "gateway_upstream": gateway_upstream,
        "gateway_config_sha256": gateway_config_sha256,
        "lifecycle_source": "authenticated-medai-api-exact-package-match",
        "lifecycle_registry_sha256": registry_sha256,
        "lifecycle_overrides_sha256": registry_sha256,
        "signed_skill_sources": skill_sources,
        "execution_policy": "routes expose review-only Skill retrieval; Skill procedures remain non-executable",
    })
    return seal_route_plan_for_review(plan, tenant, namespace)


def reconcile_skill_agentmesh_status(
    skills: list[dict[str, Any]],
    catalog_packages: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    runtime_snapshot: dict[str, Any],
    service: str,
) -> dict[str, Any]:
    """Compare local signed Skill artifacts to exact AMCP bytes, API lifecycle, and live route."""
    catalog_by_key: dict[str, dict[str, Any]] = {}
    for package in catalog_packages:
        capability = package.get("capability") if isinstance(package, dict) else None
        if not isinstance(capability, dict):
            raise ValueError("AgentMesh catalog contains a malformed package")
        key = f"{capability.get('id')}@{capability.get('version')}"
        if key in catalog_by_key:
            raise ValueError(f"AgentMesh catalog contains duplicate package {key}")
        catalog_by_key[key] = package

    registry_by_key: dict[str, dict[str, Any]] = {}
    for row in registry_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("version"), str):
            raise ValueError("MedAI capability registry contains a malformed identity")
        key = f"{row['id']}@{row['version']}"
        if key in registry_by_key:
            raise ValueError(f"MedAI capability registry contains duplicate package {key}")
        registry_by_key[key] = row

    statuses: dict[str, str] = {}
    expected_packages: dict[str, dict[str, Any]] = {}
    prepared = []
    for skill in skills:
        review_package = build_agentmesh_skill_review_package(skill)
        package, package_sha256 = canonical_amcp_registry_package(review_package)
        capability = package["capability"]
        key = f"{capability['id']}@{capability['version']}"
        if key in expected_packages:
            raise ValueError(f"multiple local Skills map to the same AgentMesh capability {key}")
        expected_packages[key] = package
        row = registry_by_key.get(key)
        if row is not None:
            statuses[key] = row["status"]
        amcp = catalog_by_key.get(key)
        local_matches = False
        if amcp is not None:
            local_package, local_digest = canonical_amcp_registry_package(amcp)
            local_matches = local_digest == package_sha256 and local_package == package

        registered_matches = False
        registry_manifest: dict[str, Any] | None = None
        if row is not None and isinstance(row.get("manifest"), str):
            try:
                registry_manifest = json.loads(row["manifest"])
            except json.JSONDecodeError as error:
                raise ValueError(f"registry package {key} contains invalid manifest JSON") from error
            registered_matches = (
                isinstance(registry_manifest, dict)
                and registry_manifest.get("source") == "agentmesh_amcp"
                and registry_manifest.get("package_sha256") == package_sha256
                and registry_manifest.get("package") == package
            )

        prepared.append({
            "skill_id": skill["id"],
            "skill_version": skill["version"],
            "skill_sha256": hashlib.sha256(json.dumps(skill, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "capability_id": capability["id"],
            "capability_version": capability["version"],
            "package_sha256": package_sha256,
            "local_package_installed": amcp is not None,
            "local_package_matches_signed_skill": local_matches,
            "registry_registered": row is not None,
            "registry_package_matches_signed_skill": registered_matches,
            "lifecycle_status": row.get("status", "not_registered") if row else "not_registered",
        })

    routed_packages = []
    for key, package in expected_packages.items():
        routed = json.loads(json.dumps(package))
        routed["capability"]["status"] = statuses.get(key, "experimental")
        routed_packages.append(routed)
    route_statuses = runtime_capability_statuses(routed_packages, runtime_snapshot, service)

    for item in prepared:
        key = f"{item['capability_id']}@{item['capability_version']}"
        route = route_statuses[key]
        item["runtime_route"] = route
        status = item["lifecycle_status"]
        if route["route_present"] and not route["scope_matches"]:
            item["reconciliation"] = "runtime_route_scope_mismatch"
        elif not item["local_package_installed"]:
            item["reconciliation"] = "local_package_missing"
        elif not item["local_package_matches_signed_skill"]:
            item["reconciliation"] = "local_package_mismatch"
        elif not item["registry_registered"]:
            item["reconciliation"] = "not_registered_in_medai"
        elif not item["registry_package_matches_signed_skill"]:
            item["reconciliation"] = "registry_package_mismatch"
        elif status in {"pending_review", "experimental", "testing", "validating"}:
            item["reconciliation"] = "route_enabled_without_approval" if route["state"] in {"active", "draining"} else "awaiting_administrative_approval"
        elif status in {"verified", "production", "production_certified"} and route["active"]:
            item["reconciliation"] = "approved_and_routed_review_only"
        elif status in {"deprecated", "rolled_back"} and route["state"] in {"active", "draining"}:
            item["reconciliation"] = "route_enabled_after_retirement"
        elif status in {"verified", "production", "production_certified"} and route["state"] == "draining":
            item["reconciliation"] = "approved_route_draining"
        elif status in {"verified", "production", "production_certified"}:
            item["reconciliation"] = "approved_route_not_active"
        else:
            item["reconciliation"] = "lifecycle_requires_review"
        item["execution_enabled"] = False

    return {
        "schema_version": "1",
        "service": service,
        "skills": prepared,
        "count": len(prepared),
        "runtime_snapshot": {key: runtime_snapshot.get(key) for key in ("control_plane_status", "tenant", "namespace", "revision", "digest", "empty")},
        "lifecycle_source": "authenticated MedAI registry",
        "interpretation_limit": "This report reconciles package identity, administrative lifecycle, and routing state only. AgentMesh routes expose a signed review-only Skill retrieval contract; they never authorize execution of the Skill procedure.",
    }
