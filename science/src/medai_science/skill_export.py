"""Verified export of learned, disabled MedAI Skills as review-only AMCP packages."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from medai_science.provenance_signing import verify_capability_proposal_signature


_SKILL_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\+[a-f0-9]{16}$")
_MAX_SKILL_BYTES = 1_048_576


def read_verified_skill(registry_dir: str | Path, skill_id: str, version: str, signing_key: bytes | None = None) -> dict[str, Any]:
    """Load one immutable Skill by its validated identity and verify local provenance."""
    if not isinstance(skill_id, str) or not _SKILL_ID.fullmatch(skill_id):
        raise ValueError("Skill ID is not a valid namespaced identifier")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("Skill version is not a deterministic content version")
    root = Path(registry_dir).expanduser().resolve()
    package_path = root / skill_id / version / "skill.json"
    try:
        package_path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("Skill package path escapes the local registry") from error
    raw = package_path.read_bytes()
    if len(raw) > _MAX_SKILL_BYTES:
        raise ValueError("Skill package exceeds the 1 MiB export limit")
    package = json.loads(raw.decode("utf-8"))
    if not isinstance(package, dict) or package.get("id") != skill_id or package.get("version") != version:
        raise ValueError("Skill package identity does not match its registry path")
    if package.get("kind") != "skill":
        raise ValueError("registry entry is not a learned Skill")
    verify_capability_proposal_signature(package, signing_key)
    if package.get("status") != "pending_review" or package.get("review_required") is not True:
        raise ValueError("only pending-review Skills can be exported through the review package")
    execution = package.get("execution")
    if not isinstance(execution, dict) or execution.get("enabled") is not False or execution.get("network") is not False:
        raise ValueError("Skill execution and network access must remain disabled")
    procedure = package.get("procedure")
    if not isinstance(procedure, list) or not procedure or len(procedure) > 100 or any(not isinstance(step, dict) for step in procedure):
        raise ValueError("Skill procedure is missing, malformed, or exceeds 100 steps")
    markdown = package.get("skill_markdown")
    if not isinstance(markdown, str) or not markdown.strip() or len(markdown.encode("utf-8")) > 262_144:
        raise ValueError("Skill documentation is missing or exceeds the export limit")
    return package


def build_agentmesh_skill_review_package(skill: dict[str, Any], signing_key: bytes | None = None) -> dict[str, Any]:
    """Compile a signed Skill into an AMCP read-only retrieval contract, never an executor."""
    if not isinstance(skill, dict):
        raise ValueError("Skill package must be an object")
    verify_capability_proposal_signature(skill, signing_key)
    if skill.get("kind") != "skill":
        raise ValueError("registry entry is not a learned Skill")
    skill_id, version = skill.get("id"), skill.get("version")
    if not isinstance(skill_id, str) or not _SKILL_ID.fullmatch(skill_id) or not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("Skill identity or content version is invalid")
    if skill.get("status") != "pending_review" or skill.get("review_required") is not True:
        raise ValueError("only review-gated Skill candidates can be exported")
    execution = skill.get("execution")
    if not isinstance(execution, dict) or execution.get("enabled") is not False or execution.get("network") is not False:
        raise ValueError("Skill execution and network access must remain disabled")
    markdown = skill.get("skill_markdown")
    if not isinstance(markdown, str) or not markdown.strip() or len(markdown.encode("utf-8")) > 262_144:
        raise ValueError("Skill documentation is missing or exceeds the export limit")
    canonical = json.dumps(skill, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(canonical) > _MAX_SKILL_BYTES:
        raise ValueError("Skill package exceeds the 1 MiB export limit")
    skill_digest = hashlib.sha256(canonical).hexdigest()
    procedure = skill.get("procedure")
    if not isinstance(procedure, list) or not procedure or len(procedure) > 100 or any(not isinstance(step, dict) for step in procedure):
        raise ValueError("Skill procedure is missing, malformed, or exceeds 100 steps")
    package = {
        "schemaVersion": "amcp/0.1",
        "capability": {
            "id": "biomedical.skill." + skill_id.replace("_", "-"),
            "version": version,
            "intent": "Retrieve a signed, disabled MedAI biomedical Skill candidate for human review; this capability never executes its procedure.",
        },
        "contract": {
            "inputs": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "const": skill_id},
                    "version": {"type": "string", "const": version},
                },
                "required": ["skill_id", "version"],
                "additionalProperties": False,
            },
            "outputs": {
                "type": "object",
                "properties": {
                    "skill": {"type": "object"},
                    "skill_markdown": {"type": "string"},
                    "skill_sha256": {"type": "string", "const": skill_digest},
                    "signature_verified": {"type": "boolean", "const": True},
                    "execution_enabled": {"type": "boolean", "const": False},
                    "review_required": {"type": "boolean", "const": True},
                },
                "required": ["skill", "skill_markdown", "skill_sha256", "signature_verified", "execution_enabled", "review_required"],
                "additionalProperties": False,
            },
            "successEvidence": [{
                "id": "signed-skill-candidate-retrieved",
                "assertion": "installation_signature_and_content_digest_are_verified_and_candidate_remains_disabled_pending_human_review",
                "acceptedTypes": ["state_assertion"],
            }],
            # The implementation reads a persisted AgentMesh/MedAI registry
            # record; use AgentMesh's state effect vocabulary for that access.
            "effects": ["state_read"],
            "idempotency": "guaranteed",
            "recovery": "retry",
        },
        "authority": {"permissions": ["biomedical.skills.read"], "approvals": [], "constraints": {}},
        "implementations": [{"id": "medai-signed-skill-reader", "kind": "mcp", "reference": "mcp://medai/medai_skills_get_candidate", "configuration": {}}],
        "x-medai-learned-skill": {
            "skill": skill,
            "skillSha256": skill_digest,
            "procedureStepCount": len(procedure),
            "lifecycle": "pending_review",
            "executionEnabled": False,
            "exportSemantics": "review package only; importing this AMCP contract does not approve or execute the learned procedure",
        },
    }
    return package


def render_agentmesh_skill_review_package(skill: dict[str, Any], signing_key: bytes | None = None) -> tuple[str, str]:
    package = build_agentmesh_skill_review_package(skill, signing_key)
    text = yaml.safe_dump(package, sort_keys=False, allow_unicode=True, width=1000)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def register_agentmesh_skill_review_package(
    skill: dict[str, Any],
    catalog_dir: str | Path,
    agentmesh_cli: str | None = None,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Validate then install a read-only review package in a chosen local AgentMesh catalog."""
    executable = agentmesh_cli or shutil.which("agentmesh")
    if not executable:
        raise ValueError("AgentMesh CLI is required; set AGENTMESH_CLI or add agentmesh to PATH")
    package_yaml, package_sha256 = render_agentmesh_skill_review_package(skill, signing_key)
    catalog = Path(catalog_dir).expanduser().resolve()
    catalog.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".medai-skill-amcp-", dir=catalog) as temporary:
        package_path = Path(temporary) / "skill-review.yaml"
        package_path.write_text(package_yaml, encoding="utf-8")
        package_path.chmod(0o600)
        for command in (
            [executable, "capabilities", "validate", str(package_path)],
            [executable, "capabilities", "install", "--catalog", str(catalog), str(package_path)],
        ):
            try:
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30, shell=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ValueError(f"AgentMesh CLI operation failed: {error}") from error
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()[:2000]
                raise ValueError(f"AgentMesh capability registration failed: {detail}")
        return {
            "catalog": str(catalog),
            "package_sha256": package_sha256,
            "capability_id": "biomedical.skill." + skill["id"].replace("_", "-"),
            "version": skill["version"],
            "validated": True,
            "installed": True,
            "review_required": True,
            "execution_enabled": False,
            "agentmesh_validation": result.stdout.strip()[:1000],
        }
