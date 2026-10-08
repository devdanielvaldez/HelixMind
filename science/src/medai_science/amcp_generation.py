"""Build review-only AgentMesh packages for signed generated scientific proposals."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from medai_science.contract_validation import capability_io_schemas


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def build_generated_capability_amcp_package(
    proposal: dict[str, Any],
    *,
    source_tenant_id: str,
    source_session_id: str,
    proposal_artifact_id: str,
    proposal_artifact_sha256: str,
) -> dict[str, Any]:
    """Create a deterministic, execution-disabled AMCP review package bound to one proposal."""
    if not isinstance(proposal, dict):
        raise ValueError("generated capability proposal must be an object")
    manifest = proposal.get("manifest")
    contract = proposal.get("contract")
    language = proposal.get("language", "python")
    if not isinstance(manifest, dict) or not isinstance(contract, dict):
        raise ValueError("generated proposal must retain its manifest and contract")
    if language not in {"python", "r"}:
        raise ValueError("generated proposal language must be python or r")
    proposal_capability_id, version = manifest.get("id"), manifest.get("version")
    if not isinstance(proposal_capability_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", proposal_capability_id):
        raise ValueError("generated capability has an invalid stable ID")
    readable_id = re.sub(r"[^a-z0-9-]+", "-", proposal_capability_id.replace(".", "-").replace("_", "-")).strip("-")[:72]
    capability_id = "biomedical.generated." + readable_id + "-" + hashlib.sha256(proposal_capability_id.encode()).hexdigest()[:16]
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("generated capability has an invalid semantic version")
    if any(not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) for value in (source_tenant_id, source_session_id, proposal_artifact_id)):
        raise ValueError("proposal tenant, session, and artifact IDs must be stable identifiers")
    if not isinstance(proposal_artifact_sha256, str) or not _DIGEST.fullmatch(proposal_artifact_sha256):
        raise ValueError("proposal artifact digest must be a lowercase SHA-256")
    input_schema, output_schema = capability_io_schemas(contract)
    intent = manifest.get("description")
    if not isinstance(intent, str) or not intent.strip() or len(intent) > 2000:
        raise ValueError("generated capability description must contain 1..2000 characters")

    package = {
        "schemaVersion": "amcp/0.1",
        "capability": {
            "id": capability_id,
            "version": version,
            "intent": (
                f"Execute the reviewed {language.upper()} capability {capability_id}@{version} in MedAI's "
                "bounded no-network sandbox against its retained input/output contract. "
                "Execution requires production lifecycle approval and an exact active AgentMesh route."
            ),
        },
        "contract": {
            "inputs": {
                "type": "object",
                "required": ["session_id", "task_id", "source_tenant_id", "source_session_id", "proposal_artifact_id", "proposal_artifact_sha256", "proposal_capability_id", "capability_id", "version", "input_data"],
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "source_tenant_id": {"type": "string", "minLength": 1, "maxLength": 128, "const": source_tenant_id},
                    "source_session_id": {"type": "string", "minLength": 1, "maxLength": 128, "const": source_session_id},
                    "proposal_artifact_id": {"type": "string", "minLength": 1, "maxLength": 128, "const": proposal_artifact_id},
                    "proposal_artifact_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$", "const": proposal_artifact_sha256},
                    "proposal_capability_id": {"type": "string", "const": proposal_capability_id},
                    "capability_id": {"type": "string", "const": capability_id},
                    "version": {"type": "string", "const": version},
                    "input_data": input_schema,
                },
            },
            "outputs": {
                "type": "object",
                "required": ["available", "capability_id", "version", "status", "execution_receipt_sha256", "validation_performed", "registered", "promoted"],
                "properties": {
                    "available": {"type": "boolean"},
                    "capability_id": {"type": "string", "const": capability_id},
                    "version": {"type": "string", "const": version},
                    "result": output_schema,
                    "status": {"type": "string", "enum": ["completed", "failed"]},
                    "execution_receipt_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "validation_performed": {"type": "boolean", "const": False},
                    "registered": {"type": "boolean", "const": False},
                    "promoted": {"type": "boolean", "const": False},
                },
            },
            "successEvidence": [{
                "id": "approved-generated-capability-execution",
                "assertion": "exact_production_lifecycle_active_agentmesh_route_signed_proposal_hash_contract_and_metered_sandbox_receipt_are_bound",
                "acceptedTypes": ["provider_receipt", "state_assertion"],
            }],
        },
        "effects": ["external_communication", "data_write"],
        "idempotency": "not_guaranteed",
        "recovery": "reconcile",
        "authority": {"permissions": ["biomedical.capabilities.execute", "biomedical.artifacts.write"]},
        "implementations": [{
            "id": "medai-generated-" + language + "-" + hashlib.sha256(f"{capability_id}@{version}".encode()).hexdigest()[:16],
            "kind": "mcp",
            "reference": "mcp://medai/medai_capabilities_execute_generated",
        }],
    }
    try:
        encoded = json.dumps(package, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("generated AMCP package must contain finite JSON values") from error
    if len(encoded) > 1_000_000:
        raise ValueError("generated AMCP package exceeds 1 MiB")
    return package
