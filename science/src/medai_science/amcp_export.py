"""Export verified generated AMCP proposals for operator review and AgentMesh validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import httpx
import yaml

from medai_science.amcp_generation import build_generated_capability_amcp_package
from medai_science.capability_lifecycle import fetch_capability_registry
from medai_science.provenance_signing import verify_capability_proposal_signature
from medai_science.sandbox import validate_python_source
from medai_science.r_sandbox import validate_r_source


_MAX_ARTIFACT_BYTES = 1_048_576
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TENANT_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _decode_artifact(artifact: Any, *, expected_session_id: str, expected_kind: str) -> dict[str, Any]:
    if not isinstance(artifact, dict) or artifact.get("session_id") != expected_session_id or artifact.get("kind") != expected_kind:
        raise ValueError(f"expected a {expected_kind} artifact in the selected session")
    payload, digest = artifact.get("payload"), artifact.get("sha256")
    if not isinstance(payload, str) or not isinstance(digest, str) or len(payload.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
        raise ValueError("generated AMCP artifact payload is malformed or exceeds 1 MiB")
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
        raise ValueError("generated AMCP artifact payload hash verification failed")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("generated AMCP artifact payload is not JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("generated AMCP artifact payload must contain a JSON object")
    return decoded


def export_generated_amcp_artifact(
    api_url: str,
    bearer_token: str,
    session_id: str,
    amcp_artifact_id: str,
    output_path: str | Path,
    *,
    transport: httpx.BaseTransport | None = None,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Verify both linked artifacts and write one exclusive, private AMCP review package."""
    if any(not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) for value in (session_id, amcp_artifact_id)):
        raise ValueError("session and artifact IDs must be stable identifiers")
    # This also validates the API origin and credential policy before any artifact request.
    fetch_capability_registry(api_url, bearer_token, transport)
    with httpx.Client(
        base_url=api_url.rstrip("/"),
        headers={"Authorization": f"Bearer {bearer_token}", "Accept": "application/json"},
        timeout=15,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        artifact_response = client.get(f"/v1/research/{session_id}/artifacts/{amcp_artifact_id}")
        artifact_response.raise_for_status()
        amcp_artifact = artifact_response.json()
        amcp_proposal = _decode_artifact(amcp_artifact, expected_session_id=session_id, expected_kind="generated_capability_amcp_proposal")
        source_session = amcp_proposal.get("source_session_id")
        source_artifact_id = amcp_proposal.get("source_proposal_artifact_id")
        source_tenant = amcp_proposal.get("source_tenant_id")
        source_digest = amcp_proposal.get("source_proposal_artifact_sha256")
        if source_session != session_id or not isinstance(source_artifact_id, str) or not isinstance(source_tenant, str):
            raise ValueError("AMCP proposal source identity is malformed or cross-session")
        if not _IDENTIFIER.fullmatch(source_artifact_id) or not _IDENTIFIER.fullmatch(source_tenant):
            raise ValueError("AMCP proposal source tenant or artifact ID is malformed")
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise ValueError("AMCP proposal source artifact digest is malformed")
        source_response = client.get(f"/v1/research/{source_session}/artifacts/{source_artifact_id}")
        source_response.raise_for_status()
        source_artifact = source_response.json()
    if not isinstance(source_artifact, dict) or source_artifact.get("sha256") != source_digest:
        raise ValueError("AMCP proposal does not bind the fetched source proposal digest")
    language = amcp_proposal.get("language", "python")
    if language not in {"python", "r"}:
        raise ValueError("AMCP proposal declares an unsupported generated source language")
    expected_source_kind = "generated_r_capability_proposal" if language == "r" else "generated_capability_proposal"
    source_proposal = _decode_artifact(source_artifact, expected_session_id=session_id, expected_kind=expected_source_kind)
    if source_proposal.get("language", "python") != language:
        raise ValueError("AMCP language does not match the signed source proposal")
    signature = verify_capability_proposal_signature(source_proposal, signing_key)
    if not signature.get("verified"):
        raise ValueError("generated proposal signature is not verified")
    source_hash = source_proposal.get("static_review", {}).get("source_sha256")
    code = source_proposal.get("source")
    source_review = validate_r_source(code) if language == "r" else validate_python_source(code)
    if source_review["source_sha256"] != source_hash or source_hash != amcp_proposal.get("source_sha256"):
        raise ValueError("AMCP proposal source hash does not match its signed generated proposal")
    expected_package = build_generated_capability_amcp_package(
        source_proposal,
        source_tenant_id=source_tenant,
        source_session_id=session_id,
        proposal_artifact_id=source_artifact_id,
        proposal_artifact_sha256=source_digest,
    )
    package = amcp_proposal.get("package")
    if package != expected_package:
        raise ValueError("generated AMCP package differs from its deterministic signed-proposal binding")
    if amcp_proposal.get("execution") != {"enabled": False, "lifecycle": "pending_review"} or amcp_proposal.get("registered") is not False or amcp_proposal.get("promoted") is not False:
        raise ValueError("generated AMCP proposal must remain disabled and pending review")

    encoded = yaml.safe_dump(package, sort_keys=False, allow_unicode=True, width=1000).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("generated AMCP YAML exceeds 1 MiB")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
        output.chmod(0o600)
    except FileExistsError as error:
        raise ValueError("refusing to overwrite an existing AMCP package") from error
    except OSError:
        try:
            output.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    capability = expected_package["capability"]
    return {
        "output": str(output.resolve()),
        "package_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_proposal_artifact_id": source_artifact_id,
        "source_proposal_artifact_sha256": source_digest,
        "capability_id": capability["id"],
        "version": capability["version"],
        "review_required": True,
        "execution_enabled": False,
        "registered": False,
        "promoted": False,
    }
