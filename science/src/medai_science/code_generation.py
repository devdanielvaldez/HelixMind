"""Provider-neutral drafting of disabled Python capability proposals."""
from __future__ import annotations

import json
import re
from typing import Any

from medai_science.capabilities import draft_manifest
from medai_science.contract_validation import capability_io_schemas
from medai_science.llm import JSONModelProvider, ModelCompletion
from medai_science.r_sandbox import validate_r_source
from medai_science.sandbox import validate_python_source
from medai_science.sources.safety import ensure_public_research_query

MAX_SPECIFICATION_BYTES = 16_384
MAX_DESCRIPTION_BYTES = 2_000
MAX_PROPOSAL_BYTES = 256 * 1024 - 4_096

_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "minLength": 1, "maxLength": 262144},
        "implementation_summary": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    "required": ["source", "implementation_summary"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You draft experimental biomedical computational Python capability source code. "
    "Return only the requested JSON. Write a deterministic module with one top-level "
    "synchronous run(input: dict) -> dict function. Use only Python standard-library "
    "imports from math, statistics, re, json, collections, itertools, functools, decimal, "
    "fractions, or explicitly requested installed scientific packages numpy, scipy, Bio, "
    "or rdkit. Do not access files, environment variables, processes, network, model APIs, "
    "or hidden data. Do not use str.format or format_map; use f-strings for interpolation. "
    "Do not make clinical decisions or claim validation. The output source "
    "will remain disabled pending sandbox checks, independent scientific test cases, and review."
)

_SYSTEM_R = (
    "You draft experimental biomedical computational R capability source code. "
    "Return only the requested JSON. Write a deterministic R file with exactly one "
    "entry point `run <- function(input)` that returns a named list representing a JSON object. "
    "Use base R and only explicitly requested installed scientific packages via package-qualified "
    "calls. Do not access files, environment variables, processes, network, model APIs, or hidden "
    "data; do not use dynamic evaluation. Do not make clinical decisions or claim validation. "
    "The output source will remain disabled pending sandbox checks, independent scientific test "
    "cases, and review."
)


def _validate_public_specification(value: Any, path: str = "$", depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("capability specification exceeds 32 nesting levels")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
            raise ValueError(f"capability specification text exceeds {MAX_DESCRIPTION_BYTES} bytes at {path}")
        ensure_public_research_query(value)
    elif isinstance(value, dict):
        if len(value) > 256:
            raise ValueError(f"capability specification contains too many fields at {path}")
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"capability specification keys must be strings at {path}")
            _validate_public_specification(key, f"{path}.<key>", depth + 1)
            _validate_public_specification(child, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        if len(value) > 256:
            raise ValueError(f"capability specification contains too many list items at {path}")
        for index, child in enumerate(value):
            _validate_public_specification(child, f"{path}[{index}]", depth + 1)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError(f"unsupported capability specification value at {path}")


def validate_python_capability_request(
    *,
    capability_id: str,
    description: str,
    source_session: str,
    contract: dict[str, Any],
    implementation_requirements: list[str],
) -> tuple[dict[str, Any], str]:
    """Validate caller input before charging or sending a model request."""
    if not isinstance(description, str) or not description.strip() or len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        raise ValueError("capability description must contain 1..2000 UTF-8 bytes")
    if not isinstance(capability_id, str) or len(capability_id) > 128:
        raise ValueError("capability id must contain at most 128 characters")
    if not isinstance(source_session, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", source_session):
        raise ValueError("source session must be a stable 1..128 character identifier")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("capability contract must be a non-empty JSON object")
    capability_io_schemas(contract)
    if not isinstance(implementation_requirements, list) or not implementation_requirements or any(not isinstance(item, str) or not item.strip() for item in implementation_requirements):
        raise ValueError("provide one or more explicit implementation requirements")
    _validate_public_specification({"description": description, "contract": contract, "requirements": implementation_requirements})
    try:
        serialized = json.dumps({"contract": contract, "requirements": implementation_requirements}, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("capability contract and requirements must be finite JSON values") from error
    if len(serialized.encode("utf-8")) > MAX_SPECIFICATION_BYTES:
        raise ValueError(f"capability specification exceeds {MAX_SPECIFICATION_BYTES} bytes")
    manifest = draft_manifest(capability_id, description.strip(), source_session)
    prompt = (
        "Draft the implementation for this disabled capability. The contract is authoritative; "
        "do not change it. The listed requirements describe behavior, not scientific test cases.\n"
        f"Capability: {capability_id}\nDescription: {description.strip()}\n"
        f"Contract and implementation requirements JSON:\n{serialized}"
    )
    return manifest, prompt


async def generate_python_capability(
    provider: JSONModelProvider,
    *,
    capability_id: str,
    description: str,
    source_session: str,
    contract: dict[str, Any],
    implementation_requirements: list[str],
) -> tuple[dict[str, Any], ModelCompletion]:
    """Generate and statically inspect code; never execute or register it."""
    manifest, prompt = validate_python_capability_request(
        capability_id=capability_id,
        description=description,
        source_session=source_session,
        contract=contract,
        implementation_requirements=implementation_requirements,
    )
    completion = await provider.complete_json(
        system=_SYSTEM,
        prompt=prompt,
        schema_name="medai_python_capability_source",
        schema=_SOURCE_SCHEMA,
    )
    source = completion.value.get("source")
    summary = completion.value.get("implementation_summary")
    if not isinstance(source, str) or not isinstance(summary, str):
        raise ValueError("model capability draft omitted source or implementation summary")
    static_review = validate_python_source(source)
    proposal = {
        "schema_version": "1",
        "manifest": manifest,
        "contract": contract,
        "source": source,
        "implementation_summary": summary,
        "generation": {
            "provider": completion.provider,
            "model": completion.model,
            "max_output_tokens": completion.max_output_tokens,
            "prompt_sha256": completion.prompt_sha256,
            "response_sha256": completion.response_sha256,
            "usage": completion.usage,
        },
        "static_review": static_review,
        "validation": {
            "security": "static checks passed; sandbox execution pending",
            "scientific": "independent expected-output cases pending",
            "review_required": True,
        },
        "execution": {"enabled": False, "network": "deny", "filesystem": "read_only"},
        "lifecycle_status": "experimental_pending_review",
        "limitations": [
            "Static checks do not establish safety or scientific correctness.",
            "Generated code has not been executed; use the pinned-image sandbox with independently authored test cases.",
            "The proposal is not registered, enabled, or promoted by this operation.",
        ],
    }
    serialized_proposal = json.dumps(proposal, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(serialized_proposal) > MAX_PROPOSAL_BYTES:
        raise ValueError(f"generated capability proposal exceeds the {MAX_PROPOSAL_BYTES}-byte artifact payload limit")
    return proposal, completion


async def generate_r_capability(
    provider: JSONModelProvider,
    *,
    capability_id: str,
    description: str,
    source_session: str,
    contract: dict[str, Any],
    implementation_requirements: list[str],
) -> tuple[dict[str, Any], ModelCompletion]:
    """Draft a disabled R proposal and apply bounded static checks; never execute it."""
    manifest, _ = validate_python_capability_request(
        capability_id=capability_id,
        description=description,
        source_session=source_session,
        contract=contract,
        implementation_requirements=implementation_requirements,
    )
    serialized = json.dumps({"contract": contract, "requirements": implementation_requirements}, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    prompt = (
        "Draft the R implementation for this disabled capability. The JSON input/output contract is authoritative.\n"
        f"Capability: {capability_id}\nDescription: {description.strip()}\n"
        f"Contract and implementation requirements JSON:\n{serialized}"
    )
    completion = await provider.complete_json(
        system=_SYSTEM_R,
        prompt=prompt,
        schema_name="medai_r_capability_source",
        schema=_SOURCE_SCHEMA,
    )
    source = completion.value.get("source")
    summary = completion.value.get("implementation_summary")
    if not isinstance(source, str) or not isinstance(summary, str):
        raise ValueError("model R capability draft omitted source or implementation summary")
    static_review = validate_r_source(source)
    proposal = {
        "schema_version": "1",
        "language": "r",
        "manifest": manifest,
        "contract": contract,
        "source": source,
        "implementation_summary": summary,
        "generation": {
            "provider": completion.provider,
            "model": completion.model,
            "max_output_tokens": completion.max_output_tokens,
            "prompt_sha256": completion.prompt_sha256,
            "response_sha256": completion.response_sha256,
            "usage": completion.usage,
        },
        "static_review": static_review,
        "validation": {
            "security": "bounded R entry-point checks passed; isolated sandbox execution pending",
            "scientific": "independent expected-output cases pending",
            "review_required": True,
        },
        "execution": {"enabled": False, "network": "deny", "filesystem": "read_only"},
        "lifecycle_status": "experimental_pending_review",
        "limitations": [
            "Static R checks are not a security boundary; execution requires the isolated pinned-image sandbox.",
            "Generated code has not been executed; use an R image containing Rscript and jsonlite with independently authored cases.",
            "The proposal is not registered, enabled, or promoted by this operation.",
        ],
    }
    serialized_proposal = json.dumps(proposal, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(serialized_proposal) > MAX_PROPOSAL_BYTES:
        raise ValueError(f"generated R capability proposal exceeds the {MAX_PROPOSAL_BYTES}-byte artifact payload limit")
    return proposal, completion
