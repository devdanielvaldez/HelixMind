"""Authenticated bridge from MedAI reviewer decisions to AgentMesh route plans."""
from __future__ import annotations

import hashlib
import ipaddress
import re
from urllib.parse import urlparse

import httpx


_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{1,200}\Z")
_STATUSES = {
    "pending_review": "experimental",
    "experimental": "experimental",
    "testing": "testing",
    "validating": "validating",
    "verified": "verified",
    "production": "production",
    "production_certified": "production_certified",
    "deprecated": "deprecated",
    "rolled_back": "rolled_back",
}


def fetch_approved_lifecycle(
    api_url: str,
    bearer_token: str,
    transport: httpx.BaseTransport | None = None,
    *,
    allow_insecure_hosts: set[str] | None = None,
) -> tuple[dict[str, str], str]:
    """Fetch capability lifecycle decisions from the authenticated MedAI API."""
    statuses, digest, _ = fetch_capability_registry(
        api_url, bearer_token, transport, allow_insecure_hosts=allow_insecure_hosts
    )
    return statuses, digest


def fetch_capability_registry(
    api_url: str,
    bearer_token: str,
    transport: httpx.BaseTransport | None = None,
    *,
    allow_insecure_hosts: set[str] | None = None,
) -> tuple[dict[str, str], str, list[dict]]:
    """Fetch one authenticated, content-hashed registry snapshot with package records."""
    parsed = urlparse(api_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MedAI API URL must be an absolute HTTP(S) origin without credentials or query")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if not loopback and parsed.hostname.casefold() not in (allow_insecure_hosts or set()):
            raise ValueError("cleartext remote MedAI API URLs are forbidden")
    if len(bearer_token) < 32 or any(character.isspace() for character in bearer_token):
        raise ValueError("MedAI lifecycle registry bearer must contain at least 32 non-whitespace bytes")
    with httpx.Client(
        base_url=api_url.rstrip("/"),
        headers={"Authorization": f"Bearer {bearer_token}", "Accept": "application/json"},
        timeout=10,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        response = client.get("/v1/capabilities")
        response.raise_for_status()
        raw = response.content
        if len(raw) > 2_000_000:
            raise ValueError("MedAI capability registry response exceeds 2 MiB")
        if "application/json" not in response.headers.get("content-type", "").casefold():
            raise ValueError("MedAI capability registry did not return JSON")
        payload = response.json()
    rows = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("MedAI capability registry response is malformed or exceeds 1000 records")
    statuses = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("MedAI capability registry contains a malformed record")
        capability_id, version, status = row.get("id"), row.get("version"), row.get("status")
        if (
            not isinstance(capability_id, str)
            or not _IDENTIFIER.fullmatch(capability_id)
            or not isinstance(version, str)
            or not _IDENTIFIER.fullmatch(version)
            or status not in _STATUSES
        ):
            raise ValueError("MedAI capability registry contains an invalid identity or lifecycle status")
        key = f"{capability_id}@{version}"
        if key in statuses:
            raise ValueError("MedAI capability registry contains duplicate identities")
        statuses[key] = _STATUSES[status]
    return statuses, hashlib.sha256(raw).hexdigest(), rows
