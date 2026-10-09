"""Request-scoped specialist authentication for the Streamable HTTP MCP server."""
from __future__ import annotations

import hmac
import json
import os
from contextvars import ContextVar
from typing import Any

import httpx

API = os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090").rstrip("/")
_API_TOKEN = os.getenv("MEDAI_API_BEARER_TOKEN", "")
_API_HEADERS = {"Authorization": f"Bearer {_API_TOKEN}"} if _API_TOKEN else {}
MCP_TRANSPORT_HEADER = b"x-medai-mcp-token"
_REQUEST_API_HEADERS: ContextVar[dict[str, str] | None] = ContextVar("medai_request_api_headers", default=None)


def api_headers() -> dict[str, str]:
    """Return the credential bound to this MCP request, or the service credential."""
    return _REQUEST_API_HEADERS.get() or _API_HEADERS


class SpecialistCredentialASGIMiddleware:
    """Bind each MCP tool request to its authenticated specialist API token."""

    max_request_bytes = 16 * 1024 * 1024

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_request_bytes:
                await self._respond(send, 413, b"MCP request exceeds the configured byte limit")
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        token, expected_agent = self._specialist_metadata(body)
        if expected_agent is not None and token is None:
            await self._respond(send, 401, b"specialist identity metadata requires an API bearer")
            return
        if token is None:
            await self._replay(scope, receive, send, body)
            return
        if len(token) < 32 or len(token) > 4096 or any(char.isspace() for char in token):
            await self._respond(send, 401, b"invalid specialist API credential")
            return
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with httpx.AsyncClient(base_url=API, headers=headers, timeout=5, follow_redirects=False) as client:
                response = await client.get("/v1/auth/whoami")
                if response.status_code != 200:
                    await self._respond(send, 401, b"specialist API credential was rejected")
                    return
                identity = response.json()
        except (httpx.HTTPError, ValueError):
            await self._respond(send, 503, b"specialist identity service is unavailable")
            return
        if (
            not isinstance(identity, dict)
            or identity.get("role") != "service"
            or not identity.get("scientific_agent_id")
            or identity.get("scientific_agent_id") != expected_agent
        ):
            await self._respond(send, 403, b"API credential is not bound to the requested specialist")
            return
        request_headers = _REQUEST_API_HEADERS.set(headers)
        try:
            await self._replay(scope, receive, send, body)
        finally:
            _REQUEST_API_HEADERS.reset(request_headers)

    @staticmethod
    def _specialist_metadata(body: bytes) -> tuple[str | None, str | None]:
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, None
        if not isinstance(request, dict) or request.get("method") != "tools/call":
            return None, None
        params = request.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict):
            return None, None
        token = meta.get("io.medai/specialist-api-bearer")
        agent_id = meta.get("io.medai/scientific-agent")
        if token is not None and not isinstance(token, str):
            return "", str(agent_id) if agent_id is not None else None
        if agent_id is not None and not isinstance(agent_id, str):
            return token if isinstance(token, str) else None, ""
        return token, agent_id

    @staticmethod
    async def _respond(send: Any, status: int, body: bytes) -> None:
        headers = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def _replay(self, scope: dict, receive: Any, send: Any, body: bytes) -> None:
        delivered = False

        async def replay_receive() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class AuthenticatedMCPTransportASGIMiddleware:
    """Require the distinct MCP transport bearer on HTTP ingress."""

    def __init__(self, app: Any, token: str | None = None):
        self.app = app
        configured = token if token is not None else os.getenv("MEDAI_MCP_BEARER_TOKEN", "")
        self.token = _validated_transport_token(configured)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope_type not in {"http", "websocket"}:
            return
        supplied = None
        credential_headers = 0
        for name, value in scope.get("headers", []):
            if name.lower() == MCP_TRANSPORT_HEADER:
                credential_headers += 1
                try:
                    supplied = value.decode("ascii", errors="strict")
                except UnicodeDecodeError:
                    supplied = ""
        candidate = supplied or ""
        matches = hmac.compare_digest(candidate.encode("utf-8"), self.token.encode("utf-8"))
        if credential_headers != 1 or not matches:
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 4401})
            else:
                await SpecialistCredentialASGIMiddleware._respond(send, 401, b"MCP transport authentication required")
            return
        safe_scope = dict(scope)
        safe_scope["headers"] = [
            (name, value)
            for name, value in scope.get("headers", [])
            if name.lower() not in {MCP_TRANSPORT_HEADER, b"authorization", b"cookie"}
        ]
        await self.app(safe_scope, receive, send)


def _validated_transport_token(token: str) -> str:
    if len(token) < 32 or len(token) > 4096 or any(char.isspace() for char in token):
        raise ValueError("MEDAI_MCP_BEARER_TOKEN must contain 32..4096 non-whitespace characters")
    if token == _API_TOKEN:
        raise ValueError("MCP transport credential must be distinct from the API service credential")
    return token
