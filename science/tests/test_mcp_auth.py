import json
import unittest
from unittest.mock import patch

from medai_science import mcp_auth


class IdentityResponse:
    status_code = 200

    def __init__(self, identity):
        self.identity = identity

    def json(self):
        return self.identity


class IdentityClient:
    identity = {"role": "service", "scientific_agent_id": "literature_librarian"}
    seen_headers = None

    def __init__(self, **kwargs):
        type(self).seen_headers = kwargs["headers"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def get(self, path):
        if path != "/v1/auth/whoami":
            raise AssertionError(path)
        return IdentityResponse(self.identity)


class SpecialistCredentialMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def make_request(self, metadata):
        return json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "medai_research_claim_tasks", "arguments": {}, "_meta": metadata},
        }).encode()

    async def run_middleware(self, metadata):
        body = self.make_request(metadata)
        calls = []
        messages = []

        async def app(scope, receive, send):
            event = await receive()
            calls.append((json.loads(event["body"]), mcp_auth.api_headers()))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        received = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        with patch("medai_science.mcp_auth.httpx.AsyncClient", IdentityClient):
            await mcp_auth.SpecialistCredentialASGIMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/mcp"}, receive, send
            )
        return calls, messages

    async def test_scoped_bearer_is_validated_and_bound_for_exact_request(self):
        token = "a" * 32
        calls, messages = await self.run_middleware({
            "io.medai/specialist-api-bearer": token,
            "io.medai/scientific-agent": "literature_librarian",
        })
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["method"], "tools/call")
        self.assertEqual(calls[0][1], {"Authorization": f"Bearer {token}"})
        self.assertEqual(messages[0]["status"], 200)
        self.assertIsNone(mcp_auth._REQUEST_API_HEADERS.get())

    async def test_rejects_missing_and_mismatched_specialist_identity(self):
        calls, missing = await self.run_middleware({"io.medai/scientific-agent": "literature_librarian"})
        self.assertEqual(calls, [])
        self.assertEqual(missing[0]["status"], 401)

        IdentityClient.identity = {"role": "service", "scientific_agent_id": "genomics_analyst"}
        try:
            calls, mismatched = await self.run_middleware({
                "io.medai/specialist-api-bearer": "b" * 32,
                "io.medai/scientific-agent": "literature_librarian",
            })
        finally:
            IdentityClient.identity = {"role": "service", "scientific_agent_id": "literature_librarian"}
        self.assertEqual(calls, [])
        self.assertEqual(mismatched[0]["status"], 403)


class AuthenticatedTransportMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, headers, expected_token="m" * 48, scope_type="http"):
        called = []
        messages = []
        forwarded_headers = []

        async def app(scope, receive, send):
            called.append(True)
            forwarded_headers.extend(scope["headers"])
            if scope["type"] == "websocket":
                await send({"type": "websocket.accept"})
            else:
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        middleware = mcp_auth.AuthenticatedMCPTransportASGIMiddleware(app, expected_token)
        scope = {"type": scope_type, "method": "POST", "path": "/mcp", "headers": headers}
        await middleware(scope, receive, send)
        return called, messages, forwarded_headers

    async def test_rejects_anonymous_and_caller_authorization_without_gateway_credential(self):
        for headers in ([], [(b"authorization", b"Bearer caller-token")]):
            called, messages, _ = await self.invoke(headers)
            self.assertEqual(called, [])
            self.assertEqual(messages[0]["status"], 401)

    async def test_websocket_transport_cannot_bypass_authentication(self):
        called, rejected, _ = await self.invoke([], scope_type="websocket")
        self.assertEqual(called, [])
        self.assertEqual(rejected[0], {"type": "websocket.close", "code": 4401})

        token = "m" * 48
        called, accepted, forwarded = await self.invoke(
            [(b"x-medai-mcp-token", token.encode())], token, "websocket"
        )
        self.assertEqual(called, [True])
        self.assertEqual(accepted[0]["type"], "websocket.accept")
        self.assertEqual(forwarded, [])

    async def test_accepts_only_the_distinct_gateway_injected_transport_credential(self):
        token = "m" * 48
        called, messages, forwarded = await self.invoke([
            (b"x-medai-mcp-token", token.encode()),
            (b"authorization", b"Bearer untrusted-caller-credential"),
            (b"cookie", b"session=private"),
        ], token)
        self.assertEqual(called, [True])
        self.assertEqual(messages[0]["status"], 200)
        self.assertEqual(forwarded, [])

        called, duplicate, _ = await self.invoke([
            (b"x-medai-mcp-token", token.encode()),
            (b"x-medai-mcp-token", token.encode()),
        ], token)
        self.assertEqual(called, [])
        self.assertEqual(duplicate[0]["status"], 401)

    async def test_rejects_invalid_or_reused_api_credentials_at_startup(self):
        for token in ("short", "a" * 5000):
            with self.assertRaises(ValueError):
                mcp_auth.AuthenticatedMCPTransportASGIMiddleware(lambda *_: None, token)
        with patch.object(mcp_auth, "_API_TOKEN", "a" * 48):
            with self.assertRaisesRegex(ValueError, "distinct"):
                mcp_auth.AuthenticatedMCPTransportASGIMiddleware(lambda *_: None, "a" * 48)


if __name__ == "__main__":
    unittest.main()
