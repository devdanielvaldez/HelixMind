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


if __name__ == "__main__":
    unittest.main()
