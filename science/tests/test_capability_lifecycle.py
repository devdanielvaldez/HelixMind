import hashlib
import unittest

import httpx

from medai_science.capability_lifecycle import fetch_approved_lifecycle


TOKEN = "a" * 40


class CapabilityLifecycleBridgeTests(unittest.TestCase):
    def test_fetches_authenticated_registry_and_maps_review_state(self):
        body = b'{"items":[{"id":"biomedical.genomics","version":"1.0.0","status":"verified"},{"id":"biomedical.draft","version":"0.1.0","status":"pending_review"}]}'
        seen = {}

        def handler(request):
            seen["authorization"] = request.headers["authorization"]
            seen["path"] = request.url.path
            return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

        statuses, digest = fetch_approved_lifecycle("https://medai.example", TOKEN, httpx.MockTransport(handler))
        self.assertEqual(seen, {"authorization": f"Bearer {TOKEN}", "path": "/v1/capabilities"})
        self.assertEqual(statuses, {"biomedical.genomics@1.0.0": "verified", "biomedical.draft@0.1.0": "experimental"})
        self.assertEqual(digest, hashlib.sha256(body).hexdigest())

    def test_rejects_invalid_or_duplicate_registry_rows(self):
        bodies = [
            b'{"items":[{"id":"bad id","version":"1.0","status":"verified"}]}',
            b'{"items":[{"id":"x","version":"1","status":"unknown"}]}',
            b'{"items":[{"id":"x","version":"1","status":"verified"},{"id":"x","version":"1","status":"deprecated"}]}',
        ]
        for body in bodies:
            with self.subTest(body=body):
                transport = httpx.MockTransport(lambda request, content=body: httpx.Response(200, headers={"content-type": "application/json"}, content=content))
                with self.assertRaises(ValueError):
                    fetch_approved_lifecycle("https://medai.example", TOKEN, transport)

    def test_rejects_cleartext_remote_redirect_and_non_json(self):
        with self.assertRaisesRegex(ValueError, "cleartext remote"):
            fetch_approved_lifecycle("http://medai.example", TOKEN)
        redirect = httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": "https://evil.example"}))
        with self.assertRaises(httpx.HTTPStatusError):
            fetch_approved_lifecycle("https://medai.example", TOKEN, redirect)
        not_json = httpx.MockTransport(lambda request: httpx.Response(200, text="{}"))
        with self.assertRaisesRegex(ValueError, "did not return JSON"):
            fetch_approved_lifecycle("https://medai.example", TOKEN, not_json)

    def test_rejects_short_bearer(self):
        with self.assertRaisesRegex(ValueError, "at least 32"):
            fetch_approved_lifecycle("http://127.0.0.1:8090", "short")

    def test_allows_only_explicit_compose_api_alias_for_internal_http(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "application/json"}, json={"items": []})
        )
        statuses, _ = fetch_approved_lifecycle(
            "http://api:8090", TOKEN, transport, allow_insecure_hosts={"api"}
        )
        self.assertEqual(statuses, {})


if __name__ == "__main__":
    unittest.main()
