import hashlib
import json
import unittest

import httpx

from medai_science.sources.dbsnp import DbSNPClient


class DbSNPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.body = json.dumps({"refsnp_id": 328, "primary_snapshot_data": {"placements_with_allele": []}}).encode()
        self.requests = []

        async def handler(request):
            self.requests.append(request)
            return httpx.Response(200, content=self.body, headers={"Content-Type": "application/json"})

        self.client = DbSNPClient(transport=httpx.MockTransport(handler))
        self.client._pace = lambda: _no_wait()

    def test_normalizes_only_rs_identifier_spelling(self):
        self.assertEqual(DbSNPClient.normalize_rsid("RS00328"), "rs328")
        for value in ("rs0", "rs-1", "328.2", "rs1234567890123", "rs1/../2"):
            with self.assertRaises(ValueError):
                DbSNPClient.normalize_rsid(value)

    async def test_fetches_exact_record_with_source_digest_and_interpretation_limit(self):
        result = await self.client.lookup("RS328")
        self.assertEqual(self.requests[0].method, "GET")
        self.assertEqual(str(self.requests[0].url), "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/328")
        self.assertEqual(result["rsid"], "rs328")
        self.assertEqual(result["source_record_sha256"], hashlib.sha256(self.body).hexdigest())
        self.assertEqual(result["source_record"]["refsnp_id"], 328)
        self.assertIn("not assumed to use GRCh37 or GRCh38", result["interpretation_limit"])

    async def test_rejects_mismatched_identity_duplicate_keys_and_non_finite_numbers(self):
        for body in (
            b'{"refsnp_id":329}',
            b'{"refsnp_id":328,"refsnp_id":328}',
            b'{"refsnp_id":328,"frequency":NaN}',
            b'[]',
        ):
            self.body = body
            with self.subTest(body=body), self.assertRaises(ValueError):
                await self.client.lookup("rs328")

    async def test_rejects_redirects_and_oversized_responses(self):
        async def redirect(request):
            return httpx.Response(302, headers={"Location": "https://example.org/"})

        client = DbSNPClient(transport=httpx.MockTransport(redirect))
        client._pace = lambda: _no_wait()
        with self.assertRaises(httpx.HTTPStatusError):
            await client.lookup("rs328")

        async def oversized(request):
            return httpx.Response(200, content=b" " * (DbSNPClient.max_response_bytes + 1))

        client = DbSNPClient(transport=httpx.MockTransport(oversized))
        client._pace = lambda: _no_wait()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            await client.lookup("rs328")


async def _no_wait():
    return None
